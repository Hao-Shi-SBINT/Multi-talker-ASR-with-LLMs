# separator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from entmax import entmax15  # optional: pip install entmax
    _HAS_ENTMAX = True
except Exception:
    _HAS_ENTMAX = False

# --- your custom modules (unchanged) ---
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_t, c_t):
        combined = torch.cat([x_t, h_t], dim=-1)
        gates = self.W(combined)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class StackedCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, use_layernorm=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(CustomLSTMCell(in_size, hidden_size))
            if use_layernorm:
                self.norms.append(nn.LayerNorm(hidden_size))

    def forward(self, x):
        # x: [B, T, input_size]
        B, T, _ = x.size()
        device = x.device
        h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # [B, input_size]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](x_t, h[l], c[l])
                x_t = h[l]
                if self.norms:
                    x_t = self.norms[l](x_t)
                x_t = self.dropout(x_t)
            outputs.append(x_t.unsqueeze(1))  # [B, 1, hidden_size]
        return torch.cat(outputs, dim=1)  # [B, T, hidden_size]


class MaskSeparator(nn.Module):
    """
    A robust frame-wise routing separator for multi-talker ASR.

    Pipeline:
      1) Pre-projection (optional) aligns encoder features to `hidden_size`
      2) Temporal core (StackedCustomLSTM or nn.LSTM) produces shared features y (B,T,H)
      3) Routing head outputs N-way logits per frame -> masks m(B,T,N)
      4) Head i output = y * m[..., i][:, :, None]

    This design encourages heads to "compete" for frames and prevents late-epoch starvation,
    especially for 3-speaker training.

    Args:
        in_dim (int):  input dim (encoder output size)
        hidden_size (int): internal temporal feature dim
        talker_numbers (int): number of heads/speakers N (>=2)
        num_layers (int): LSTM layers
        dropout (float): temporal dropout
        use_pre_ln (bool): LayerNorm after pre-projection
        use_post_ln (bool): LayerNorm after temporal core
        route (str): 'softmax' | 'topk' | 'entmax'
        temperature (float): softmax/entmax temperature (smaller -> sharper)
        topk_k (int): keep top-k heads per frame if route='topk'
        force_preproj (bool): apply pre-proj even if in_dim == hidden_size
        use_branch_ff (bool): per-head tiny FF to increase separability
        use_custom_lstm (bool): True to use your StackedCustomLSTM; False to use nn.LSTM
    """
    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        talker_numbers: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_pre_ln: bool = True,
        use_post_ln: bool = True,
        route: str = "topk",
        temperature: float = 0.7,
        topk_k: int = 2,
        force_preproj: bool = True,
        use_branch_ff: bool = True,
        use_custom_lstm: bool = True,
    ):
        super().__init__()
        assert talker_numbers >= 2, "talker_numbers must be >= 2"
        assert route in ("softmax", "topk", "entmax"), "route must be softmax/topk/entmax"

        self.N = int(talker_numbers)
        self.H = int(hidden_size)
        self.route = route
        self.tau = float(temperature)
        self.topk_k = int(topk_k)

        # 1) Pre-projection: learnable alignment from encoder space to hidden_size.
        need_proj = force_preproj or (in_dim != hidden_size)
        self.pre_proj = nn.Linear(in_dim, hidden_size) if need_proj else nn.Identity()
        self.pre_ln = nn.LayerNorm(hidden_size) if use_pre_ln else nn.Identity()

        # 2) Temporal core: choose your own LSTM or vanilla nn.LSTM.
        if use_custom_lstm:
            # Your custom implementation present in the repo:
            # StackedCustomLSTM(input_size, hidden_size, num_layers, dropout, use_layernorm=False)
            self.core = StackedCustomLSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_layernorm=False,
            )
        else:
            # Fallback to PyTorch LSTM (uncomment if desired)
            self.core = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )

        self.post_ln = nn.LayerNorm(hidden_size) if use_post_ln else nn.Identity()

        # 3) Routing head: frame-wise logits for N heads
        self.mask_logits = nn.Linear(hidden_size, self.N)

        # Optional: tiny per-head FF to further decorrelate head features
        self.use_branch_ff = use_branch_ff
        if use_branch_ff:
            self.branch_ff = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                ) for _ in range(self.N)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        # Conservative Xavier init to avoid early oscillation
        if isinstance(self.pre_proj, nn.Linear):
            nn.init.xavier_uniform_(self.pre_proj.weight)
            nn.init.zeros_(self.pre_proj.bias)
        if isinstance(self.mask_logits, nn.Linear):
            nn.init.xavier_uniform_(self.mask_logits.weight)
            nn.init.zeros_(self.mask_logits.bias)
        if self.use_branch_ff:
            for ff in self.branch_ff:
                for m in ff:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

    def _route_weights(self, L: torch.Tensor) -> torch.Tensor:
        """
        Convert logits L(B,T,N) -> masks m(B,T,N)
          - 'softmax': temperature-softmax over heads
          - 'topk'   : temperature-softmax then keep top-k heads per frame, renormalize
          - 'entmax' : entmax15 (sparse), requires entmax package
        """
        if self.route == "softmax":
            m = (L / self.tau).softmax(dim=-1)

        elif self.route == "topk":
            probs = (L / self.tau).softmax(dim=-1)     # (B,T,N)
            k = min(self.topk_k, probs.size(-1))
            topv, topi = probs.topk(k=k, dim=-1)       # (B,T,k)
            sparse = torch.zeros_like(probs).scatter_(-1, topi, topv)
            m = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        else:  # 'entmax'
            if not _HAS_ENTMAX:
                raise RuntimeError("route='entmax' requires: pip install entmax")
            m = entmax15(L / self.tau, dim=-1)

        return m

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): (B, T, in_dim) encoder features.

        Returns:
            outs (List[Tensor]): length-N; each is (B, T, hidden_size)
            m (Tensor): routing masks (B, T, N) for optional regularization/monitoring
        """
        # 1) Pre-projection + LN
        y = self.pre_proj(x)      # (B,T,H)
        y = self.pre_ln(y)

        # 2) Temporal core
        if isinstance(self.core, nn.LSTM):
            y, _ = self.core(y)   # (B,T,H)
        else:
            y = self.core(y)      # (B,T,H) from StackedCustomLSTM

        y = self.post_ln(y)       # (B,T,H)

        # 3) Routing weights
        L = self.mask_logits(y)   # (B,T,N)
        m = self._route_weights(L)

        # 4) Build per-head outputs
        outs = []
        if self.use_branch_ff:
            for i in range(self.N):
                yi = self.branch_ff[i](y) * m[..., i].unsqueeze(-1)
                outs.append(yi)
        else:
            for i in range(self.N):
                yi = y * m[..., i].unsqueeze(-1)
                outs.append(yi)

        return outs, m

