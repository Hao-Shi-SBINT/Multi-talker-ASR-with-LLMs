import torch
import torch.nn as nn
import torch.nn.functional as F

def _disable_lstm_flatten(module: nn.Module):
    # 将所有 nn.LSTM 的 flatten_parameters 置为 no-op，避免 shared storage
    for m in module.modules():
        if isinstance(m, nn.LSTM):
            m.flatten_parameters = lambda *a, **k: None

class StableSoftmaxSeparator(nn.Module):
    """
    A stable separator for multi-talker CTC:
      - Uses PyTorch's nn.LSTM as the temporal core (robust init & dropout handling).
      - Pure softmax routing (no top-k). Optional sharpening (gamma) and tiny-prob pruning (pmin).
      - Optional output alignment to match each head's expected CTC input dimension.

    Args:
        in_dim        : input feature dim from encoder (e.g., 1024)
        sep_dim       : internal separator hidden dim (e.g., 796)
        num_heads     : number of talkers / heads (2 or 3)
        num_layers    : LSTM layers (default 2)
        dropout       : LSTM inter-layer dropout (applied only between layers)
        bidirectional : use bi-LSTM; output dim stays sep_dim
        use_layernorm : apply LayerNorm before and after LSTM
        route_tau     : temperature for softmax routing (smaller -> sharper)
        route_gamma   : optional sharpening exponent; None means disabled
        route_pmin    : optional tiny-prob threshold; None means disabled
        out_align_dim : if not None and != sep_dim, add a linear per-head to map sep_dim -> out_align_dim
    """
    def __init__(
        self,
        in_dim: int,
        sep_dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_layernorm: bool = True,
        route_tau: float = 0.6,
        route_gamma: float | None = None,
        route_pmin: float | None = None,
        out_align_dim: int | None = None,
    ):
        super().__init__()
        assert num_heads >= 2, "num_heads must be >= 2"
        self.N = num_heads
        self.sep_dim = sep_dim

        # ----- projection + (optional) LN -----
        self.pre_proj = nn.Linear(in_dim, sep_dim)
        self.pre_ln   = nn.LayerNorm(sep_dim) if use_layernorm else nn.Identity()

        # ----- LSTM core (robust & simple) -----
        lstm_hidden = sep_dim // (2 if bidirectional else 1)
        assert lstm_hidden * (2 if bidirectional else 1) == sep_dim, \
            "sep_dim must be divisible by 2 when bidirectional=True"
        self.lstm = nn.LSTM(
            input_size=sep_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.post_ln = nn.LayerNorm(sep_dim) if use_layernorm else nn.Identity()

        # ----- routing logits -> N heads (softmax only) -----
        self.mask_logits = nn.Linear(sep_dim, self.N)
        self.route_tau   = route_tau
        self.route_gamma = route_gamma
        self.route_pmin  = route_pmin

        # ----- small head-specific FF (kept light to avoid overfitting) -----
        def make_branch():
            return nn.Sequential(
                nn.Linear(sep_dim, sep_dim),
                nn.ReLU(),
                nn.LayerNorm(sep_dim) if use_layernorm else nn.Identity(),
            )
        self.branch_ff = nn.ModuleList([make_branch() for _ in range(self.N)])

        # ----- optional per-head alignment to CTC input dim (e.g., 796 -> 1024) -----
        self.out_align_dim = out_align_dim
        if out_align_dim is not None and out_align_dim != sep_dim:
            self.align = nn.ModuleList([nn.Linear(sep_dim, out_align_dim) for _ in range(self.N)])
        else:
            self.align = None

        self._reset_linear_(self.pre_proj)
        self._reset_linear_(self.mask_logits)
        for b in self.branch_ff:
            b.apply(self._reset_linear_)
        if self.align is not None:
            for a in self.align:
                self._reset_linear_(a)

        _disable_lstm_flatten(self)

    @staticmethod
    def _reset_linear_(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _route_masks(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Pure softmax routing with optional sharpening / tiny-prob pruning.
        logits: (B, T, N) -> masks: (B, T, N) with sum over N == 1
        """
        masks = F.softmax(logits / self.route_tau, dim=-1)  # (B,T,N)

        if self.route_gamma is not None:
            masks = masks.pow(self.route_gamma)
            masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.route_pmin is not None and self.route_pmin > 0.0:
            masks = torch.where(masks >= self.route_pmin, masks, torch.zeros_like(masks))
            masks = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return masks

    def forward(self, feats: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Args:
            feats: (B, T, in_dim) encoder features
            attn_mask: optional (B, T) bool/0-1; if provided, it's only for potential future use/logging.
        Returns:
            outs:   list of per-head features, each (B, T, sep_dim) or (B, T, out_align_dim) if align enabled
            masks:  (B, T, N) soft routing masks (sum over N == 1)
            logits: (B, T, N) pre-softmax routing logits (for diagnostics)
            x:      (B, T, sep_dim) shared trunk features (after LSTM)
        """
        # Trunk
        x = self.pre_ln(self.pre_proj(feats))  # (B,T,sep_dim)
        x, _ = self.lstm(x)                    # (B,T,sep_dim)
        x = self.post_ln(x)                    # (B,T,sep_dim)

        # Routing
        logits = self.mask_logits(x)           # (B,T,N)
        masks  = self._route_masks(logits)     # (B,T,N)

        # Per-head branches
        outs = []
        for i in range(self.N):
            mi = masks[..., i].unsqueeze(-1)   # (B,T,1)
            xi = mi * x                        # (B,T,sep_dim)
            yi = self.branch_ff[i](xi)         # (B,T,sep_dim)
            if self.align is not None:
                yi = self.align[i](yi)         # (B,T,out_align_dim)
            outs.append(yi)

        return outs, masks, logits, x

