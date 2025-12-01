# separator.py
import torch
import torch.nn as nn

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


class Separator(nn.Module):
    """
    Pipeline:
      (1) Pre-projection: Linear(in_dim -> hidden_size) + optional activation + LayerNorm
      (2) StackedCustomLSTM(input_size=hidden_size, hidden_size=hidden_size, ...)
      (3) Post-LSTM LayerNorm
      (4) N symmetric branches (per head/private):
          MLP: Linear(hidden_size->hidden_size) + ReLU + Linear(hidden_size->in_dim) + ReLU
          (+ optional LayerNorm on in_dim)

    I/O:
      x: (B, T, in_dim) -> List[N] of (B, T, in_dim)

    NOTE:
      Each branch output dim = in_dim (same as your current behavior),
      so downstream CTC heads that expect in_dim will keep working.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        talker_numbers: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,              # per-time-step dropout inside your custom LSTM
        use_lstm_layernorm: bool = False,  # pass to StackedCustomLSTM
        proj_activation: str | None = "relu",  # 'relu' | 'gelu' | None
        use_branch_ln: bool = True,        # LayerNorm after each branch
        branch_dropout: float = 0.0,       # optional dropout inside each branch
        break_symmetry_eps: float = 1e-3,  # tiny bias offset to break symmetry
    ):
        super().__init__()
        assert talker_numbers >= 2, "talker_numbers must be >= 2"
        self.talker_numbers = talker_numbers
        self.hidden_size = hidden_size
        self.in_dim = in_dim

        # (1) Pre-projection to hidden_size
        self.pre_proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.pre_act = {"relu": nn.ReLU(), "gelu": nn.GELU(), None: nn.Identity()}[proj_activation]
        self.pre_ln  = nn.LayerNorm(hidden_size)

        # (2) Custom LSTM over hidden_size
        self.lstm = StackedCustomLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layernorm=use_lstm_layernorm,
        )

        # (3) Post-LSTM LayerNorm
        self.post_ln = nn.LayerNorm(hidden_size)

        # (4) Per-speaker/private branches
        def make_branch():
            layers = [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ]
            if branch_dropout and branch_dropout > 0:
                layers.append(nn.Dropout(branch_dropout))

            layers += [
                nn.Linear(hidden_size, in_dim),
                nn.ReLU(),
            ]
            if use_branch_ln:
                layers.append(nn.LayerNorm(in_dim))
            return nn.Sequential(*layers)

        self.sep_branches = nn.ModuleList([make_branch() for _ in range(talker_numbers)])

        # ---- init linears ----
        nn.init.xavier_uniform_(self.pre_proj.weight)
        nn.init.zeros_(self.pre_proj.bias)

        # branch init + tiny symmetry breaking
        for bi, m in enumerate(self.sep_branches):
            lin1 = m[0]
            lin2 = m[3] if isinstance(m[2], nn.Dropout) else m[2]  # handle optional dropout
            nn.init.xavier_uniform_(lin1.weight); nn.init.zeros_(lin1.bias)
            nn.init.xavier_uniform_(lin2.weight); nn.init.zeros_(lin2.bias)

            # break symmetry slightly so branches don't start identical
            if break_symmetry_eps and break_symmetry_eps > 0:
                lin2.bias.data += break_symmetry_eps * bi
        # ----------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            List[talker_numbers] of (B, T, in_dim)
        """
        y = self.pre_proj(x)      # (B, T, hidden_size)
        y = self.pre_act(y)
        y = self.pre_ln(y)

        y = self.lstm(y)          # (B, T, hidden_size)
        y = self.post_ln(y)       # (B, T, hidden_size)

        outs = [branch(y) for branch in self.sep_branches]  # N × (B, T, in_dim)
        return outs

