import numpy as np
import torch
import torch.nn as nn

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
    Multi-speaker separator with a shared trunk and N independent branches.
    Each branch: Linear -> ReLU -> LayerNorm -> Dropout -> (optional residual add)
    Output: List[Tensor] of length N, each with shape (B, T, H)
    """
    def __init__(
        self,
        hidden_size: int,
        talker_numbers: int,
        dropout: float = 0.2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.N = talker_numbers
        self.H = hidden_size
        self.use_residual = use_residual

        # Shared backbone for temporal modeling
        self.lstm = StackedCustomLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
        )
        self.pre_ln = nn.LayerNorm(hidden_size)

        # Per-branch subnetwork (independent LayerNorm stabilizes each head’s scale)
        def make_branch():
            return nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.ReLU(),
                nn.LayerNorm(self.H),
                nn.Dropout(dropout),
            )
        self.sep_branches = nn.ModuleList([make_branch() for _ in range(self.N)])

        # Optional residual connection scale (learnable): y_i = branch(h) + res_scale * h
        self.res_scale = nn.Parameter(torch.tensor(1.0)) if use_residual else None

        self._reset_parameters()

    def _reset_parameters(self):
        """Conservative init to reduce early gradient spikes."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, T, H)
        """
        Args:
            x: Input features (B, T, H)
        Returns:
            List[Tensor]: length N, each (B, T, H)
        """
        # Shared trunk
        h = self.lstm(x)   # (B, T, H)
        h = self.pre_ln(h)

        # Per-branch processing
        outs = []
        for branch in self.sep_branches:
            y = branch(h)  # (B, T, H)
            if self.use_residual:
                y = y + self.res_scale * h
            outs.append(y)

        return outs  # List[(B, T, H)]
