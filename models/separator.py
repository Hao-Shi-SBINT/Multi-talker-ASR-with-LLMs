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
    def __init__(self, hidden_size, talker_numbers: int, dropout=0.2):
        super().__init__()
        self.talker_numbers = talker_numbers

        self.lstm = StackedCustomLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)

        def make_branch():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )

        # 动态创建 N 条支路
        self.sep_branches = nn.ModuleList([make_branch() for _ in range(talker_numbers)])

    def forward(self, x):
        """
        x: (B, T, H) 或 (B, H)（按你原来的 StackedCustomLSTM 接口）
        返回: List[Tensor]，长度 = talker_numbers，每个与 x 同形状的 hidden 大小
        """
        x = self.lstm(x)
        x = self.norm(x)

        outs = [branch(x) for branch in self.sep_branches]
        return outs  # e.g. [out1, out2, ..., outN]

