import torch
import torch.nn as nn
import torch.nn.functional as F

class SepMixGatedFusion(nn.Module):
    """
    对每个 frame，利用 mixed + sep 一起算一个 gate，做加权融合：
        fused = sep + g * (mix - sep)
              = (1-g)*sep + g*mix   （g in [0,1]）
    这样 sep 是 base，mixed 提供修正。
    """
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = 2 * d_model

        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),         # 输出 g(t) ∈ (0,1)，逐维 gate
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        sep_feat: torch.Tensor,   # [B, T, D]  per-speaker sep feature
        mix_feat: torch.Tensor,   # [B, T, D]  mixed feature (broadcast 给每个 speaker)
        mask: torch.Tensor = None # [B, T]，True=valid，False=pad（可选）
    ) -> torch.Tensor:
        assert sep_feat.shape == mix_feat.shape
        B, T, D = sep_feat.shape

        # 拼起来做门控
        x = torch.cat([sep_feat, mix_feat], dim=-1)  # [B,T,2D]
        g = self.gate_net(x)                         # [B,T,D]

        # residual 融合：以 sep 为 base，只加一个 mix - sep 的 delta
        fused = sep_feat + g * (mix_feat - sep_feat) # [B,T,D]
        fused = self.out_norm(fused)

        if mask is not None:
            fused = fused * mask.unsqueeze(-1)  # pad 位置归零
        return fused

