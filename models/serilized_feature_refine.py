import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


class CTCAwareFrameRefiner(nn.Module):
    """
    不用 token-level，完全在帧级上做 CTC-aware 特征 refine。

    输入:
      sep_hidden_list: K * [B, T, D]  # 每个 speaker 的 serialized 特征
      mixed_hidden:    [B, T, D]      # mixed encoder 特征 (假设已经对齐同样 T)
      enc_mask:        [B, T]         # True=valid, False=pad
      ctc_modules:     K * CTC 模块   # 和 sep 对应的 CTC head

    输出:
      refined_sep_list: K * [B, T, D] # 每个 speaker 的 refined 特征
    """

    def __init__(self, d_model: int, hidden_factor: int = 2):
        super().__init__()
        h = d_model * hidden_factor

        # 一个小 MLP，用来算 gate 和 delta
        # 输入: [sep, mixed, p_nonblank] -> 输出两个向量:
        #   gate:  [B, T, 1]   决定 mixed 参与多少
        #   delta: [B, T, D]   对 sep 做一个残差修正
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, h),
            nn.ReLU(),
            nn.Linear(h, d_model + 1),   # 前 D 维是 delta，最后 1 维是 gate_logit
        )

        self.ln_sep   = nn.LayerNorm(d_model)
        self.ln_mixed = nn.LayerNorm(d_model)

    @torch.no_grad()
    def _ctc_p_nonblank(
        self,
        ctc_module: nn.Module,
        sep_hidden: Tensor,     # [B, T, D]
    ) -> Tensor:
        """
        返回 p_nonblank = 1 - p_blank, 形状 [B, T]
        使用你的 CTC.log_softmax 接口。
        """
        log_probs = ctc_module.log_softmax(sep_hidden)   # [B, T, V]

        # 取 blank id
        if hasattr(ctc_module, "blank_id"):
            blank_id = int(ctc_module.blank_id)
        elif hasattr(ctc_module, "ctc_loss") and hasattr(ctc_module.ctc_loss, "blank"):
            blank_id = int(ctc_module.ctc_loss.blank)
        else:
            blank_id = log_probs.size(-1) - 1

        p_blank = log_probs[..., blank_id].exp()         # [B, T]
        p_nonblank = 1.0 - p_blank
        return p_nonblank

    def forward(
        self,
        sep_hidden_list: List[Tensor],      # K * [B, T, D]
        mixed_hidden: Tensor,               # [B, T, D]
        enc_mask: Tensor,                   # [B, T] True=valid
        ctc_modules: List[nn.Module],       # K * CTC modules
    ) -> List[Tensor]:
        B, T, D = mixed_hidden.shape
        device = mixed_hidden.device

        mixed_norm = self.ln_mixed(mixed_hidden)

        refined_list: List[Tensor] = []

        for k, sep_hidden in enumerate(sep_hidden_list):
            ctc_k = ctc_modules[k]

            sep_norm = self.ln_sep(sep_hidden)    # [B, T, D]

            # p_nonblank: [B, T]
            p_nonblank = self._ctc_p_nonblank(ctc_k, sep_hidden)  # no grad

            # 拼接特征: [sep, mixed, p_nonblank]
            p_nonblank_feat = p_nonblank.unsqueeze(-1)            # [B, T, 1]
            feat = torch.cat(
                [sep_norm, mixed_norm, p_nonblank_feat], dim=-1
            )                                                     # [B, T, 2D+1]

            out = self.mlp(feat)                                  # [B, T, D+1]
            delta, gate_logit = out[..., :D], out[..., D:]        # [B,T,D], [B,T,1]

            gate = torch.sigmoid(gate_logit)                      # [B,T,1]
            gate = gate * p_nonblank_feat + 0.1 * (1.0 - p_nonblank_feat)
            # 高 p_nonblank → gate 接近 sigmoid; 低 p_nonblank → gate 更小一点

            # 用 gate 融合 sep & mixed，再加 delta 做细调
            fused = sep_hidden + gate * (mixed_hidden - sep_hidden) + delta  # [B,T,D]

            # mask padding 帧，保持为 0（或者保持原 sep）
            if enc_mask is not None:
                mask = enc_mask.unsqueeze(-1)       # [B,T,1], True=valid
                fused = fused * mask + sep_hidden * (~mask)

            refined_list.append(fused)

        return refined_list

