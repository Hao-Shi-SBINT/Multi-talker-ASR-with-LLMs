import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mt_ctctoken_builder import MultiSpkCTCTokenBuilder

class CTCAwareTinyCrossAttnAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mem_dim: int,
        attn_dim: int = 512,
        dropout: float = 0.0,
    ):
        """
        CTC-aware 版本的 Tiny cross-attention。

        hidden_size: LLaMA hidden size (e.g., 4096)
        mem_dim:     acoustic feature dim (e.g., encoder/sep dim)
        attn_dim:    small bottleneck dim for attention (<< hidden_size)
        """
        super().__init__()
        # 投到低维空间做 attention
        self.q_proj = nn.Linear(hidden_size, attn_dim)
        self.k_proj = nn.Linear(mem_dim, attn_dim)
        self.v_proj = nn.Linear(mem_dim, attn_dim)

        # 把 context 映射回 LLaMA hidden 维度
        self.out_proj = nn.Linear(attn_dim, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

        # 新增：控制 CTC 置信度在 logits 上影响强度的 scale
        self.conf_scale = nn.Parameter(torch.tensor(1.0))
        # 新增：整体声学 cross-attn 的 gate
        self.cross_gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden: Tensor,              # [B, L_q, H] 文本侧 hidden
        mem: Tensor,                 # [B, T_m, D] 声学 serialized / fused 特征 (帧级或 token 级)
        mem_sep: Tensor,
        mem_mask: Tensor = None,     # [B, T_m] bool, True=padding
        mem_conf: Tensor = None, 
        mem_ctc_mask: Tensor = None,
        ctc_modules: Tensor = None,     # [B, T_m] float in [0,1], CTC-based confidence
    ) -> Tensor:
        """
        hidden:   [B, L_q, H]  文本 hidden（训练 L_q=L，推理时 L_q=1 也 OK）
        mem:      [B, T_m, D]  声学特征
        mem_mask: [B, T_m] bool, True=padding，需要被 mask 掉
        mem_conf: [B, T_m] float, 每个声学 token 的置信度（比如 1 - p_blank）
        """

        if mem is None:
            return hidden

        B, L_q, H = hidden.size()
        _, T_m, D = mem.size()

        # 1) LN + 线性映射到低维
        h_norm = self.ln_in(hidden)        # [B, L_q, H]
        Q = self.q_proj(h_norm)            # [B, L_q, A]
        K = self.k_proj(mem)               # [B, T_m, A]
        V = self.v_proj(mem)               # [B, T_m, A]

        # 2) scaled dot-product attention
        #    QK^T: [B, L_q, T_m]
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))

        # 2.1 padding mask：True=pad → -inf
        if mem_mask is not None:
            attn_scores = attn_scores.masked_fill(
                mem_mask.unsqueeze(1),  # [B, 1, T_m]
                float("-inf"),
            )

        # 2.2 CTC 置信度 refine (logit 级别)
        if mem_conf is not None:
            # mem_conf: [B, T_m] ∈ [0,1]
            conf = mem_conf.clamp(min=1e-6)     # 避免 log(0)
            conf = conf.unsqueeze(1)            # [B, 1, T_m]
            # log(conf) 作为 bias，加到 attention logits 中
            attn_scores = attn_scores + self.conf_scale * conf.log()

        # 3) softmax 得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L_q, T_m]

        # 3.1 再乘一次 conf（value 级别 refine），再归一化
        if mem_conf is not None:
            conf = mem_conf.clamp(min=1e-6).unsqueeze(1)  # [B, 1, T_m]
            attn_weights = attn_weights * conf            # 可靠 token 权重大
            attn_weights = attn_weights / (
                attn_weights.sum(dim=-1, keepdim=True) + 1e-6
            )

        attn_weights = self.dropout(attn_weights)

        # 4) 上下文向量: [B, L_q, A]
        context = torch.matmul(attn_weights, V)  # [B, L_q, A]

        # 5) 映射回 hidden_size，并 residual + LN (+ gate)
        context_h = self.out_proj(context)       # [B, L_q, H]
        out = self.ln_out(hidden + self.cross_gate * context_h)
        return out

