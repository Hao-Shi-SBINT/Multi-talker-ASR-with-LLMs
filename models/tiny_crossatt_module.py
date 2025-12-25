import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCrossAttnAdapter(nn.Module):
    def __init__(self, hidden_size: int, mem_dim: int, attn_dim: int = 512, dropout: float = 0.0):
        """
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

    def forward(self, hidden, mem, mem_sep, mem_mask=None, mem_conf=None, mem_ctc_mask=None, ctc_modules=None):
        """
        hidden: [B, L_q, H]   文本侧 hidden（训练时 L_q=L，推理时 L_q=1 也可以）
        mem:    [B, T_m, D]   声学 serialized / fused 特征
        mem_mask: [B, T_m] bool, True=padding，需要在 attn 里屏蔽
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
        attn_scores = torch.matmul(
            Q, K.transpose(1, 2)
        ) / (Q.size(-1) ** 0.5)

        if mem_mask is not None:
            # mem_mask: True 表示 padding，要设成 -inf
            attn_scores = attn_scores.masked_fill(
                mem_mask.unsqueeze(1),  # [B, 1, T_m]
                float("-inf"),
            )

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L_q, T_m]
        attn_weights = self.dropout(attn_weights)

        # 3) 上下文向量: [B, L_q, A]
        context = torch.matmul(attn_weights, V)

        # 4) 映射回 hidden_size，并 residual + LN
        context_h = self.out_proj(context)         # [B, L_q, H]
        out = self.ln_out(hidden + context_h)
        return out

