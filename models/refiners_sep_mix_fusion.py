import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FrameCrossAttnLayer(nn.Module):
    """
    一层 frame-level 的 Transformer block：
      - self-attn: 在 mixed 帧上做自注意力
      - cross-attn: Q = frame(mix)，K/V = token(CTC embedding)
      - FFN
    """
    def __init__(self, d_model: int, nhead: int, ff_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(ff_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(ff_ratio * d_model), d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_frame: torch.Tensor,          # [B, Tm, d_model]   mix 帧特征 (Q 源)
        tok_mem: torch.Tensor,          # [B, N,  d_model]   CTC token embedding (K/V 源)
        tok_mask: Optional[torch.Tensor] = None,  # [B, N], True=pad，一般可以 None
        frame_mask: Optional[torch.Tensor] = None # [B, Tm], True=pad（如果要 mask mix 帧）
    ) -> torch.Tensor:
        # 1) self-attn on frames
        h, _ = self.self_attn(
            x_frame, x_frame, x_frame,
            key_padding_mask=frame_mask,   # True=pad
            need_weights=False,
        )
        x = self.norm1(x_frame + self.dropout(h))

        # 2) cross-attn: Q = frame, K/V = token
        h, _ = self.cross_attn(
            query=x,            # [B, Tm, d_model]
            key=tok_mem,        # [B, N,  d_model]
            value=tok_mem,
            key_padding_mask=tok_mask,     # True=pad（一般没有就 None）
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(h))

        # 3) FFN
        h = self.ffn(x)
        x = self.norm3(x + self.dropout(h))
        return x  # [B, Tm, d_model]


class CrossAttnFrameExtractorNoVocab(nn.Module):
    """
    对单个 speaker：
      - 输入：
          y_emb:   [B, N,  d_tok]   该 speaker 的 CTC embedding（已用 LLaMA embed_tokens 得好）
          h_mix:   [B, Tm, d_enc]   mixed encoding
      - 输出：
          z_frame: [B, Tm, d_out]   该 speaker 的 frame-level encoding（时间维 = Tm）
    """
    def __init__(
        self,
        d_enc: int,                # mixed feature dim
        d_tok: int,                # CTC embedding dim（一般 = LLaMA hidden）
        d_model: int,              # 内部 transformer 的隐层维度
        nhead: int = 8,
        num_layers: int = 2,
        d_out: Optional[int] = None,  # 输出维度，默认 = d_enc，方便直接替 sep_hidden
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_out is None:
            d_out = d_enc

        self.d_enc = d_enc
        self.d_tok = d_tok
        self.d_model = d_model
        self.d_out = d_out

        # mixed → d_model
        self.proj_mix_in = nn.Linear(d_enc, d_model)

        # token embedding → d_model
        if d_tok != d_model:
            self.proj_tok = nn.Linear(d_tok, d_model)
        else:
            self.proj_tok = nn.Identity()

        # 多层 frame-level Transformer
        self.layers = nn.ModuleList([
            FrameCrossAttnLayer(d_model=d_model, nhead=nhead, ff_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 输出映射到 d_out（比如 d_out = d_enc，方便直接替换 sep_hidden）
        if d_model != d_out:
            self.proj_out = nn.Linear(d_model, d_out)
        else:
            self.proj_out = nn.Identity()

    def forward(
        self,
        y_emb: torch.Tensor,           # [B, N,  d_tok]
        h_mix: torch.Tensor,           # [B, Tm, d_enc]
        mix_mask: Optional[torch.Tensor] = None,  # [B, Tm], True=pad（你的 encoder_attention_mask_ctc 取反）
        tok_mask: Optional[torch.Tensor] = None,  # [B, N], True=pad（一般 None）
    ) -> torch.Tensor:
        B, N, _ = y_emb.shape
        B2, Tm, _ = h_mix.shape
        assert B == B2, "batch size mismatch between y_emb and h_mix"

        # 1) mixed → d_model
        x_frame = self.proj_mix_in(h_mix)     # [B, Tm, d_model]

        # 2) token emb → d_model
        tok_mem = self.proj_tok(y_emb)        # [B, N, d_model]

        # 3) 多层 frame cross-attn
        for layer in self.layers:
            x_frame = layer(
                x_frame=x_frame,
                tok_mem=tok_mem,
                tok_mask=tok_mask,
                frame_mask=mix_mask,
            )  # [B, Tm, d_model]

        # 4) 映射到输出维度
        z_frame = self.proj_out(x_frame)      # [B, Tm, d_out]
        return z_frame


def build_multispeaker_prefix_from_embeds(
    extractor: CrossAttnFrameExtractorNoVocab,
    h_mix: torch.Tensor,                        # [B, Tm, d_enc]
    ctc_emb_list: List[torch.Tensor],           # list[K] of [B, N_k, d_llm]
    ctc_ids_list: List[torch.Tensor],           # list[K] of [B, N_k] (用于 labels)
    speaker_order: List[int],                   # 长度 K 的 perm，比如 [0,1,2]
    llama_embed: nn.Embedding,                  # LLaMA 的 embed_tokens
    mix_mask: Optional[torch.Tensor] = None     # [B, Tm], True=pad
):
    """
    返回：
        z_list:     list[K] of [B, N_k, d_model]  每个 speaker 的 per-token 声学 encoding
        z_serial:   [B, N_total, d_model]         按 speaker_order 串起来的声学前缀
        y_sot_ids:  [B, N_total]                  对应的 SOT 文本 ids (直接拼接 ctc_ids_list)
        prefix_emb: [B, N_total, d_model]         = text_emb + z_serial
    """
    K = len(ctc_emb_list)
    assert len(ctc_ids_list) == K
    assert len(speaker_order) == K

    # 1) 对每个 speaker 抽取 z_k
    z_list: List[torch.Tensor] = []
    for k in range(K):
        y_emb_k = ctc_emb_list[k]                # [B, N_k, d_llm]
        z_k = extractor(y_emb_k, h_mix, mix_mask=mix_mask)  # [B, N_k, d_model]
        z_list.append(z_k)

    # 2) 按 speaker_order 排序并拼接
    z_ordered = [z_list[k] for k in speaker_order]         # list[K] of [B, N_k, d_model]
    ids_ordered = [ctc_ids_list[k] for k in speaker_order] # list[K] of [B, N_k]

    z_serial = torch.cat(z_ordered, dim=1)   # [B, N_total, d_model]
    y_sot_ids = torch.cat(ids_ordered, dim=1)  # [B, N_total]

    """
    # 3) LLaMA 文本 embedding + 声学 prefix 叠加
    text_emb = llama_embed(y_sot_ids)        # [B, N_total, d_model]
    prefix_emb = text_emb + z_serial         # [B, N_total, d_model]
    """
    prefix_emb = z_serial

    return z_list, z_serial, y_sot_ids, prefix_emb

