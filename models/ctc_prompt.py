from typing import List, Tuple
import torch
import torch.nn as nn

def build_multi_ctc_prefix_from_heads(
    ctc_transcription_list: List[torch.Tensor],  # list[K] of [B, L_k] (已经去掉blank并pad)
    decoder: nn.Module,                          # LLaMA / Qwen decoder model
    pad_id: int,                                 # 你传进来的 pad_token_id
    max_prefix_len_per_head: int = 64,           # 每个head最多保留多少token（可调）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        ctc_transcription_list:
            长度为 K 的 list，每个元素是形状 [B, L_k] 的 long tensor。
            - 每一行是一条序列（已经过 CTC collapse，blank 已移除）
            - 右侧用 pad_id 进行 padding。
        decoder:
            下游 LLM（LLaMA / Qwen 等），用于拿 embedding。
        pad_id:
            用于 padding 的 token id（应与 decoder.config.pad_token_id 一致）。
        max_prefix_len_per_head:
            每个 head 最多保留的 token 数，用于限制前缀长度。

    Returns:
        prefix_embeds: [B, L_total, d_model]
        prefix_mask:   [B, L_total] bool，True 为有效位置
        prefix_ids:    [B, L_total] long，padding 位置为 pad_id
    """
    assert len(ctc_transcription_list) > 0, "ctc_transcription_list must not be empty."

    K = len(ctc_transcription_list)
    B = ctc_transcription_list[0].size(0)
    device = ctc_transcription_list[0].device

    # sanity check: batch size 一致
    for i, t in enumerate(ctc_transcription_list):
        assert t.size(0) == B, f"CTC head {i} has different batch size: {t.size(0)} vs {B}"

    # 从 decoder 拿 embedding 层
    if hasattr(decoder, "model") and hasattr(decoder.model, "embed_tokens"):
        embed = decoder.model.embed_tokens
    else:
        embed = decoder.get_input_embeddings()

    # 如果 decoder.config 里有 pad_token_id，最好检查一下是否一致
    dec_pad_id = getattr(getattr(decoder, "config", None), "pad_token_id", None)
    if dec_pad_id is not None and dec_pad_id != pad_id:
        print(f"[WARN] pad_id mismatch: decoder.config.pad_token_id={dec_pad_id}, "
              f"but function pad_id={pad_id}. Using pad_id={pad_id} for prefix_ids.")

    # -------- 核心：按 batch 聚合每个 head 的有效 token --------
    # all_ids_per_b[b] 是一个 list，里面按 head 顺序放多个 1D tensor
    all_ids_per_b: List[List[torch.Tensor]] = [[] for _ in range(B)]
    max_L_total = 0

    for head_idx, ctc_ids in enumerate(ctc_transcription_list):
        # ctc_ids: [B, L_k]
        for b in range(B):
            seq_b = ctc_ids[b]                      # [L_k]
            # 去掉 pad_id
            valid_ids = seq_b[seq_b != pad_id]      # [L_valid]
            if max_prefix_len_per_head is not None and \
               valid_ids.size(0) > max_prefix_len_per_head:
                valid_ids = valid_ids[:max_prefix_len_per_head]

            if valid_ids.numel() > 0:
                all_ids_per_b[b].append(valid_ids)

    # 计算每个 sample 合并后的总长度
    lengths: List[int] = []
    for b in range(B):
        if len(all_ids_per_b[b]) == 0:
            lengths.append(0)
        else:
            total_len_b = sum(seq.size(0) for seq in all_ids_per_b[b])
            lengths.append(total_len_b)
            if total_len_b > max_L_total:
                max_L_total = total_len_b

    # 确保至少有 1 个位置，避免 [B, 0, d_model]
    if max_L_total == 0:
        max_L_total = 1

    # 分配最终的 prefix_ids / mask
    prefix_ids = torch.full(
        (B, max_L_total),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    prefix_mask = torch.zeros(
        (B, max_L_total),
        dtype=torch.bool,
        device=device,
    )

    for b in range(B):
        # if len(all_ids_per_b[b]) == 0:
            # 这个 sample 所有 head 都给不出 token，保持全 pad / 全 False
        #     continue
        ids_b = torch.cat(all_ids_per_b[b], dim=0)   # [L_total_b]
        Lb = ids_b.size(0)
        prefix_ids[b, :Lb] = ids_b
        prefix_mask[b, :Lb] = True

    """
    # debug：检查是否越界
    vocab_size = embed.weight.size(0)
    if prefix_ids.max() >= vocab_size or prefix_ids.min() < 0:
        print("[WARN] CTC prefix ids out of range:",
              "min =", prefix_ids.min().item(),
              "max =", prefix_ids.max().item(),
              "lm_vocab =", vocab_size)
        prefix_ids = prefix_ids.clamp(min=0, max=vocab_size - 1)
    """

    # embedding
    prefix_embeds = embed(prefix_ids)  # [B, L_total, d_model]

    return prefix_embeds, prefix_mask, prefix_ids

