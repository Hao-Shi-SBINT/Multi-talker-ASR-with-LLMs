import torch
from typing import List, Tuple

@torch.no_grad()
def split_k_speakers_and_lengths(
    labels: torch.Tensor,         # (B, L)
    k_speakers: int,              # 说话人数 K
    sep_id: int = 128256,         # 分段用的 special token
    pad_token_id: int = 0,        # 输出中用于 padding 的 token
    ignore_id: int = -100,        # 原始标签里用于占位/忽略的值
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    将标签序列按 sep_id 切成 K 段（说话人1..K），去掉 ignore_id，分别按段位右侧 padding，
    并返回每段的 labels 以及每段的有效长度。
    语义与原始三说话人代码一致：若某行分隔符数量 < K-1，则跳过该样本，不计入输出。

    Returns:
        labels_per_spk:  长度 K 的列表；第 i 段形状为 (B_eff, max_len_i)，内部已将 ignore_id→pad_token_id
        lengths_per_spk: 长度 K 的列表；第 i 段形状为 (B_eff,) 的真实长度（不含 pad）
        其中 B_eff 是满足条件（拥有至少 K-1 个 sep_id）的样本数。
    """
    device = labels.device
    B, L = labels.shape

    # 收集每段的序列（CPU累积，再搬回device）和各段的 length
    segments_list: List[List[torch.Tensor]] = [[] for _ in range(k_speakers)]
    lengths_list:  List[List[int]]          = [[] for _ in range(k_speakers)]
    max_len_per_seg = [0] * k_speakers

    for b in range(B):
        row = labels[b]
        sep_idx = (row == sep_id).nonzero(as_tuple=True)[0]
        if sep_idx.nelement() < (k_speakers - 1):
            # 与你原代码一致：不足 K-1 个分隔符就跳过该样本
            continue

        # 取前 K-1 个分隔符，构造 K 段区间 [start, end)
        sep_idx = sep_idx[:k_speakers - 1].tolist()
        starts = [0] + [i + 1 for i in sep_idx]
        ends   = sep_idx + [L]

        for i, (s, e) in enumerate(zip(starts, ends)):
            seg = row[s:e]
            seg = seg[seg != ignore_id].long().cpu()     # 过滤 ignore_id
            segments_list[i].append(seg)
            lengths_list[i].append(seg.numel())
            if seg.numel() > max_len_per_seg[i]:
                max_len_per_seg[i] = seg.numel()

    # 若全部被跳过，返回空
    if len(segments_list[0]) == 0:
        empty_labels = [torch.empty((0, 0), dtype=torch.long, device=device) for _ in range(k_speakers)]
        empty_lens   = [torch.empty((0,),    dtype=torch.long, device=device) for _ in range(k_speakers)]
        return empty_labels, empty_lens

    # 逐段位做 padding（先用 ignore_id 占位，再整体替换为 pad_token_id）
    labels_per_spk:  List[torch.Tensor] = []
    lengths_per_spk: List[torch.Tensor] = []

    for i in range(k_speakers):
        max_len = max_len_per_seg[i]
        padded_rows = []
        # 把长度堆成 (B_eff,)
        len_tensor = torch.tensor(lengths_list[i], dtype=torch.long, device=device)

        for seg in segments_list[i]:
            if seg.numel() < max_len:
                pad = torch.full((max_len - seg.numel(),), ignore_id, dtype=torch.long)
                seg = torch.cat([seg, pad], dim=0)
            padded_rows.append(seg.to(device, non_blocking=True))

        seg_i = torch.stack(padded_rows, dim=0) if max_len > 0 else torch.empty((len(padded_rows), 0), dtype=torch.long, device=device)
        seg_i = seg_i.masked_fill(seg_i == ignore_id, pad_token_id)

        labels_per_spk.append(seg_i)   # (B_eff, max_len_i)
        lengths_per_spk.append(len_tensor)  # (B_eff,)

    return labels_per_spk, lengths_per_spk

