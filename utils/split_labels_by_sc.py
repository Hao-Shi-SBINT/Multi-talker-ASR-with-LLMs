import torch
from typing import List, Tuple, Optional

@torch.no_grad()
def split_k_speakers_and_lengths(
    labels: torch.Tensor,              # (B, L)
    k_speakers: int,                   # expected number of speakers K
    sep_id: int,                       # special token that separates speakers
    pad_token_id: int,                 # padding token id for outputs
    ignore_id: Optional[int] = -100,   # tokens to drop inside segments (if present)
    allow_empty_segment: bool = True,  # False → any empty segment raises
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split each sample into exactly K segments using `sep_id`.
    - If a sample has fewer or more than (K-1) separators, raise ValueError.
    - Keeps batch size (B) unchanged.
    - Returns:
        labels_per_spk:  list of length K; each tensor is (B, Lmax_i)
        lengths_per_spk: list of length K; each tensor is (B,)
    """
    device = labels.device
    B, L = labels.shape

    # Collect per-head segments and lengths first (on CPU for flexibility)
    seg_lists: List[List[torch.Tensor]] = [[] for _ in range(k_speakers)]
    len_lists: List[List[int]]          = [[] for _ in range(k_speakers)]
    max_len_per_head = [0] * k_speakers

    for b in range(B):
        row = labels[b]
        seps = (row == sep_id).nonzero(as_tuple=True)[0].tolist()
        expected = k_speakers - 1
        found = len(seps)
        if found != expected:
            raise ValueError(
                f"[split_k_speakers_and_lengths_strict] Sample index {b}: "
                f"found {found} separators (token id={sep_id}) but expected {expected}. "
                f"labels[b].shape={row.shape}"
            )

        # Build exactly K segments: [0, s0), (s0, s1), ..., (s_{K-2}, L)
        starts = [0] + [i + 1 for i in seps]
        ends   = seps + [L]

        for i, (s, e) in enumerate(zip(starts, ends)):
            seg = row[s:e]
            if ignore_id is not None:
                seg = seg[seg != ignore_id]
            seg = seg.long().cpu()

            if seg.numel() == 0 and not allow_empty_segment:
                raise ValueError(
                    f"[split_k_speakers_and_lengths_strict] Sample {b}, speaker-slot {i} "
                    f"resulted in an empty segment while allow_empty_segment=False."
                )

            seg_lists[i].append(seg)
            seg_len = seg.numel()
            len_lists[i].append(seg_len)
            if seg_len > max_len_per_head[i]:
                max_len_per_head[i] = seg_len

    # Pad per-head to max length and stack back to device
    labels_per_spk:  List[torch.Tensor] = []
    lengths_per_spk: List[torch.Tensor] = []

    for i in range(k_speakers):
        max_len = max_len_per_head[i]
        if max_len == 0:
            seg_i = torch.full((B, 0), pad_token_id, dtype=torch.long, device=device)
        else:
            rows = []
            for seg in seg_lists[i]:
                if seg.numel() < max_len:
                    pad = torch.full((max_len - seg.numel(),), pad_token_id, dtype=torch.long)
                    seg = torch.cat([seg, pad], dim=0)
                rows.append(seg.to(device, non_blocking=True))
            seg_i = torch.stack(rows, dim=0)  # (B, Lmax_i)

        len_i = torch.tensor(len_lists[i], dtype=torch.long, device=device)  # (B,)
        labels_per_spk.append(seg_i)
        lengths_per_spk.append(len_i)

    return labels_per_spk, lengths_per_spk

