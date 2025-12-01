import torch
from typing import List, Tuple, Optional

@torch.no_grad()
def split_k_speakers_and_lengths(
    labels: torch.Tensor,               # (B, L)
    k_speakers: int,                    # expected number of speakers K
    sep_id: int,                        # special token that separates speakers
    pad_token_id: int,                  # padding token id for outputs
    ignore_id: Optional[int] = -100,    # tokens to drop inside segments (if present)
    end_token_id: Optional[int] = -100,
    allow_empty_segment: bool = True,   # False → any empty segment raises
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    device = labels.device
    B, L = labels.shape

    seg_lists: List[List[torch.Tensor]] = [[] for _ in range(k_speakers)]
    len_lists: List[List[int]]          = [[] for _ in range(k_speakers)]
    max_len_per_head = [0] * k_speakers

    for b in range(B):
        row = labels[b]

        # trim by end_token_id (often same as ignore_id=-100)
        if end_token_id is not None:
            pos = (row == end_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                first_end = pos[0].item()
                row = row[:first_end]

        seps = (row == sep_id).nonzero(as_tuple=True)[0].tolist()
        expected = k_speakers - 1
        found = len(seps)
        if found != expected:
            raise ValueError(
                f"[split_k_speakers_and_lengths_strict] Sample index {b}: "
                f"found {found} separators (token id={sep_id}) but expected {expected}. "
                f"labels[b].shape={row.shape}"
            )

        # IMPORTANT: use current row length after trimming
        L_row = row.numel()
        starts = [0] + [i + 1 for i in seps]
        ends   = seps + [L_row]

        for i, (s, e) in enumerate(zip(starts, ends)):
            seg = row[s:e]

            # drop ignore_id inside the segment
            if ignore_id is not None:
                seg = seg[seg != ignore_id]

            # ---- NEW: right-trim pad_token_id ----
            if pad_token_id is not None and seg.numel() > 0:
                keep = (seg != pad_token_id)
                if keep.any():
                    last = keep.nonzero(as_tuple=True)[0][-1].item()
                    seg = seg[: last + 1]
                else:
                    seg = seg[:0]
            # -------------------------------------

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
            seg_i = torch.stack(rows, dim=0)

        len_i = torch.tensor(len_lists[i], dtype=torch.long, device=device)
        labels_per_spk.append(seg_i)
        lengths_per_spk.append(len_i)

    return labels_per_spk, lengths_per_spk

