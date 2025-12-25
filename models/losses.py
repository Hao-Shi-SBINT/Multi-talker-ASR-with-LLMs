# losses.py
import itertools
import torch
import torch.nn as nn
import os


def build_perm(N: int, mode: str | None, step: int, rotate_every: int):
    """
    Reuse your old perm policy when NOT using PIT.
    mode: None | 'swap01' | 'reverse' | 'rotate'
    """
    if mode is None:
        return list(range(N))
    if mode == "swap01":
        assert N >= 2
        p = list(range(N))
        p[0], p[1] = p[1], p[0]
        return p
    if mode == "reverse":
        return list(reversed(range(N)))
    if mode == "rotate":
        k = (step // max(1, rotate_every)) % N
        base = list(range(N))
        return base[k:] + base[:k]
    raise ValueError(f"Unknown perm_mode: {mode}")


def batch_swap_stats(best_perm: torch.Tensor):
    """
    best_perm: (B, N) long，每行是该样本选中的最佳置换。
    返回: swapped_cnt, total, swapped_ratio
    """
    B, N = best_perm.shape
    ident = torch.arange(N, device=best_perm.device).unsqueeze(0).expand(B, N)
    is_identity = (best_perm == ident).all(dim=1)   # True 表示没交换
    swapped = ~is_identity                          # True 表示发生了交换
    swapped_cnt = int(swapped.sum().item())
    total = int(B)
    swapped_ratio = swapped_cnt / max(1, total)
    return swapped_cnt, total, swapped_ratio


def pit_ctc_loss(
    talker_ctc,                 # List[nn.Module], 长度 N
    sep_hidden_states,          # List[Tensor], 每个 (B, T, H)
    hlens,                      # (B,)
    label_spks,                 # List[Tensor], 每个 (B, Lmax_j)
    label_spks_lengths,         # List[Tensor], 每个 (B,)
    reduce: str = "mean",       # "mean" | "sum" | "none"
    max_perms: int | None = None,
    return_details: bool = False,
):
    """
    返回：
      - loss_ctc: 标量（当 reduce in {"mean","sum"}），或 (B,N)（当 reduce="none"）
      - details(可选): dict，含
          * "assigned_losses": (B, N) 每个样本在最佳排列下，各 head 的逐样本损失
          * "per_head_mean":  (N,)  按 batch 平均后的每个 head 损失
          * "best_perm":      (B, N) 每个样本的最佳排列，给出 head i 对应的 target 索引
    约定：每个 CTC head 的 forward 必须在 reduce=False 时返回 (B,) 逐样本损失。
    """
    device = sep_hidden_states[0].device
    N = len(talker_ctc)
    assert len(sep_hidden_states) == len(label_spks) == len(label_spks_lengths) == N

    B = hlens.size(0)
    # 计算两两 (head i, target j) 的逐样本损失 => (N, N, B)
    pair_losses = []
    for i in range(N):
        row = []
        for j in range(N):
            # 要求 ctc_head 返回逐样本 (B,)；确保内部 CTCLoss(reduction="none") 且不再做时间步归一化
            li = talker_ctc[i](sep_hidden_states[i].float(), hlens, label_spks[j], label_spks_lengths[j])
            if li.dim() == 0:
                raise RuntimeError("CTC returned scalar; ensure CTCLoss(reduction='none') and no extra reduction.")
            if li.dim() > 1:
                li = li.reshape(-1)
            row.append(li)  # (B,)
        pair_losses.append(torch.stack(row, dim=0))  # (N, B)
    pair_losses = torch.stack(pair_losses, dim=0)     # (N, N, B)

    # 枚举/采样排列
    if max_perms is None or N <= 3:
        perms = list(itertools.permutations(range(N)))
    else:
        # 简单采样一些排列（可替换为更高级的近似）
        base = list(itertools.permutations(range(N)))
        perms = base[:max_perms]
    P = len(perms)
    perms_tensor = torch.tensor(perms, device=device, dtype=torch.long)  # (P, N)

    # 计算每个排列、每个样本的总损失：sum_i loss[i, perm[i], b]
    # pair_losses: (N,N,B) -> 方便按 batch 维度聚合
    PL = pair_losses  # alias
    perm_losses = []
    for p in range(P):
        idx = perms_tensor[p]  # (N,)
        # 选取 (i, idx[i], :) -> (N, B) 后再按 i 求和 -> (B,)
        gather_ij = torch.stack([PL[i, idx[i], :] for i in range(N)], dim=0)  # (N,B)
        perm_losses.append(gather_ij.sum(dim=0))  # (B,)
    perm_losses = torch.stack(perm_losses, dim=0)  # (P, B)

    # 逐样本选择最优排列
    best_idx = perm_losses.argmin(dim=0)           # (B,)
    best_perm = perms_tensor[best_idx]             # (B, N)

    # 取出最佳排列下，各样本各 head 的损失 -> (B, N)
    # 先把 pair_losses 变为 (B, N, N)，再用 gather
    PL_bnn = pair_losses.permute(2, 0, 1).contiguous()   # (B,N,N)
    assigned_losses = torch.gather(PL_bnn, dim=2, index=best_perm.unsqueeze(-1)).squeeze(-1)  # (B,N)

    # 汇总
    if reduce == "mean":
        loss_ctc = assigned_losses.mean()
    elif reduce == "sum":
        loss_ctc = assigned_losses.sum()
    elif reduce == "none":
        loss_ctc = assigned_losses  # (B,N)
    else:
        raise ValueError(f"Unknown reduce={reduce}")

    if not return_details:
        return loss_ctc

    per_head_mean = assigned_losses.mean(dim=0)  # (N,)
    details = {
        "assigned_losses": assigned_losses,  # (B,N)
        "per_head_mean": per_head_mean,      # (N,)
        "best_perm": best_perm,              # (B,N)
    }
    return loss_ctc, details


class HybridLoss(nn.Module):
    """
    A flexible loss module that can compute:
    - Attention loss only
    - Serialized CTC loss only
    - Hybrid loss (weighted combination of attention + CTC)

    Args kept from your version:
        alpha, mode, blank_id, enable_blank_check, log_every_steps, rotate_every
    Added (backward-compatible):
        use_pit (bool): enable PIT for CTC.
        pit_until (int): only use PIT for the first N steps; <=0 means never.
        pit_every (int): do PIT every k steps to save compute.
        pit_max_perms (int|None): limit permutations when N>3 (sampling).
    """
    def __init__(self, alpha: float = 0.7, mode: str = 'hybrid',
                 blank_id: int | None = None,
                 enable_blank_check: bool = False,
                 log_every_steps: int = 0,
                 rotate_every: int = 100,
                 use_pit: bool = False,
                 pit_until: int = 1_000,
                 pit_every: int = 1,
                 pit_max_perms: int | None = None):
        super().__init__()
        assert mode in ('attention', 'ctc', 'hybrid'), "mode must be 'attention', 'ctc', or 'hybrid'"
        self.alpha = alpha
        self.mode = mode
        self.ce_loss = nn.CrossEntropyLoss()

        # your old permutation knobs
        self.perm_mode = None        # None | "swap01" | "reverse" | "rotate"
        self.rotate_every = rotate_every

        # checks / debug
        self.blank_id = blank_id
        self.enable_blank_check = enable_blank_check
        self.log_every_steps = int(log_every_steps)
        self.log_dict = {}

        # PIT knobs
        self.use_pit = use_pit
        self.pit_until = pit_until
        self.pit_every = pit_every
        self.pit_max_perms = pit_max_perms

    def forward(
        self,
        decoder_outputs=None,
        labels=None,
        decoder_vocab_size=None,
        talker_ctc=None,
        sep_hidden_states=None,
        encoder_attention_mask_ctc=None,
        label_spks=None,
        label_spks_lengths=None,
        talker_numbers=1,
        shared_params=None,
        return_dict=True,
    ):
        loss_attn = 0.0
        loss_ctc = 0.0

        # -----------------------------
        # Attention loss (if needed)
        # -----------------------------
        if self.mode in ('attention', 'hybrid'):
            if decoder_outputs is None or labels is None or decoder_vocab_size is None:
                raise ValueError("decoder_outputs, labels, decoder_vocab_size must be provided for attention loss")
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_attn = self.ce_loss(logits.reshape(-1, decoder_vocab_size), labels.reshape(-1))

        # -----------------------------
        # CTC loss (if needed)
        # -----------------------------
        if self.mode in ('ctc', 'hybrid'):
            if (talker_ctc is None or sep_hidden_states is None or encoder_attention_mask_ctc is None
                or label_spks is None or label_spks_lengths is None):
                raise ValueError("CTC related inputs must be provided for CTC loss")

            N = int(talker_numbers)
            assert len(talker_ctc) == N, f"len(talker_ctc)={len(talker_ctc)} != talker_numbers={N}"
            assert len(sep_hidden_states) == len(label_spks) == len(label_spks_lengths) == N, \
                "Mismatch among heads/labels/lengths"

            hlens = encoder_attention_mask_ctc.sum(dim=1).long()  # (B,)
            B = hlens.size(0)
            for i in range(N):
                x, y, yl = sep_hidden_states[i], label_spks[i], label_spks_lengths[i]
                assert x.size(0) == y.size(0) == yl.size(0) == B, f"batch dim mismatch @head {i}"
                assert yl.dtype in (torch.int32, torch.int64), f"length dtype must be int @head {i}"

            # Optional blank-range check
            step = int(getattr(self, "global_step", 0))
            if self.enable_blank_check and (self.blank_id is not None) and (step % max(1, self.log_every_steps or 1000) == 0):
                with torch.no_grad():
                    for i in range(N):
                        if label_spks_lengths[i].sum().item() > 0:
                            max_id = int(label_spks[i].max().item())
                            assert max_id < self.blank_id, \
                                f"[CTC blank check] head {i}: target id {max_id} >= blank_id {self.blank_id}"

            do_pit = False

            # ===== losses.py: replace the tail of HybridLoss.forward with this =====
            if do_pit:
                loss_ctc, pit_info = pit_ctc_loss(
                    talker_ctc=talker_ctc,
                    sep_hidden_states=sep_hidden_states,
                    hlens=hlens,
                    label_spks=label_spks,
                    label_spks_lengths=label_spks_lengths,
                    reduce="mean",
                    max_perms=self.pit_max_perms if N > 3 else None,
                    return_details=True,
                )
                assigned = pit_info["assigned_losses"]          # (B, N)
                ctc_per_head = [assigned[:, i] for i in range(N)]  # list of (B,)
                best_perm = pit_info["best_perm"]

            else:
                perm = build_perm(N, self.perm_mode, step=step, rotate_every=self.rotate_every)
                sep_hidden_states = [sep_hidden_states[j] for j in perm]
                label_spks         = [label_spks[j]         for j in perm]
                label_spks_lengths = [label_spks_lengths[j] for j in perm]

                ctc_per_head = []
                with torch.cuda.amp.autocast(enabled=False):
                    for i, ctc_head in enumerate(talker_ctc):
                        li = ctc_head(
                            sep_hidden_states[i].float(),
                            hlens,
                            label_spks[i],
                            label_spks_lengths[i],
                        )
                        if li.dim() == 0:
                            # fallback: make it per-sample for consistency
                            li = li.unsqueeze(0).expand(B)
                        elif li.dim() > 1:
                            li = li.reshape(-1)
                        ctc_per_head.append(li)  # each (B,)
                loss_ctc = torch.stack([l.mean() for l in ctc_per_head]).mean()


                # ---- per-head loss logging (after ctc_per_head computed) ----
                if ctc_per_head is not None:
                     # ctc_per_head: list of (B,) or scalar
                     per_head_mean = []
                     for Li in ctc_per_head:
                        if Li.dim() > 0:
                            per_head_mean.append(Li.mean().detach())
                        else:
                            per_head_mean.append(Li.detach())

                     # print("[CTC per-head mean]:", [x.item() for x in per_head_mean])
                # -----------------------------------------------------------

            """
            # -------- debug: grad conflict on shared params --------
            if shared_params is not None and torch.is_grad_enabled():
                shared_params = [p for p in shared_params if p.requires_grad]
                grads = []
                for Li in ctc_per_head:
                    Li_use = Li.mean() if Li.dim() > 0 else Li
                    gi = torch.autograd.grad(
                        Li_use, shared_params,
                        retain_graph=True,
                        allow_unused=True
                    )
                    gi = [g if g is not None else torch.zeros_like(p)
                          for g, p in zip(gi, shared_params)]
                    grads.append(gi)
                # ...(your cosine debug stays same)
                # -------- debug: grad conflict on shared params --------
                def flat(glist):
                    return torch.cat([g.reshape(-1) for g in glist])

                gflat = [flat(g) for g in grads]

                cos_mat = [[0.0] * N for _ in range(N)]
                conflict_cnt, total_cnt = 0, 0
                min_cos, min_pair = 1.0, None

                for i in range(N):
                        for j in range(i + 1, N):
                            cos_ij = torch.nn.functional.cosine_similarity(
                                gflat[i], gflat[j], dim=0, eps=1e-12
                            ).item()
                            cos_mat[i][j] = cos_ij
                            total_cnt += 1
                            if cos_ij < 0:
                                conflict_cnt += 1
                            if cos_ij < min_cos:
                                min_cos = cos_ij
                                min_pair = (i, j)

                if int(os.environ.get("RANK", "0")) == 0:
                        print("perm_used (as best_perm):", perm)
                        print("grad_cosine_matrix(upper):", cos_mat)
                        print("conflict_rate:", conflict_cnt / max(1, total_cnt))
                        print("worst_conflict_pair:", min_pair, "cos=", min_cos)
                # --------------------------------------
            """

        # -----------------------------
        # Combine (return scalar!)
        # -----------------------------
        if self.mode == 'attention':
            total_loss = loss_attn
            self.last_ctc_per_head = None
        elif self.mode == 'ctc':
            total_loss = loss_ctc
            self.last_ctc_per_head = ctc_per_head
        elif self.mode == 'hybrid':
            total_loss = self.alpha * loss_attn + (1.0 - self.alpha) * loss_ctc
            self.last_ctc_per_head = ctc_per_head

        return total_loss

