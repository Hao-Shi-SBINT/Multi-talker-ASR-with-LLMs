from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Helpers
# ---------------------------
def resample_to_length_1d(x_bt: torch.Tensor, T_target: int, mode: str = "nearest") -> torch.Tensor:
    """Resample [B, T] -> [B, T_target] along time (safe 1D)."""
    assert x_bt.dim() == 2, f"expected [B,T], got {x_bt.shape}"
    if x_bt.size(1) == T_target:
        return x_bt
    x = x_bt.float().unsqueeze(1)  # [B,1,T]
    if mode == "nearest":
        y = F.interpolate(x, size=T_target, mode="nearest")
    else:
        y = F.interpolate(x, size=T_target, mode="linear", align_corners=False)
    return y.squeeze(1)


@torch.no_grad()
def compute_A_H_from_logits(
    logits: torch.Tensor, blank_id: int, input_is_log_probs: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [B, T, V]
    Returns:
      A: [B, T] = 1 - P(blank | t)   (non-blank posterior)
      H: [B, T] = token entropy at t
    """
    if input_is_log_probs:
        p = logits.exp()
    else:
        p = logits.softmax(dim=-1)
    logp = (p.clamp_min(1e-12)).log()
    A = 1.0 - p[..., blank_id]
    H = -(p * logp).sum(dim=-1)
    return A, H


@torch.no_grad()
def find_spikes_from_A(A_bt: torch.Tensor, thresh: float = 0.6, min_dist: int = 4) -> torch.Tensor:
    """
    Peak picking on A (non-blank posterior).
    Returns spikes indices [B, S] with -1 padding.
    """
    B, T = A_bt.shape
    spikes = []
    for b in range(B):
        a = A_bt[b]
        idx = []
        last = -min_dist - 1
        for t in range(T):
            if a[t] >= thresh and (t - last) >= min_dist:
                l, r = max(0, t - 1), min(T - 1, t + 1)
                if a[t] >= a[l] and a[t] >= a[r]:
                    idx.append(t); last = t
        spikes.append(torch.tensor(idx, device=A_bt.device, dtype=torch.long) if idx
                      else torch.full((0,), -1, device=A_bt.device, dtype=torch.long))
    S = max((x.numel() for x in spikes), default=0)
    if S == 0:
        return torch.full((B, 0), -1, device=A_bt.device, dtype=torch.long)
    outs = []
    for x in spikes:
        if x.numel() < S:
            x = torch.cat([x, torch.full((S - x.numel(),), -1, device=A_bt.device, dtype=torch.long)], dim=0)
        outs.append(x)
    return torch.stack(outs, dim=0)  # [B, S]


def prune_spikes_topk_by_local_A_simple(
    A: torch.Tensor, spikes: torch.Tensor, k: int, r: int = 8
) -> torch.Tensor:
    """
    Keep top-k spikes by local mean(A) within radius r, per batch.
    Returns [B, k] (padded with -1 if needed).
    """
    B, S = spikes.shape
    T = A.size(1)
    scores = spikes.new_full((B, S), -1e9, dtype=A.dtype)
    for b in range(B):
        for s in range(S):
            ti = int(spikes[b, s].item())
            if 0 <= ti < T:
                t0, t1 = max(0, ti - r), min(T, ti + r + 1)
                scores[b, s] = A[b, t0:t1].mean()
    k_eff = max(1, min(k, S))
    _, topk_idx = scores.topk(k=k_eff, dim=1)
    gathered = [spikes[b, topk_idx[b]] for b in range(B)]
    out = torch.stack(gathered, dim=0)  # [B, k_eff]
    if k_eff < k:
        pad = spikes.new_full((B, k - k_eff), -1)
        out = torch.cat([out, pad], dim=1)
    return out


# ---------------------------
# Pooling (batch-adaptive length)
# ---------------------------
def spike_pool_gaussian_flex(
    h_ctc: torch.Tensor,           # [B, T_hi, d_c]
    spikes: torch.Tensor,          # [B, S]  (with -1 padding)
    A: Optional[torch.Tensor] = None,  # [B, T_hi]
    r: int = 8,
    sigma: float = 4.0,
    per_spk_max: Optional[int] = None,  # optional per-sample cap before pooling
    top_m: Optional[int] = None,        # optional extra cap
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian pooling with batch-adaptive length:
      - For each sample, keep its own number of valid spikes (capped by per_spk_max/top_m)
      - Pad within the batch to S_max, return Z:[B,S_max,d_c] and valid_mask:[B,S_max]
    """
    B, T, D = h_ctc.shape
    S0 = spikes.size(1)
    eff_indices, eff_counts = [], []
    for b in range(B):
        idx = [int(spikes[b, s].item()) for s in range(S0) if 0 <= int(spikes[b, s].item()) < T]
        if per_spk_max is not None and len(idx) > per_spk_max:
            idx = idx[:per_spk_max]
        if top_m is not None and len(idx) > top_m:
            idx = idx[:top_m]
        eff_indices.append(idx)
        eff_counts.append(len(idx))

    S_max = max(eff_counts, default=0)
    if S_max == 0:
        return h_ctc.new_zeros(B, 0, D), h_ctc.new_zeros(B, 0)

    rows_b, masks_b = [], []
    for b in range(B):
        rows, mask = [], []
        for ti in eff_indices[b]:
            t0, t1 = max(0, ti - r), min(T, ti + r + 1)
            t = torch.arange(t0, t1, device=h_ctc.device)
            w = torch.exp(-0.5 * ((t - ti) / sigma) ** 2)
            if A is not None:
                w = w * A[b, t]
            z = (h_ctc[b, t] * (w[:, None] / (w.sum() + 1e-6))).sum(dim=0)  # [d_c]
            rows.append(z); mask.append(1)
        while len(rows) < S_max:
            rows.append(h_ctc.new_zeros(D)); mask.append(0)
        rows_b.append(torch.stack(rows, dim=0))
        masks_b.append(torch.tensor(mask, device=h_ctc.device, dtype=h_ctc.dtype))

    Z = torch.stack(rows_b, dim=0)   # [B, S_max, d_c]
    M = torch.stack(masks_b, dim=0)  # [B, S_max]
    return Z, M


# ---------------------------
# Slot-PE (optional)
# ---------------------------
class SlotPE(nn.Module):
    """Speaker Slot Positional Encoding: X + sum_k alpha_k * tag_k."""
    def __init__(self, d_model: int, K: int, scale: float = 1.0):
        super().__init__()
        self.tags = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.scale = scale

    def _normalize_A(self, A_res):
        # Expect [B,K,T]; support list[K] of [B,T] or [B,T,K]
        if isinstance(A_res, (list, tuple)):
            A = torch.stack(list(A_res), dim=1)  # [B,K,T] or [B,K,T,1]
            if A.dim() == 4 and A.size(-1) == 1:
                A = A.squeeze(-1)
        else:
            A = A_res
            if A.dim() == 4 and A.size(-1) == 1:
                A = A.squeeze(-1)
            if A.dim() != 3:
                raise ValueError(f"SlotPE expects [B,K,T], got {A.shape}")
            if A.size(1) != self.tags.size(0) and A.size(2) == self.tags.size(0):
                A = A.transpose(1, 2)  # [B,K,T]
        return A

    def forward(self, X_prefix: torch.Tensor, A_resampled):
        A = self._normalize_A(A_resampled)                   # [B,K,T]
        alpha = A / (A.sum(dim=1, keepdim=True) + 1e-6)      # [B,K,T]
        slot = torch.einsum('bkt,kd->btd', alpha, self.tags) # [B,T,d]
        return X_prefix + self.scale * slot


# ---------------------------
# Batch-first Multi-Head Attention wrapper
# ---------------------------
class BTMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, bias=bias, batch_first=True)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, need_weights: bool = False):
        out, w = self.mha(Q, K, V, need_weights=need_weights, average_attn_weights=False)
        return out, w  # out: [B,Tq,d], w: [B, nH, Tq, Tk]


# ---------------------------
# Projector-only CTC Bridge (with adaptive-length pooling)
# ---------------------------
class CTCBridgeSparseSlot(nn.Module):
    """
    Projector-only fusion:
      - Memory = projector features (mixed) projected to d_model
      - For each speaker: spikes -> pooled anchors (adaptive length) -> queries
      - Cross-attend over memory; confidence gate; concat tracks; optional Slot-PE
      - Output = fused acoustic prefix [B, S_tot, d_model]
    """
    def __init__(
        self,
        d_proj: int,            # projector feature dim (encoder_hidden_states.size(-1))
        d_c: int,               # CTC pre-softmax dim (sep_hidden_stages[k].size(-1))
        d_model: int,           # LLM hidden size
        K: int,                 # num speakers
        n_heads: int = 4,
        gate_r: int = 8,
        top_m: int = 64,
        slot_scale: float = 1.0,
        attn_dropout: float = 0.0,
        use_slot_pe: bool = True,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.gate_r = gate_r
        self.top_m = top_m

        # memory (projector-only)
        self.proj_mem = nn.Linear(d_proj, d_model)

        # per-speaker map d_c -> 2*d_model (use first half as query seed)
        self.kv_ctc = nn.ModuleList([nn.Linear(d_c, 2 * d_model) for _ in range(K)])
        self.q_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn = BTMHA(d_model, n_heads=n_heads, dropout=attn_dropout, bias=True)
        self.slot_pe = SlotPE(d_model, K, scale=slot_scale) if use_slot_pe else None

    def build_guided_fusion_prefix(
        self,
        proj_feats: torch.Tensor,                 # [B, T_p, d_proj]  (projector output; mixed)
        h_ctc_list: List[torch.Tensor],           # list[K] of [B, T_hi, d_c]  (CTC pre-softmax)
        A_list: List[torch.Tensor],               # list[K] of [B, T_hi]
        spikes_list: List[torch.Tensor],          # list[K] of [B, S]
        *,
        per_spk_max: int = 32,
        add_slot_pe: bool = True,
    ) -> torch.Tensor:
        B = proj_feats.size(0)

        # (1) memory from projector only
        M_mem = self.proj_mem(proj_feats)         # [B, T_p, d_model]
        K_mem, V_mem = M_mem, M_mem

        # (2) per-speaker anchors -> queries -> fused tracks (adaptive length)
        fused_tracks = []
        for k in range(self.K):
            A_k = A_list[k]                        # [B, T_hi]
            spikes_k_all = spikes_list[k]          # [B, S]

            # Cap by local A (top-k) to avoid huge S
            k_keep = min(per_spk_max, spikes_k_all.size(1), self.top_m if self.top_m is not None else 10_000)
            spikes_k = prune_spikes_topk_by_local_A_simple(A_k, spikes_k_all, k=k_keep, r=self.gate_r)  # [B,k_keep]

            # Adaptive-length pooling within the batch (pad to this speaker's S_max_k)
            Z_k, valid_mask = spike_pool_gaussian_flex(
                h_ctc=h_ctc_list[k],
                spikes=spikes_k,
                A=A_k,
                r=self.gate_r, sigma=4.0,
                per_spk_max=None,   # already capped above
                top_m=None
            )  # Z_k:[B,S_max_k,d_c], valid_mask:[B,S_max_k]

            if Z_k.size(1) == 0 or valid_mask.sum() == 0:
                continue

            # d_c -> 2*d_model, take K-part as query seed
            K_seed, _ = self.kv_ctc[k](Z_k).chunk(2, dim=-1)         # [B,S_max_k,d_model]
            Q_k = torch.tanh(self.q_proj(K_seed))                    # [B,S_max_k,d_model]

            # Cross-attend over memory
            fused_k, _ = self.attn(Q_k, K_mem, V_mem, need_weights=False)  # [B,S_max_k,d_model]
            fused_k = self.o_proj(fused_k)

            # Confidence gate by local mean(A); zero-out invalid positions
            with torch.no_grad():
                Thi = A_k.size(1)
                conf = fused_k.new_zeros(fused_k.size(0), fused_k.size(1))  # [B,S_max_k]
                # We need the original spike indices after pruning (align by order)
                # Use spikes_k which holds the pruned indices (with possible -1 pads)
                for b in range(fused_k.size(0)):
                    for s in range(fused_k.size(1)):
                        if valid_mask[b, s] <= 0:
                            continue
                        ti = int(spikes_k[b, s].item())
                        if 0 <= ti < Thi:
                            t0, t1 = max(0, ti - self.gate_r), min(Thi, ti + self.gate_r + 1)
                            conf[b, s] = A_k[b, t0:t1].mean()
                gk = torch.sigmoid(2.0 * conf) * valid_mask         # [B,S_max_k]
            fused_k = fused_k * gk.unsqueeze(-1)                     # [B,S_max_k,d_model]

            fused_tracks.append(fused_k)

        if len(fused_tracks) == 0:
            return K_mem.new_zeros(B, 0, self.d_model)

        # (3) concat tracks across speakers (each track can have its own S_max_k)
        X_fused = torch.cat(fused_tracks, dim=1)   # [B, S_tot, d_model] where S_tot = sum_k S_max_k

        # (4) optional Slot-PE
        if add_slot_pe and (self.slot_pe is not None):
            A_resampled = []
            for k, fk in enumerate(fused_tracks):
                Sk = fk.size(1)
                Ar = resample_to_length_1d(A_list[k], Sk, mode="nearest")  # [B,Sk]
                A_resampled.append(Ar)
            X_fused = self.slot_pe(X_fused, A_resampled)

        return X_fused

    def forward(
        self,
        proj_feats: torch.Tensor,                 # [B, T_p, d_proj]
        h_ctc_list: List[torch.Tensor],
        A_list: List[torch.Tensor],
        spikes_list: List[torch.Tensor],
        *,
        per_spk_max: int = 32,
    ):
        X = self.build_guided_fusion_prefix(
            proj_feats=proj_feats,
            h_ctc_list=h_ctc_list, A_list=A_list, spikes_list=spikes_list,
            per_spk_max=per_spk_max, add_slot_pe=True
        )
        aux = {"align_loss": X.new_tensor(0.0)}
        return X, aux

