# refiners.py
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utilities for CTC signals
# =========================

def resample_to_len_1d(x_bt: torch.Tensor, T_target: int, mode: str = "nearest") -> torch.Tensor:
    """
    x_bt: [B, T] -> [B, T_target]
    Safe 1D resample for guidance signals (A/H). Use 'nearest' to avoid align_corners issues.
    """
    if x_bt.size(1) == T_target:
        return x_bt
    x = x_bt.unsqueeze(1)  # [B,1,T]
    y = F.interpolate(
        x,
        size=T_target,
        mode=("nearest" if mode == "nearest" else "linear"),
        align_corners=False if mode != "nearest" else None,
    )
    return y.squeeze(1)


def entropy_from_logits(logits_btv: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V] -> frame-wise entropy [B, T]
    """
    logp = F.log_softmax(logits_btv, dim=-1)     # [B,T,V]
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)                  # [B,T]
    return H


def nonblank_from_logits(logits_btv: torch.Tensor, blank_id: int) -> torch.Tensor:
    """
    1 - P(blank), from logits: [B,T,V] -> [B,T]
    """
    p = F.softmax(logits_btv, dim=-1)            # [B,T,V]
    return 1.0 - p[..., blank_id]                # [B,T]


@torch.no_grad()
def build_guidance_from_ctc_logits(
    logits_list: List[torch.Tensor],
    blank_id: int,
    T_target: int,
    resample_mode: str = "nearest",
    aggregate: str = "max",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    From multiple CTC logits (one per speaker), build A/H lists and global guidance aligned to T_target.

    Args:
        logits_list:  list[K] of [B, Ts_k, V]
        blank_id:     int, index of blank token in CTC vocabulary
        T_target:     int, target time length (usually len of x_m)
        resample_mode: "nearest" or "linear"
        aggregate:    "max" or "mean" for A_global

    Returns:
        A_list:   list[K] of [B, T_target]  (1 - P(blank))
        H_list:   list[K] of [B, T_target]  (frame entropy)
        A_global: [B, T_target]             (aggregated across speakers)
        H_global: [B, T_target]             (mean of entropies across speakers)
    """
    A_list_raw = [nonblank_from_logits(lg, blank_id) for lg in logits_list]  # list[K] of [B,Ts]
    H_list_raw = [entropy_from_logits(lg) for lg in logits_list]             # list[K] of [B,Ts]

    A_list = [resample_to_len_1d(A, T_target, mode=resample_mode) for A in A_list_raw]
    H_list = [resample_to_len_1d(H, T_target, mode=resample_mode) for H in H_list_raw]

    A_stack = torch.stack(A_list, dim=-1)  # [B,T,K]
    if aggregate == "max":
        A_global = A_stack.max(dim=-1).values
    elif aggregate == "mean":
        A_global = A_stack.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported aggregate: {aggregate}")
    A_global = A_global.clamp(0, 1)

    H_stack = torch.stack(H_list, dim=-1)  # [B,T,K]
    H_global = H_stack.mean(dim=-1)

    return A_list, H_list, A_global, H_global


# =========================
# Continuous refiner blocks
# =========================

class DynamicLPF(nn.Module):
    """CTC-guided dynamic low-pass filtering (per-frame smoothing without killing details)."""
    def __init__(self, d_in: int, k: int = 9):
        super().__init__()
        self.k = k
        self.gen = nn.Sequential(
            nn.Linear(2, d_in), nn.SiLU(), nn.Linear(d_in, k), nn.Softmax(dim=-1)
        )
        self.dw = nn.Conv1d(d_in, d_in, k, padding=k // 2, groups=d_in, bias=False)
        nn.init.dirac_(self.dw.weight)  # near-identity at init

    def forward(self, x: torch.Tensor, A: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # x:[B,T,C], A/H:[B,T]
        B, T, C = x.shape
        coeff = self.gen(torch.stack([A, H], dim=-1))  # [B,T,k]  (not used as true conv kernel, see below)
        x_ch = x.transpose(1, 2)                       # [B,C,T]
        y = self.dw(x_ch)                              # [B,C,T]
        center = coeff[..., self.k // 2]               # [B,T]
        y = (center.unsqueeze(1) * x_ch + (1 - center).unsqueeze(1) * y)
        return y.transpose(1, 2)


class LocalSelfAttn(nn.Module):
    """Local self-attention with additive band mask."""
    def __init__(self, d_model: int, n_heads: int = 8, band: int = 16, dropout: float = 0.0):
        super().__init__()
        self.band = band
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.ln(x)

        # 2D additive mask: [T, T]  (0=keep, -inf=mask)
        mask = x.new_full((T, T), float("-inf"))
        for t in range(T):
            L0, L1 = max(0, t - self.band), min(T, t + self.band + 1)
            mask[t, L0:L1] = 0.0

        y = self.attn(qkv, qkv, qkv, attn_mask=mask, need_weights=False)[0]
        return x + y


class CrossRepair(nn.Module):
    """Cross-attend to the original mixed features (as memory) to repair over-smoothed details."""
    def __init__(self, d_model: int, n_heads: int = 8, band: int = 24, dropout: float = 0.0):
        super().__init__()
        self.band = band
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x_refined: torch.Tensor, x_mem: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_refined.shape
        Q = self.ln_q(x_refined)
        KV = self.ln_kv(x_mem)
        mask = x_refined.new_full((B, T, T), float("-inf"))
        for t in range(T):
            L0, L1 = max(0, t - self.band), min(T, t + self.band + 1)
            mask[:, t, L0:L1] = 0.0
        y = self.attn(Q, KV, KV, attn_mask=mask, need_weights=False)[0]
        y = x_refined + y
        return y + self.ffn(y)


class SoftSpeakerRouter(nn.Module):
    """Optional soft routing using per-speaker A_k(t); keeps a single continuous stream."""
    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.mix = nn.Linear(K, 1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, A_list: Optional[List[torch.Tensor]]) -> torch.Tensor:
        if (A_list is None) or (len(A_list) == 0):
            return x
        A = torch.stack(A_list, dim=-1)       # [B,T,K]
        w = torch.softmax(A, dim=-1)          # [B,T,K]
        g = torch.sigmoid(self.mix(w)).squeeze(-1)  # [B,T]
        y = x * (0.5 + 0.5 * g.unsqueeze(-1))
        return self.proj(y)


class ContinuousRefiner(nn.Module):
    """
    x_m -> DynamicLPF -> LocalSelfAttn -> (SoftRouter) -> CrossRepair -> LN
    Length T is unchanged. Trainable with ONLY downstream CE loss.
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        band_local: int = 16,
        band_repair: int = 24,
        K_spk: int = 0,
    ):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lpf": DynamicLPF(d_model, k=9),
                        "lsa": LocalSelfAttn(d_model, n_heads=n_heads, band=band_local, dropout=0.0),
                        "router": SoftSpeakerRouter(d_model, K_spk) if K_spk > 0 else nn.Identity(),
                        "repair": CrossRepair(d_model, n_heads=n_heads, band=band_repair, dropout=0.0),
                    }
                )
            )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x_m: torch.Tensor,                           # [B,Tm,d_in]
        A_global: Optional[torch.Tensor] = None,     # [B,Tm]
        H_global: Optional[torch.Tensor] = None,     # [B,Tm]
        A_list: Optional[List[torch.Tensor]] = None  # list[K] of [B,Tm]
    ) -> torch.Tensor:
        if H_global is None:
            H_global = A_global if A_global is not None else x_m.new_zeros(x_m.size(0), x_m.size(1))
        y = self.inp(x_m)  # [B,T,d_model]
        for blk in self.blocks:
            y = y + blk["lpf"](y, A_global if A_global is not None else 0 * y[..., 0],
                                  H_global if H_global is not None else 0 * y[..., 0])
            y = blk["lsa"](y)
            y = blk["router"](y, A_list) if not isinstance(blk["router"], nn.Identity) else y
            y = blk["repair"](y, self.inp(x_m))
        return self.out_ln(y)


# ==========================================
# High-level wrapper: CTCGuidedRefiner
# ==========================================

class CTCGuidedRefiner(nn.Module):
    """
    One-stop module:
      - Accepts either CTC logits directly, or (serialized_ctc, sep_hidden_list) to compute logits.
      - Builds A_list / H_list / A_global / H_global aligned to x_m length.
      - Runs ContinuousRefiner to get refined features (same length as x_m).
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        K_spk: int,
        n_layers: int = 2,
        n_heads: int = 8,
        band_local: int = 16,
        band_repair: int = 24,
        aggregate: str = "max",
        resample_mode: str = "nearest",
    ):
        super().__init__()
        self.aggregate = aggregate
        self.resample_mode = resample_mode
        self.refiner = ContinuousRefiner(
            d_in=d_in,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            band_local=band_local,
            band_repair=band_repair,
            K_spk=K_spk,
        )

    def _get_logits_list(
        self,
        logits_list: Optional[List[torch.Tensor]],
        serialized_ctc: Optional[nn.ModuleList],
        sep_hidden_list: Optional[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if logits_list is not None:
            return logits_list
        if (serialized_ctc is None) or (sep_hidden_list is None):
            raise ValueError("Provide either logits_list or (serialized_ctc, sep_hidden_list).")
        with torch.no_grad():  # guidance only; remove no_grad if you want gradients through CTC
            return [ctc.logits(h) for ctc, h in zip(serialized_ctc, sep_hidden_list)]

    def forward(
        self,
        x_m: torch.Tensor,                           # [B,Tm,d_in]
        blank_id: int,
        logits_list: Optional[List[torch.Tensor]] = None,              # list[K] of [B,Ts,V]
        serialized_ctc: Optional[nn.ModuleList] = None,
        sep_hidden_list: Optional[List[torch.Tensor]] = None,          # list[K] of [B,Ts,d_in]
    ) -> torch.Tensor:
        B, Tm, _ = x_m.shape
        # 1) get CTC logits
        lg_list = self._get_logits_list(logits_list, serialized_ctc, sep_hidden_list)

        # 2) build guidance aligned to Tm
        A_list, H_list, A_global, H_global = build_guidance_from_ctc_logits(
            lg_list,
            blank_id=blank_id,
            T_target=Tm,
            resample_mode=self.resample_mode,
            aggregate=self.aggregate,
        )

        # 3) refine with ContinuousRefiner
        X_ref = self.refiner(
            x_m=x_m,
            A_global=A_global,
            H_global=H_global,
            A_list=A_list,
        )
        return X_ref  # [B,Tm,d_model]


# ==========================================
# Per-speaker extraction + concat
# ==========================================

class LocalCrossRepair(nn.Module):
    """Optional: local cross-attn repair using original mixed memory."""
    def __init__(self, d_model: int, n_heads: int = 8, band: int = 24, dropout: float = 0.0):
        super().__init__()
        self.band = band
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x_refined: torch.Tensor, x_mem: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_refined.shape
        Q  = self.ln_q(x_refined)
        KV = self.ln_kv(x_mem)

        # 2D additive mask: [T, T]
        mask = x_refined.new_full((T, T), float("-inf"))
        for t in range(T):
            L0, L1 = max(0, t - self.band), min(T, t + self.band + 1)
            mask[t, L0:L1] = 0.0

        y = self.attn(Q, KV, KV, attn_mask=mask, need_weights=False)[0]
        y = x_refined + y
        return y + self.ffn(y)



def _resample_to_len_private(x_bt: torch.Tensor, T_target: int, mode: str = "nearest") -> torch.Tensor:
    if x_bt.size(1) == T_target:
        return x_bt
    x = x_bt.unsqueeze(1)
    y = F.interpolate(
        x,
        size=T_target,
        mode=("nearest" if mode == "nearest" else "linear"),
        align_corners=False if mode != "nearest" else None,
    )
    return y.squeeze(1)


def _nonblank_private(logits: torch.Tensor, blank_id: int) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return 1.0 - p[..., blank_id]


class CTCPerSpeakerExtractorConcatSoftmax(nn.Module):
    """
    Use CTC posteriors to extract per-speaker streams from mixed features, then concatenate.
    - No hard cutting. Soft mask per speaker: m_k(t) = softmax_k A_k(t).
    - Optional local cross-attn repair per speaker.
    - Adds a learnable speaker tag to each stream.
    Output: [B, K*T, d_model]
    """
    def __init__(
        self,
        d_in: int,                    # dim of mixed features (before LLM)
        d_model: int,                 # LLM hidden size
        K_spk: int,                   # number of speakers
        use_repair: bool = True,      # enable local cross-attn repair
        n_heads: int = 8,
        band_repair: int = 24,
        resample_mode: str = "nearest"
    ):
        super().__init__()
        self.K = K_spk
        self.resample_mode = resample_mode

        self.proj_in = nn.Linear(d_in, d_model)   # project mixed to model dim
        self.spk_tags = nn.Parameter(torch.randn(K_spk, d_model) * (1.0 / (d_model ** 0.5)))

        self.repair = LocalCrossRepair(d_model, n_heads=n_heads, band=band_repair, dropout=0.0) \
                        if use_repair else None

    def _build_A_list(
        self,
        Tm: int,
        blank_id: int,
        logits_list: Optional[List[torch.Tensor]] = None,
        serialized_ctc: Optional[nn.ModuleList] = None,
        sep_hidden_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Return list[K] of [B,Tm] non-blank posteriors aligned to mixed length.
        """
        if logits_list is None:
            if (serialized_ctc is None) or (sep_hidden_list is None):
                raise ValueError("Provide logits_list or (serialized_ctc, sep_hidden_list).")
            with torch.no_grad():
                logits_list = [ctc.logits(h) for ctc, h in zip(serialized_ctc, sep_hidden_list)]
        A_raw = [_nonblank_private(lg, blank_id) for lg in logits_list]  # list[K] of [B,Ts]
        A = [_resample_to_len_private(a, Tm, mode=self.resample_mode) for a in A_raw]
        return A  # list[K] of [B,Tm]

    def forward(
        self,
        x_m: torch.Tensor,   # [B,Tm,d_in]
        blank_id: int,
        logits_list: Optional[List[torch.Tensor]] = None,     # list[K] of [B,Ts,V]
        serialized_ctc: Optional[nn.ModuleList] = None,
        sep_hidden_list: Optional[List[torch.Tensor]] = None, # list[K] of [B,Ts,d_in]
    ) -> torch.Tensor:
        B, Tm, _ = x_m.shape
        X = self.proj_in(x_m)                    # [B,Tm,d_model]
        A_list = self._build_A_list(Tm, blank_id, logits_list, serialized_ctc, sep_hidden_list)

        # soft routing weights across speakers per frame
        A_stack = torch.stack(A_list, dim=-1)    # [B,Tm,K]
        W = torch.softmax(A_stack, dim=-1)       # [B,Tm,K]

        # per-speaker extraction + tag
        streams = []
        for k in range(self.K):
            w = W[..., k:k+1]                    # [B,Tm,1]
            Xk = X * w                           # [B,Tm,d_model]
            # add speaker tag
            Xk = Xk + self.spk_tags[k].view(1, 1, -1)

            # optional local repair using original mixed (projected)
            if self.repair is not None:
                Xk = self.repair(Xk, X)

            streams.append(Xk)                   # list of [B,Tm,d_model]

        # concat along time: [B, K*Tm, d_model]
        X_concat = torch.cat(streams, dim=1)
        return X_concat


class CTCPerSpeakerExtractorConcatNNG(nn.Module):
    """
    Use CTC posteriors to extract per-speaker streams from mixed features, then concatenate.
    This version uses non-normalized gating (sigmoid), which supports overlap:
        W_hat_k(t) = sigmoid(alpha * (A_k(t) - beta))
    Priority of guidance inputs:
        1) A_list (List[K] of [B,Tm])   <-- 推荐：外部先处理好再传入
        2) logits_list + blank_id
        3) serialized_ctc + sep_hidden_list + blank_id

    Output: [B, K*Tm, d_model]
    """
    def __init__(
        self,
        d_in: int,                    # dim of mixed features (before LLM)
        d_model: int,                 # LLM hidden size
        K_spk: int,                   # number of speakers
        use_repair: bool = True,      # enable local cross-attn repair
        n_heads: int = 8,
        band_repair: int = 24,
        resample_mode: str = "nearest",
        # 非归一化门控参数（固定值更稳；需要也可改为可学习）
        route_alpha: float = 6.0,
        route_beta:  float = 0.50,
        # 流内归一：避免门控后幅度过小
        use_stream_ln: bool = True,
        # 说话人标签
        use_tags: bool = True,
    ):
        super().__init__()
        self.K = K_spk
        self.resample_mode = resample_mode

        self.proj_in = nn.Linear(d_in, d_model)   # project mixed to model dim
        self.use_stream_ln = use_stream_ln
        if use_stream_ln:
            self.ln_stream = nn.LayerNorm(d_model)

        self.use_tags = use_tags
        if use_tags:
            self.spk_tags = nn.Parameter(torch.randn(K_spk, d_model) * (1.0 / (d_model ** 0.5)))

        self.repair = LocalCrossRepair(d_model, n_heads=n_heads, band=band_repair, dropout=0.0) \
                        if use_repair else None

        # 非归一化门控超参（默认固定）
        self.route_alpha = nn.Parameter(torch.tensor(route_alpha), requires_grad=False)
        self.route_beta  = nn.Parameter(torch.tensor(route_beta),  requires_grad=False)

    # ---- A_list 的统一构建入口：若显式提供 A_list 就直接用，否则再算 ----
    def _build_A_list(
        self,
        Tm: int,
        blank_id: Optional[int] = None,
        *,
        A_list: Optional[List[torch.Tensor]] = None,            # 优先
        logits_list: Optional[List[torch.Tensor]] = None,       # list[K] of [B,Ts,V]
        serialized_ctc: Optional[nn.ModuleList] = None,
        sep_hidden_list: Optional[List[torch.Tensor]] = None,   # list[K] of [B,Ts,d_in]
    ) -> List[torch.Tensor]:
        """
        Return list[K] of [B,Tm] non-blank posteriors aligned to mixed length.
        Priority: A_list -> logits_list -> (serialized_ctc+sep_hidden_list)
        """
        if A_list is not None:
            # 直接使用外部提供的 A_list（需已重采样到 Tm）
            assert len(A_list) == self.K, f"Expect {self.K} speakers, got {len(A_list)}"
            for a in A_list:
                assert a.dim() == 2 and a.size(1) == Tm, "Each A_k must be [B,Tm]"
            return A_list

        # 若无 A_list，则需要 blank_id
        if blank_id is None:
            raise ValueError("blank_id is required when A_list is not provided.")

        # 若给了 logits_list，则用它来计算
        if logits_list is not None:
            assert len(logits_list) == self.K, f"Expect {self.K} logits, got {len(logits_list)}"
            A_raw = [_nonblank_private(lg, blank_id) for lg in logits_list]   # list[K] of [B,Ts]
            A = [_resample_to_len_private(a, Tm, mode=self.resample_mode) for a in A_raw]
            return A

        # 否则使用 serialized_ctc + sep_hidden_list 计算 logits 再转 A
        if (serialized_ctc is None) or (sep_hidden_list is None):
            raise ValueError("Provide A_list, or logits_list+blank_id, or (serialized_ctc+sep_hidden_list+blank_id).")

        with torch.no_grad():
            logits_list = [ctc.logits(h) for ctc, h in zip(serialized_ctc, sep_hidden_list)]
        A_raw = [_nonblank_private(lg, blank_id) for lg in logits_list]
        A = [_resample_to_len_private(a, Tm, mode=self.resample_mode) for a in A_raw]
        return A

    def forward(
        self,
        x_m: torch.Tensor,   # [B,Tm,d_in]
        *,
        # 三种输入方式，三选一（优先 A_list）
        A_list: Optional[List[torch.Tensor]] = None,            # List[K] of [B,Tm]
        blank_id: Optional[int] = None,
        logits_list: Optional[List[torch.Tensor]] = None,       # list[K] of [B,Ts,V]
        serialized_ctc: Optional[nn.ModuleList] = None,
        sep_hidden_list: Optional[List[torch.Tensor]] = None,   # list[K] of [B,Ts,d_in]
    ) -> torch.Tensor:
        B, Tm, _ = x_m.shape
        X = self.proj_in(x_m)                                   # [B,Tm,d_model]

        # 1) 得到对齐到 Tm 的 A_list（优先使用外部 A_list）
        A_list = self._build_A_list(
            Tm, blank_id,
            A_list=A_list,
            logits_list=logits_list,
            serialized_ctc=serialized_ctc,
            sep_hidden_list=sep_hidden_list,
        )  # list[K] of [B,Tm]

        # 2) 非归一化门控（支持 overlap）
        A_stack = torch.stack(A_list, dim=-1)                   # [B,Tm,K]
        W_hat   = torch.sigmoid(self.route_alpha * (A_stack - self.route_beta))  # [B,Tm,K], not normalized

        # 3) 逐说话人抽取 + tag + 可选修复 + 流内 LN
        streams = []
        for k in range(self.K):
            w = W_hat[..., k:k+1]                               # [B,Tm,1]
            Xk = X * w                                          # [B,Tm,d_model]
            if self.repair is not None:
                Xk = self.repair(Xk, X)
            if self.use_tags:
                Xk = Xk + self.spk_tags[k].view(1, 1, -1)
            if self.use_stream_ln:
                Xk = self.ln_stream(Xk)
            streams.append(Xk)

        # 4) 时间拼接： [B, K*Tm, d_model]
        X_concat = torch.cat(streams, dim=1)
        return X_concat

