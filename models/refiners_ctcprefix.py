# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class CTCPerSpeakerExtractorConcatFrameGuided(nn.Module):
    """
    Frame-level extractor & repair guided by CTC frame info.

    Key ideas:
      - Build per-speaker frame weights A_k(t) = 1 - P(blank|t) from CTC logits (or use provided).
      - Resample A_k and per-speaker features to mixed timeline Tm.
      - Non-normalized gating (NNG): per speaker mask w_k(t) derived from A_k(t); overlap allowed.
      - Repair-by-fusion (frame-wise): fuse mixed X(t) with speaker k's resampled features Hk_r(t),
        then gate the fusion by w_k(t) to produce X_k(t). Finally concat X_k streams along time.

    Input:
      x_m:               [B, Tm, d_in_m]     # mixed features (after projector/encoder)
      blank_id:          int                 # CTC blank id
      sep_hidden_list:   List[K]*[B, Tk, d_in_s]  # per-speaker CTC-encoder hiddens (CTC head inputs)
      serialized_ctc:    optional nn.ModuleList length K (to compute logits internally)
      logits_list:       optional List[K]*[B, Tk, V] (if you already computed)
      A_list_opt:        optional List[K]*[B, Tk]    (if you already have 1-P(blank))

    Output:
      X_concat:          [B, K*Tm, d_model]  # time-concat per-speaker streams (frame-level)
      M_concat:          [B, K*Tm] (bool)    # mask (all True here)

    Notes:
      - Overlap is supported (non-normalized gating).
      - Lightweight & stable for 2–3 spk; avoids tokenization, keeps frame granularity.
      - You can switch gate mode / smoothing / entropy down-weight for 3spk robustness.
    """

    def __init__(
        self,
        d_in_m: int,            # dim of mixed features x_m
        d_in_s: int,            # dim of per-speaker sep features (CTC pre-softmax hidden)
        d_model: int,           # LLM hidden size
        K_spk: int,             # number of speakers
        *,
        use_weighted_A: bool = True,   # if logits provided, A_k = 1-P(blank); else A_k=None
        smooth_win: int = 3,           # mean smoothing on A_k; 0=off; 3~5 helps 3spk
        entropy_lambda: float = 0.0,    # >0 -> down-weight high-entropy frames (needs logits)
        prob_floor: float = 0.0,        # when building A from logits: treat (1-p_blank)<prob_floor as 0
        gate_mode: str = "pow",         # "pow" or "sigmoid"
        gate_gamma: float = 1.0,        # for "pow": w = clamp(A**gamma, 0, gate_cap)
        gate_cap: float = 1.25,         # cap for non-normalized gate
        fuse_hidden: int = 0,           # 0: Linear(2D->D); >0: MLP 2D->H->D
        depth_smooth: int = 0,          # temporal DWConv smoothing on fused stream; 0=off (e.g., 5 or 7)
        tag_scale: float = 1.2,         # per-speaker additive tag
        dropout: float = 0.0,           # dropout inside fusion
        use_ln: bool = True,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.K = K_spk
        self.use_weighted_A = use_weighted_A
        self.smooth_win = int(smooth_win)
        self.entropy_lambda = float(entropy_lambda)
        self.prob_floor = float(prob_floor)
        self.gate_mode = gate_mode
        self.gate_gamma = float(gate_gamma)
        self.gate_cap = float(gate_cap)
        self.tag_scale = float(tag_scale)
        self.depth_smooth = int(depth_smooth)

        # project inputs to model dim
        self.proj_m = nn.Linear(d_in_m, d_model)
        self.proj_s = nn.Linear(d_in_s, d_model)    # shared for all speakers
        Dm = self.proj_m.out_features

        # fusion: [X; Hk_r] -> Y_k (frame-wise)
        if fuse_hidden > 0:
            self.fuse = nn.Sequential(
                nn.Linear(2 * Dm, fuse_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fuse_hidden, Dm),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Linear(2 * Dm, Dm),
            )

        # optional temporal smoothing (depthwise conv)
        if self.depth_smooth and self.depth_smooth > 1:
            k = self.depth_smooth
            pad = (k - 1) // 2
            self.dwconv = nn.Conv1d(Dm, Dm, kernel_size=k, groups=Dm, padding=pad)
        else:
            self.dwconv = None

        # per-speaker tags (additive)
        self.spk_tags = nn.Parameter(torch.randn(K_spk, Dm) / (Dm ** 0.5))

        # polish
        self.use_ln = use_ln
        self.use_ffn = use_ffn
        if use_ln:
            self.ln = nn.LayerNorm(Dm)
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(Dm, 4 * Dm),
                nn.GELU(),
                nn.Linear(4 * Dm, Dm),
            )

        self._dim_checked = False

    # -------------------- public forward --------------------

    def forward(
        self,
        x_m: torch.Tensor,                                      # [B,Tm,d_in_m]
        blank_id: int,
        *,
        sep_hidden_list: List[torch.Tensor],                    # K*[B,Tk,d_in_s]
        serialized_ctc: Optional[nn.ModuleList] = None,
        logits_list: Optional[List[torch.Tensor]] = None,       # K*[B,Tk,V]
        A_list_opt: Optional[List[torch.Tensor]] = None,        # K*[B,Tk]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(sep_hidden_list) == self.K, f"expect K={self.K} sep streams"

        B, Tm, _ = x_m.shape
        X = self.proj_m(x_m)                                    # [B,Tm,D]
        Dm = X.size(-1)

        # one-time dim check
        if not self._dim_checked:
            if self.use_ln:
                assert self.ln.normalized_shape == (Dm,)
            if self.use_ffn:
                lin0: nn.Linear = self.ffn[0]
                lin2: nn.Linear = self.ffn[2]
                assert lin0.in_features == Dm and lin2.out_features == Dm
            assert self.spk_tags.size(1) == Dm
            self._dim_checked = True

        # ---- Build A_k (if needed) & resample speaker features to Tm ----
        if (A_list_opt is None) and self.use_weighted_A:
            if (logits_list is None) and (serialized_ctc is not None):
                logits_list = [ctc.logits(h) for ctc, h in zip(serialized_ctc, sep_hidden_list)]
            if logits_list is None:
                raise ValueError("Need logits_list or serialized_ctc to build A=1-P(blank).")
            A_list = [self._nonblank_from_logits(lg, blank_id) for lg in logits_list]  # K*[B,Tk]
            if self.entropy_lambda > 0.0:
                A_list = [self._entropy_down_weight(A, lg) for A, lg in zip(A_list, logits_list)]
            if self.prob_floor > 0.0:
                A_list = [A * (A >= self.prob_floor).float() for A in A_list]
            if self.smooth_win and self.smooth_win > 1:
                A_list = [self._smooth_mean_1d(A, self.smooth_win) for A in A_list]
        else:
            A_list = A_list_opt  # could be None

        streams = []
        for k in range(self.K):
            Hk = sep_hidden_list[k]                              # [B,Tk,d_in_s]
            Hk = self.proj_s(Hk)                                 # [B,Tk,D]
            Hk_r = self._resample_time(Hk, Tm)                   # [B,Tm,D]

            if A_list is not None:
                Ak = self._resample_time_1d(A_list[k], Tm)       # [B,Tm]
                wk = self._make_gate(Ak)                         # [B,Tm]
            else:
                wk = X.new_ones(B, Tm)                           # fallback: all-ones gate

            # Frame-wise fusion (repair): [X; Hk_r] -> Y_k
            Yk = self.fuse(torch.cat([X, Hk_r], dim=-1))         # [B,Tm,D]
            if self.dwconv is not None:
                # depthwise temporal smoothing
                Yk = self.dwconv(Yk.transpose(1, 2)).transpose(1, 2)

            # Apply gate (non-normalized, allow overlap). Two common choices:
            #  - Residual blend: Xk = (1 - s)*X + s*Yk, where s = sigmoid(beta*wk)
            #  - Masked add:    Xk = X + wk*Yk
            # Here we use residual blend (more stable):
            s = torch.sigmoid(wk).unsqueeze(-1)                  # [B,Tm,1]
            Xk = (1.0 - s) * X + s * Yk                          # [B,Tm,D]

            # Add speaker tag (additive)
            Xk = Xk + self.tag_scale * self.spk_tags[k].view(1,1,-1)

            # Optional polish
            if self.use_ln:
                Xk = self.ln(Xk)
            if self.use_ffn:
                Xk = Xk + self.ffn(Xk)

            streams.append(Xk)                                    # list of [B,Tm,D]

        # ---- concat per speaker along time ----
        X_concat = torch.cat(streams, dim=1)                      # [B,K*Tm,D]
        M_concat = torch.ones(B, self.K * Tm, dtype=torch.bool, device=X_concat.device)
        return X_concat, M_concat

    # -------------------- helpers --------------------

    @staticmethod
    def _nonblank_from_logits(lg: torch.Tensor, blank_id: int) -> torch.Tensor:
        # lg: [B,T,V] -> A = 1 - P(blank|t): [B,T]
        p = lg.log_softmax(dim=-1).exp()
        return 1.0 - p[..., blank_id]

    def _entropy_down_weight(self, A: torch.Tensor, lg: torch.Tensor) -> torch.Tensor:
        # A *= sigmoid(-λ * H(t)), H: frame entropy over vocab (stabilize 3spk)
        if self.entropy_lambda <= 0.0:
            return A
        p = lg.log_softmax(dim=-1).exp()
        H = -(p * (p.clamp_min(1e-12)).log()).sum(-1)  # [B,T]
        return A * torch.sigmoid(-self.entropy_lambda * H)

    @staticmethod
    def _smooth_mean_1d(A: torch.Tensor, win: int) -> torch.Tensor:
        if win <= 1:
            return A
        pad = (win - 1) // 2
        x = A.float().unsqueeze(1)  # [B,1,T]
        w = torch.ones(1, 1, win, device=A.device) / win
        x = F.pad(x, (pad, pad), mode="replicate")
        y = F.conv1d(x, w).squeeze(1)
        return y[:, :A.size(1)].to(A.dtype)

    @staticmethod
    def _resample_time(x_btD: torch.Tensor, T: int) -> torch.Tensor:
        # x: [B,T0,D] -> [B,T,D]
        return F.interpolate(x_btD.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)

    @staticmethod
    def _resample_time_1d(A_bt: torch.Tensor, T: int) -> torch.Tensor:
        # A: [B,T0] -> [B,T]
        return F.interpolate(A_bt.unsqueeze(1), size=T, mode="linear", align_corners=False).squeeze(1)

    def _make_gate(self, A_bt: torch.Tensor) -> torch.Tensor:
        """
        Build per-speaker non-normalized gate w from A.
          - 'pow':     w = clamp(A**gamma, 0, gate_cap)
          - 'sigmoid': w = cap * sigmoid(gamma * (A-0.5))
        """
        if self.gate_mode == "pow":
            w = torch.clamp(A_bt.clamp_min(0.0) ** self.gate_gamma, max=self.gate_cap)
        elif self.gate_mode == "sigmoid":
            w = self.gate_cap * torch.sigmoid(self.gate_gamma * (A_bt - 0.5))
        else:
            raise ValueError(f"Unknown gate_mode={self.gate_mode}")
        return w

