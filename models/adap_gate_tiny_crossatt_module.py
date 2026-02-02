import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapGatedTinyCrossAttnAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mem_dim: int,
        attn_dim: int = 512,
        dropout: float = 0.0,
        # adaptive-rank LoRA settings
        r_max: int = 64,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        init_rank_logit: float = 2.0,   # sigmoid(2)≈0.88: 先开着，后期靠预算/稀疏压下去
        freeze_base: bool = True,
        apply_lora_to: str = "qkv_out",   # "q_out" or "qkv_out"
    ):
        super().__init__()

        # ======= Base cross-attn projections (KEEP NAMES for ckpt loading) =======
        self.q_proj = nn.Linear(hidden_size, attn_dim)
        self.k_proj = nn.Linear(mem_dim, attn_dim)
        self.v_proj = nn.Linear(mem_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, hidden_size)

        if freeze_base:
            for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

        self.apply_lora_to = apply_lora_to
        self.r_max = int(r_max)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = self.lora_alpha / max(1, self.r_max)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # ======= LoRA params (NEW keys; base keys unchanged) =======
        # Helper to init A/B
        def _init_lora(in_dim, out_dim):
            A = nn.Parameter(torch.empty(self.r_max, in_dim))        # [r, in]
            B = nn.Parameter(torch.empty(out_dim, self.r_max))       # [out, r]
            nn.init.kaiming_uniform_(A, a=5**0.5)
            nn.init.zeros_(B)
            rank_logits = nn.Parameter(torch.full((self.r_max,), float(init_rank_logit)))
            return A, B, rank_logits

        # Q LoRA
        self.q_lora_A, self.q_lora_B, self.q_rank_logits = _init_lora(
            in_dim=hidden_size, out_dim=attn_dim
        )
        # Out LoRA
        self.out_lora_A, self.out_lora_B, self.out_rank_logits = _init_lora(
            in_dim=attn_dim, out_dim=hidden_size
        )

        # Optional: K/V LoRA
        if apply_lora_to == "qkv_out":
            self.k_lora_A, self.k_lora_B, self.k_rank_logits = _init_lora(
                in_dim=mem_dim, out_dim=attn_dim
            )
            self.v_lora_A, self.v_lora_B, self.v_rank_logits = _init_lora(
                in_dim=mem_dim, out_dim=attn_dim
            )
        else:
            self.k_lora_A = self.k_lora_B = self.k_rank_logits = None
            self.v_lora_A = self.v_lora_B = self.v_rank_logits = None

        # ======= rest unchanged =======
        self.dropout = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(hidden_size)
        self.ln_out = nn.LayerNorm(hidden_size)

        self.gate_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    # ----------------- LoRA core -----------------
    def _rank_gates(self, rank_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(rank_logits)  # [r] in (0,1)

    def _lora_linear(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, rank_logits: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in]
        A: [r, in], B: [out, r], rank_logits: [r]
        return: [..., out]
        """
        x_d = self.lora_dropout(x)
        xa = F.linear(x_d, A)                         # [..., r]
        xa = xa * self._rank_gates(rank_logits)       # gate per-rank
        delta = F.linear(xa, B)                       # [..., out]
        return self.lora_scale * delta

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self._lora_linear(x, self.q_lora_A, self.q_lora_B, self.q_rank_logits)

    def _out(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x) + self._lora_linear(x, self.out_lora_A, self.out_lora_B, self.out_rank_logits)

    def _k(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_lora_to == "qkv_out":
            return self.k_proj(x) + self._lora_linear(x, self.k_lora_A, self.k_lora_B, self.k_rank_logits)
        return self.k_proj(x)

    def _v(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_lora_to == "qkv_out":
            return self.v_proj(x) + self._lora_linear(x, self.v_lora_A, self.v_lora_B, self.v_rank_logits)
        return self.v_proj(x)

    # ----------------- forward (same behavior) -----------------
    def forward(
        self,
        hidden,
        mem,
        mem_sep=None,
        mem_mask=None,
        mem_conf=None,
        mem_ctc_mask=None,
        ctc_modules=None,
    ):
        if mem is None:
            return hidden

        h_norm = self.ln_in(hidden)     # [B, Lq, H]
        Q = self._q(h_norm)             # [B, Lq, A]
        K = self._k(mem)                # [B, Tm, A]
        V = self._v(mem)                # [B, Tm, A]

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

        if mem_mask is not None:
            attn_scores = attn_scores.masked_fill(mem_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)   # [B, Lq, A]
        context_h = self._out(context)            # [B, Lq, H]

        out_base = self.ln_out(hidden + context_h)
        delta = out_base - hidden

        g = torch.sigmoid(self.gate_logit)        # scalar
        out = hidden + g * delta
        return out

    # ----------------- scheme-1 helpers (budget + pruning) -----------------
    def rank_usage(self) -> torch.Tensor:
        """
        Soft expected active ranks (sum of sigmoid(rank_logits)).
        For global budget: sum this across all adapters/layers outside.
        """
        usage = self._rank_gates(self.q_rank_logits).sum() + self._rank_gates(self.out_rank_logits).sum()
        if self.apply_lora_to == "qkv_out":
            usage = usage + self._rank_gates(self.k_rank_logits).sum() + self._rank_gates(self.v_rank_logits).sum()
        return usage

    @torch.no_grad()
    def prune_lora_ranks(self, gate_threshold: float = 0.1, keep_at_least: int = 1):
        """
        Physically prune LoRA ranks whose gate < threshold.
        This DOES NOT change base module names/weights.
        """
        def _prune(A, B, logits):
            g = torch.sigmoid(logits)
            keep = g >= gate_threshold
            idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)

            if idx.numel() < keep_at_least:
                topk = torch.topk(g, k=min(keep_at_least, g.numel())).indices
                idx = topk.sort().values

            A_new = nn.Parameter(A[idx].contiguous())
            B_new = nn.Parameter(B[:, idx].contiguous())
            logits_new = nn.Parameter(logits[idx].contiguous())
            return A_new, B_new, logits_new

        self.q_lora_A, self.q_lora_B, self.q_rank_logits = _prune(self.q_lora_A, self.q_lora_B, self.q_rank_logits)
        self.out_lora_A, self.out_lora_B, self.out_rank_logits = _prune(self.out_lora_A, self.out_lora_B, self.out_rank_logits)

        if self.apply_lora_to == "qkv_out":
            self.k_lora_A, self.k_lora_B, self.k_rank_logits = _prune(self.k_lora_A, self.k_lora_B, self.k_rank_logits)
            self.v_lora_A, self.v_lora_B, self.v_rank_logits = _prune(self.v_lora_A, self.v_lora_B, self.v_rank_logits)

        # update r_max and scale to the pruned size (use Q's new size as reference)
        self.r_max = int(self.q_lora_A.size(0))
        self.lora_scale = self.lora_alpha / max(1, self.r_max)

