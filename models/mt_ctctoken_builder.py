import torch
from torch import nn, Tensor
from typing import List, Tuple


class MultiSpkCTCTokenBuilder(nn.Module):
    """
    从多说话人的 sep_hidden + CTC 模块，构建 token-level 声学 memory：

      sep_hidden_list: K * [B, T, D]
      encoder_attention_mask_ctc: [B, T], True=有效, False=pad
      ctc_modules: K 个 CTC 模块 (比如 self.serialized_ctc)

    输出:
      acoustic_mem:  [B, L_total, D]   # token-level feature，已按 speaker 顺序 concat
      acoustic_mask:[B, L_total]       # True=padding
      acoustic_conf:[B, L_total]       # 0~1, CTC 置信度 (1 - mean p_blank_seg)
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _ctc_forward(
        self,
        ctc_module: nn.Module,
        sep_hidden: Tensor,   # [B, T, D]
    ) -> Tuple[Tensor, int]:
        """
        用 CTC 模块算 log_probs 和 blank_id。

        你给的接口是:
            def log_softmax(self, hs_pad):  # hs_pad: [B,T,eprojs]
                return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

        所以这里直接调 ctc_module.log_softmax(hs_pad)。

        返回:
          log_probs: [B, T, V]
          blank_id:  int
        """
        # 1) 直接用 CTC 自带的 log_softmax
        log_probs = ctc_module.log_softmax(sep_hidden)  # [B, T, V]

        # 2) 取 blank id
        if hasattr(ctc_module, "blank_id"):
            blank_id = int(ctc_module.blank_id)
        elif hasattr(ctc_module, "ctc_loss") and hasattr(ctc_module.ctc_loss, "blank"):
            blank_id = int(ctc_module.ctc_loss.blank)
        else:
            # fallback：最后一个维度作为 blank
            blank_id = log_probs.size(-1) - 1

        return log_probs, blank_id

    def _build_one_speaker(
        self,
        sep_hidden: Tensor,          # [B, T, D]
        enc_mask: Tensor,            # [B, T], True=valid
        ctc_module: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        对单个 speaker:
          - CTC argmax path 做 segment
          - 每个 segment 用 sep_hidden 平均
          - conf = 1 - mean(p_blank_seg)

        返回:
          tok_feats: [B, L_max, D]
          tok_mask:  [B, L_max] (True=padding)
          tok_conf:  [B, L_max] (0~1)
        """
        B, T, D = sep_hidden.shape
        device = sep_hidden.device

        # 1) CTC 前向，拿到 log_probs & blank id（无梯度）
        log_probs, blank_id = self._ctc_forward(ctc_module, sep_hidden)  # [B,T,V]

        # 2) argmax path & p_blank
        path = log_probs.argmax(dim=-1)                  # [B, T]
        p_blank = log_probs[..., blank_id].exp()         # [B, T]

        all_feats: List[Tensor] = []
        all_conf: List[Tensor] = []
        lengths: List[int] = []

        for b in range(B):
            feats_b = []
            conf_b = []

            prev_token = None
            current_indices: List[int] = []

            mask_b = enc_mask[b]  # [T], True=valid

            for t in range(T):
                if not bool(mask_b[t]):
                    break

                tok = int(path[b, t])

                if tok == blank_id:
                    # 遇到 blank：结束当前 segment
                    if current_indices:
                        seg = sep_hidden[b, current_indices]        # [L_seg, D]
                        seg_feat = seg.mean(dim=0)                  # [D]

                        p_blank_seg = p_blank[b, current_indices].mean()
                        conf_seg = float(1.0 - p_blank_seg)         # 0~1

                        feats_b.append(seg_feat)
                        conf_b.append(conf_seg)
                        current_indices = []
                    prev_token = None
                    continue

                # 非 blank
                if prev_token is None or tok != prev_token:
                    # 开新 segment（这里假设 blank 已经在上面 flush 掉）
                    current_indices = [t]
                    prev_token = tok
                else:
                    current_indices.append(t)

            # 序列末尾还有 segment，flush 一次
            if current_indices:
                seg = sep_hidden[b, current_indices]
                seg_feat = seg.mean(dim=0)
                p_blank_seg = p_blank[b, current_indices].mean()
                conf_seg = float(1.0 - p_blank_seg)
                feats_b.append(seg_feat)
                conf_b.append(conf_seg)

            lengths.append(len(feats_b))

            if feats_b:
                all_feats.append(torch.stack(feats_b, dim=0))              # [L_b, D]
                all_conf.append(torch.tensor(conf_b, device=device))       # [L_b]
            else:
                all_feats.append(sep_hidden.new_zeros((0, D)))
                all_conf.append(sep_hidden.new_zeros((0,), device=device))

        # 3) pad 到 batch 维
        max_L = max(lengths) if lengths else 0
        tok_feats = sep_hidden.new_zeros((B, max_L, D))
        tok_conf  = sep_hidden.new_zeros((B, max_L))
        tok_mask  = torch.ones(B, max_L, dtype=torch.bool, device=device)  # True=padding

        for b in range(B):
            L_b = lengths[b]
            if L_b == 0:
                continue
            tok_feats[b, :L_b] = all_feats[b]
            tok_conf[b, :L_b]  = all_conf[b].clamp(0.0, 1.0)
            tok_mask[b, :L_b]  = False  # False=非padding

        return tok_feats, tok_mask, tok_conf

    def forward(
        self,
        sep_hidden_list: List[Tensor],          # K * [B, T, D]
        encoder_attention_mask_ctc: Tensor,     # [B, T], True=valid
        ctc_modules: List[nn.Module],          # K * CTC module
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        多说话人：
          - 对每个 speaker 调 _build_one_speaker
          - 然后在 token 维 concat

        返回:
          acoustic_mem:  [B, L_total, D]
          acoustic_mask: [B, L_total]  True=padding
          acoustic_conf: [B, L_total]  0~1
        """
        K = len(sep_hidden_list)
        assert len(ctc_modules) == K, "ctc_modules 数量要和 sep_hidden_list 一致"

        mem_list: List[Tensor] = []
        mask_list: List[Tensor] = []
        conf_list: List[Tensor] = []

        for k in range(K):
            sep_k = sep_hidden_list[k]              # [B, T, D]
            ctc_k = ctc_modules[k]

            tok_feats_k, tok_mask_k, tok_conf_k = self._build_one_speaker(
                sep_hidden=sep_k,
                enc_mask=encoder_attention_mask_ctc,
                ctc_module=ctc_k,
            )

            mem_list.append(tok_feats_k)    # [B, L_k, D]
            mask_list.append(tok_mask_k)    # [B, L_k]
            conf_list.append(tok_conf_k)    # [B, L_k]

        acoustic_mem  = torch.cat(mem_list, dim=1)   # [B, sum(L_k), D]
        acoustic_mask = torch.cat(mask_list, dim=1)  # [B, sum(L_k)]
        acoustic_conf = torch.cat(conf_list, dim=1)  # [B, sum(L_k)]

        return acoustic_mem, acoustic_mask, acoustic_conf

