import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _align_ctc_output(src: torch.Tensor, dst: torch.Tensor, key: str) -> torch.Tensor:
    """
    Align CTC output layer parameters when src and dst have different vocab sizes.

    Assumptions:
      - src and dst share the same ordering for the overlapping token ids.
      - Extra tokens only appear at the tail of src or dst.

    Strategy:
      - Let n = min(src_vocab, dst_vocab) = min(src.size(0), dst.size(0)).
      - Copy src[:n] -> dst[:n].
      - If src_vocab > dst_vocab: extra tokens in src are dropped.
      - If dst_vocab > src_vocab: dst's extra rows/elements remain as originally initialized.

    Works for:
      - bias:   shape = [V]
      - weight: shape = [V, H]
    """
    if src.ndim == 1:
        # bias vector: [V]
        n = min(src.size(0), dst.size(0))
        logger.info(f"[align_ctc] {key}: copying {n} elements (src={src.size(0)}, dst={dst.size(0)})")
        dst[:n] = src[:n]
        return dst

    elif src.ndim == 2:
        # weight matrix: [V, H]
        if src.size(1) != dst.size(1):
            logger.warning(
                f"[align_ctc] hidden size mismatch for {key}: "
                f"src_hidden={src.size(1)}, dst_hidden={dst.size(1)}. Skip alignment."
            )
            return dst

        n = min(src.size(0), dst.size(0))
        logger.info(
            f"[align_ctc] {key}: copying {n} rows (src_vocab={src.size(0)}, dst_vocab={dst.size(0)})"
        )
        dst[:n, :] = src[:n, :]
        return dst

    else:
        logger.warning(f"[align_ctc] Unexpected tensor ndim for {key}: {src.ndim}, skipping alignment.")
        return dst


def load_sep_ctc_from_partial(model, ckpt_path: str):
    """
    Load pre-trained separator + serialized_ctc weights into the model.

    ckpt_path should point to a partial state_dict file that contains keys like:
      - "separator.*"
      - "serialized_ctc.0.*"
      - "serialized_ctc.1.*"
      - ...

    Behavior:
      - If name and shape match exactly: load directly.
      - If name matches but shape differs AND key contains "ctc_lo":
          -> do partial alignment along the vocab dimension using _align_ctc_output().
      - Otherwise: skip and log.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"pretrain_separator_path not found: {ckpt_path}")

    partial_sd = torch.load(ckpt_path, map_location="cpu")

    model_sd = model.state_dict()
    matched = {}
    skipped = []

    for k, v in partial_sd.items():
        if k in model_sd:
            dst = model_sd[k]

            # Case 1: shapes are identical -> direct copy
            if dst.shape == v.shape:
                matched[k] = v

            # Case 2: shapes differ, but this is a CTC output layer we want to align
            elif "ctc_lo" in k and dst.ndim in (1, 2):
                logger.info(
                    f"[load_sep_ctc] aligning CTC output for {k}: "
                    f"src={tuple(v.shape)}, dst={tuple(dst.shape)}"
                )
                aligned = _align_ctc_output(v, dst.clone(), k)
                matched[k] = aligned

            # Case 3: other mismatches -> skip
            else:
                skipped.append((k, v.shape, dst.shape))
        else:
            # Key does not exist in current model
            skipped.append((k, v.shape, None))

    logger.info(f"[load_sep_ctc] matched params: {len(matched)}, skipped: {len(skipped)}")
    if skipped:
        logger.warning(
            "Some keys were skipped when loading separator+CTC "
            "(name not found or shape mismatch). Showing first few:"
        )
        for k, src_shape, dst_shape in skipped[:10]:
            logger.warning(f"  - {k}: src={src_shape}, dst={dst_shape}")

    # Apply updates
    model_sd.update(matched)
    model.load_state_dict(model_sd)

    return model

