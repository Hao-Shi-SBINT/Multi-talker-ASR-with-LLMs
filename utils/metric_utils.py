# Author: Hao at 2025-07-02
# metric_utils.py
# -----------------------------------------------------------
# Helper to load the Word Error Rate (WER) metric via `evaluate`
# and provide a `compute_metrics` callback for Hugging Face Trainer.
#
# Usage:
#   from metric_utils import build_compute_metrics
#   compute_metrics = build_compute_metrics(tokenizer, cache_dir)
#   trainer = Trainer(..., compute_metrics=compute_metrics)
# -----------------------------------------------------------

import evaluate
import numpy as np  # Optional, but HF Trainer sometimes passes numpy arrays

def compute_metrics(tokenizer, cache_dir: str = None, ignore_id: int = -100):
    """
    Return a `compute_metrics` function bound to the given tokenizer.

    Parameters
    ----------
    tokenizer:
        The tokenizer used for decoding model outputs.
    cache_dir: str, optional
        Where to cache the metric files.

    Returns
    -------
    callable
        A function `compute_metrics(pred)` compatible with HF Trainer.
    """
    metric = evaluate.load("wer")

    def computing_metrics(pred):
        """
        Compute WER between predictions and references.

        `pred` is the EvalPrediction object passed by HF Trainer.
        """
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 with pad_token_id to enable decoding
        label_ids[label_ids == ignore_id] = tokenizer.pad_token_id

        # Decode
        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return computing_metrics

