# Created by Hao at 2025-07-01
# unfreeze_utils.py
# -----------------------------------------------------------
# Minimal, self‑contained logic to unfreeze specific parameters
# for a WavLM encoder + BART decoder style model, mirroring the
# exact behaviour from the original inline code snippet.
#
# Usage:
#   from unfreeze_manual import unfreeze_selected_params
#   unfreeze_selected_params(model, model_args)
#
# -----------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

def checking_trainable_params(model):
    """Quick utility to print the number of trainable parameters."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(f"Total params: {total:,}; Trainable: {trainable:,} ({trainable/total:.2%})")

def print_status(model) -> None:
    """
    Log every parameter’s `requires_grad` flag.

    Example log line:
        encoder.layers.0.self_attn.q_proj.weight is trainable: True
    """
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            logger.info(f"{full_name} is trainable: {param.requires_grad}")

def unfreeze_selected_params(model, train_mode, model_args) -> None:
    """Apply hard‑coded unfreeze rules to *model*.

    Args
    ----
    model:
        The model instance with ``encoder`` and ``decoder`` sub‑modules.
    model_args:
        An argparse/dataclass namespace exposing ``freeze_encoder``.
    """

    # 1) Optionally unfreeze the whole encoder
    if not getattr(model_args, "freeze_encoder", True):
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True
        logger.info("Encoder fully unfrozen (freeze_encoder=False).")

    # 2) Except training only with CTC loss 
    #    Always keep specific encoder parts trainable
    #    The adapter in encoder is the 3-CNN layers downsampling-projector layer
    if(train_mode != 'ctc'):
        for name, param in model.encoder.named_parameters():
            if "adapter" in name or "masked_spec_embed" in name:
                param.requires_grad = True

    for name, param in model.encoder.named_parameters():
        for _partial in model_args.partial_encoder_unfreeze:
            if(_partial in name):
                param.requires_grad = True

    # 3) Keep encoder-decoder projection layers trainable
    if(train_mode != 'ctc') and (not model_args.ctc_bridge):
        for name, param in model.named_parameters():
            if "enc_to_dec_proj" in name:
                param.requires_grad = True

    # 4) Unfreeze other parameters
    for name, param in model.named_parameters():
        for _partial in model_args.partial_others_unfreeze:
            if(_partial in name):
                param.requires_grad = True

    # 5) Unfreeze select decoder components
    for name, param in model.decoder.named_parameters():
        for _partial in model_args.partial_decoder_unfreeze:
            if(_partial in name):
                param.requires_grad = True

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        logger.info("Feature extractor in Encoder is un-freezed.")

    logger.info("unfreeze_selected_params finished.")
    logger.info("=" * 48)
    logger.info("Parameter statistics after unfreezing.")
    checking_trainable_params(model)

    print_status(model)

