#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract a subset of tensors (encoder/separator/serialized_ctc) from a .safetensors file
and save to a new .safetensors with Transformers-compatible metadata.

Usage:
  python extract_sub_safetensors.py \
    --ckpt_path exp_ctc_finished/.../model.safetensors \
    --save_path encoder_ckpt/encoder_1b_noisy.safetensors \
    --keep_prefix encoder. separator. serialized_ctc.

If your keys are like "self.encoder.xxx", use:
  --keep_prefix self.encoder. self.separator. self.serialized_ctc.
"""

import os
import sys
import argparse
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file


def strip_common_wrappers(key: str) -> str:
    """Strip common wrappers added by DDP/Lightning/custom wrappers."""
    for p in ("module.", "model.", "_forward_module.", "wrapped_module."):
        if key.startswith(p):
            return key[len(p):]
    return key


def extract_tensors(
    ckpt_path: str,
    keep_prefix: tuple[str, ...],
    strip_wrappers: bool = True,
) -> tuple[OrderedDict, list[str]]:
    """Load safetensors and keep only keys that start with any prefix in keep_prefix."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    state = load_file(ckpt_path)  # dict[str, Tensor], loaded on CPU by default

    kept = OrderedDict()
    dropped = []

    for k, v in state.items():
        kk = strip_common_wrappers(k) if strip_wrappers else k
        if kk.startswith(keep_prefix):
            kept[kk] = v.detach().cpu()
        else:
            dropped.append(kk)

    return kept, dropped


def save_safetensors_pt(
    tensors: OrderedDict,
    save_path: str,
    extra_metadata: dict | None = None,
):
    """Save tensors to safetensors with mandatory metadata for Transformers compatibility."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    metadata = {"format": "pt"}  # <-- critical for transformers from_pretrained()
    if extra_metadata:
        # safetensors metadata values must be strings
        for k, v in extra_metadata.items():
            metadata[str(k)] = str(v)

    save_file(tensors, save_path, metadata=metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Input .safetensors path")
    parser.add_argument("--save_path", type=str, required=True, help="Output .safetensors path")
    parser.add_argument(
        "--keep_prefix",
        type=str,
        nargs="+",
        default=["encoder.", "separator.", "serialized_ctc."],
        help="List of key prefixes to keep. Example: encoder. separator. serialized_ctc.",
    )
    parser.add_argument(
        "--strip_wrappers",
        action="store_true",
        help="Strip common wrappers like module./model. before matching prefixes",
    )
    parser.add_argument(
        "--print_keys",
        action="store_true",
        help="Print first N keys and exit (for debugging prefixes).",
    )
    parser.add_argument("--print_n", type=int, default=80, help="How many keys to print if --print_keys")

    args = parser.parse_args()

    # Debug: inspect keys
    if args.print_keys:
        sd = load_file(args.ckpt_path)
        keys = list(sd.keys())
        print(f"[INFO] total keys: {len(keys)}")
        for i, k in enumerate(keys[: args.print_n]):
            print(f"{i:04d}: {k}")
        return

    keep_prefix = tuple(args.keep_prefix)

    kept, dropped = extract_tensors(
        ckpt_path=args.ckpt_path,
        keep_prefix=keep_prefix,
        strip_wrappers=args.strip_wrappers,
    )

    if len(kept) == 0:
        # 给你一个更明确的提示：前缀没匹配上
        sd = load_file(args.ckpt_path)
        keys = list(sd.keys())
        example = "\n".join(keys[:50])
        raise RuntimeError(
            "No tensors were kept. Your --keep_prefix probably doesn't match.\n"
            f"Given keep_prefix={keep_prefix}\n"
            "Here are first 50 keys in the file:\n"
            f"{example}\n"
            "Try using prefixes like: self.encoder. / model.encoder. / module.encoder. etc."
        )

    # Save with mandatory metadata: format=pt
    save_safetensors_pt(
        tensors=kept,
        save_path=args.save_path,
        extra_metadata={
            "extracted_from": os.path.basename(args.ckpt_path),
            "kept_prefix": ",".join(keep_prefix),
            "num_tensors": str(len(kept)),
        },
    )

    print(f"[OK] loaded: {args.ckpt_path}")
    print(f"[OK] saved : {args.save_path}")
    print(f"[OK] kept tensors   : {len(kept)}")
    print(f"[OK] dropped tensors: {len(dropped)}")
    print("[OK] kept key examples:")
    for k in list(kept.keys())[:40]:
        print("  ", k)


if __name__ == "__main__":
    main()

