#!/usr/bin/env python
import argparse
import os
from safetensors.torch import safe_open, save_file


VALID_FORMATS = {"pt", "tf", "flax", "mlx"}


def fix_safetensors_if_needed(path: str, default_format: str = "pt") -> None:
    """
    Check a safetensors file's metadata. If metadata is missing or invalid,
    reload tensors and save again with metadata['format'] = default_format.
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return

    print(f"[INFO] Inspecting safetensors file: {path}")
    # Read tensors + metadata
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        keys = list(f.keys())

        # Determine whether we need to fix metadata
        if meta is None:
            print("[INFO] metadata is None -> need to fix")
            needs_fix = True
            meta = {}
        else:
            file_format = meta.get("format")
            if file_format in VALID_FORMATS:
                print(f"[INFO] metadata['format'] = {file_format!r}, valid -> no fix needed")
                needs_fix = False
            else:
                print(f"[INFO] metadata['format'] is {file_format!r}, invalid -> need to fix")
                needs_fix = True

        if not needs_fix:
            return

        # Load all tensors into memory
        tensors = {k: f.get_tensor(k) for k in keys}

    # Update metadata and rewrite file
    meta["format"] = default_format
    tmp_path = path + ".tmp"

    print(f"[INFO] Rewriting safetensors with metadata: format={default_format!r}")
    save_file(tensors, tmp_path, metadata=meta)
    os.replace(tmp_path, path)
    print("[INFO] Fix done.")


def main():
    parser = argparse.ArgumentParser(
        description="Fix safetensors metadata if missing/invalid (add metadata['format'])."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory that contains model.safetensors",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model.safetensors",
        help="Safetensors filename in output_dir (default: model.safetensors)",
    )
    parser.add_argument(
        "--default_format",
        type=str,
        default="pt",
        help="Value to set for metadata['format'] if missing/invalid (default: pt)",
    )
    args = parser.parse_args()

    model_path = os.path.join(args.output_dir, args.model_name)
    fix_safetensors_if_needed(model_path, default_format=args.default_format)


if __name__ == "__main__":
    main()

