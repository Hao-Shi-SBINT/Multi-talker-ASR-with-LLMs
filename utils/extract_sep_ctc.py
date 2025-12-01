import torch
from pathlib import Path
from safetensors.torch import load_file  # for .safetensors

# We only want to extract these modules
PREFIXES = [
    "separator.",
    "serialized_ctc.",
]


def load_state_dict(path: str):
    """
    Load a state_dict from a checkpoint file and normalize keys.

    Supports:
      - PyTorch checkpoints (.pt, .bin, etc.) via torch.load
      - Safetensors files (.safetensors) via safetensors.torch.load_file
    """
    path_obj = Path(path)
    suffix = path_obj.suffix

    if suffix == ".safetensors":
        # Safetensors stores a plain dict[str, Tensor] (no nested "state_dict" key)
        sd = load_file(path_obj)
    else:
        # Fallback to torch.load for typical PyTorch checkpoints
        ckpt = torch.load(path_obj, map_location="cpu")

        # Handle common checkpoint formats
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

    # Remove "module." prefix if saved under DDP
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def extract_sep_ctc(sd):
    """
    Filter out only separator and serialized_ctc parameters.
    """
    part = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in PREFIXES):
            part[k] = v
    return part


def main(src_ckpt, dst_path):
    sd = load_state_dict(src_ckpt)
    part_sd = extract_sep_ctc(sd)

    print(f"Total params in src: {len(sd)}")
    print(f"Extracted separator + serialized_ctc params: {len(part_sd)}")

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(part_sd, dst_path)
    print(f"Saved partial state_dict to: {dst_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_ckpt", type=str, required=True,
        help="Path to the trained model checkpoint (.pt/.bin/.safetensors)"
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="Where to save the extracted separator+ctc state_dict (.pt)"
    )
    args = parser.parse_args()
    main(args.src_ckpt, args.out_path)

