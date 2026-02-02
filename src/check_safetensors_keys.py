import os
import re
import json
import argparse
from safetensors import safe_open

def iter_safetensors_files(path: str):
    if os.path.isdir(path):
        files = [
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.endswith(".safetensors")
        ]
        files.sort()
        if not files:
            raise FileNotFoundError(f"No .safetensors found in folder: {path}")
        return files
    else:
        if not path.endswith(".safetensors"):
            raise ValueError(f"Input must be a folder or a .safetensors file: {path}")
        return [path]

def read_all_keys_and_meta(sf_path: str, with_meta: bool):
    """
    Returns:
      keys_sorted: List[str]
      meta: Dict[key -> {shape, dtype}] if with_meta else {}
    """
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    keys_sorted = sorted(keys)
    meta = {}

    if with_meta:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in keys_sorted:
                t = f.get_tensor(k)  # loads tensor; ok for moderate ckpt sizes
                meta[k] = {"shape": list(t.shape), "dtype": str(t.dtype)}

    return keys_sorted, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a .safetensors file OR a folder containing .safetensors files.")
    parser.add_argument("--name", type=str, default=None,
                        help="Exact parameter names to check. Comma-separated, e.g. a,b,c")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Regex pattern to match keys, e.g. 'gate|adapter|lora'")
    parser.add_argument("--contains", type=str, default=None,
                        help="Substring to match keys (case-sensitive).")
    parser.add_argument("--show-meta", action="store_true",
                        help="Also output shape/dtype. (Slower, loads tensors)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional: write full key list (and meta) to JSON file.")
    args = parser.parse_args()

    names = []
    if args.name:
        names = [x.strip() for x in args.name.split(",") if x.strip()]

    regex = re.compile(args.pattern) if args.pattern else None

    files = iter_safetensors_files(args.ckpt)

    # Aggregate across shards
    all_keys = set()
    per_file = {}  # file -> {keys, meta}
    for sf in files:
        keys, meta = read_all_keys_and_meta(sf, with_meta=args.show_meta)
        per_file[sf] = {"keys": keys, "meta": meta}
        all_keys.update(keys)

    all_keys_sorted = sorted(all_keys)

    print(f"\nFound {len(files)} safetensors file(s).")
    for sf in files:
        print(f" - {sf}  (tensors: {len(per_file[sf]['keys'])})")

    print(f"\n=== TOTAL UNIQUE KEYS ACROSS ALL FILES: {len(all_keys_sorted)} ===")

    # 1) Print full content (keys + optional meta)
    print("\n=== FULL KEY LIST ===")
    for k in all_keys_sorted:
        if args.show_meta:
            # find first file that contains it to print meta
            shape_dtype = None
            for sf in files:
                if k in per_file[sf]["meta"]:
                    m = per_file[sf]["meta"][k]
                    shape_dtype = f"shape={tuple(m['shape'])} dtype={m['dtype']}"
                    break
            if shape_dtype:
                print(f"{k}  [{shape_dtype}]")
            else:
                print(k)
        else:
            print(k)

    # 2) Check exact names
    if names:
        print("\n=== EXACT NAME CHECK ===")
        for n in names:
            print(f"[{'FOUND' if n in all_keys else 'MISS '}] {n}")

    # 3) Pattern / substring matches
    if regex or args.contains:
        matched = []
        for k in all_keys_sorted:
            if regex and regex.search(k):
                matched.append(k)
                continue
            if args.contains and (args.contains in k):
                matched.append(k)

        print(f"\n=== MATCHED ({len(matched)}) ===")
        for k in matched:
            if args.show_meta:
                shape_dtype = None
                for sf in files:
                    if k in per_file[sf]["meta"]:
                        m = per_file[sf]["meta"][k]
                        shape_dtype = f"shape={tuple(m['shape'])} dtype={m['dtype']}"
                        break
                if shape_dtype:
                    print(f"{k}  [{shape_dtype}]")
                else:
                    print(k)
            else:
                print(k)

    # 4) Optional JSON dump
    if args.out:
        dump = {
            "files": files,
            "total_unique_keys": len(all_keys_sorted),
            "keys": all_keys_sorted,
        }
        if args.show_meta:
            # merge meta from shards; if duplicates, keep first found
            merged_meta = {}
            for sf in files:
                for k, v in per_file[sf]["meta"].items():
                    if k not in merged_meta:
                        merged_meta[k] = v
            dump["meta"] = merged_meta

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON to: {args.out}")

if __name__ == "__main__":
    main()

