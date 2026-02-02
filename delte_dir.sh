#!/usr/bin/env bash
set -euo pipefail

ROOT="exp_crossatt_adap_finished"
DOIT=0

# 用法:
#   bash delete_ckpt_dirs.sh                 # 只列出将要删除的目录
#   bash delete_ckpt_dirs.sh --doit          # 真正删除
#   bash delete_ckpt_dirs.sh /path/to/root --doit
#   find exp_sot_finished -type d -name 'checkpoint*' -print -exec rm -rf {} +
if [[ $# -ge 1 && "$1" != "--doit" ]]; then
  ROOT="$1"
  shift
fi
if [[ $# -ge 1 && "$1" == "--doit" ]]; then
  DOIT=1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Not a directory: $ROOT" >&2
  exit 1
fi

echo "[INFO] Searching under: $ROOT"
echo "[INFO] Targets: directories named checkpoint*"

mapfile -d '' TARGETS < <(find "$ROOT" -type d -name 'checkpoint*' -print0)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[INFO] No checkpoint* directories found."
  exit 0
fi

echo "[INFO] Found ${#TARGETS[@]} directories:"
for p in "${TARGETS[@]}"; do
  echo "  $p"
done

if [[ $DOIT -eq 0 ]]; then
  echo
  echo "[DRY-RUN] Nothing deleted. Re-run with --doit to delete."
  exit 0
fi

echo
echo "[INFO] Deleting..."
printf '%s\0' "${TARGETS[@]}" | xargs -0 rm -rf --

echo "[INFO] Done."

