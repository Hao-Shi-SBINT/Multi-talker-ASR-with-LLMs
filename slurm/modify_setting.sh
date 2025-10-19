#!/usr/bin/env bash
set -euo pipefail

# Standalone bulk updater for key=value lines in files like submit_*
# Skips lines whose value starts with $ (e.g., stage=${stage})
# Works on GNU/Linux and macOS. No external helper needed.

dir="."
pattern="submit_*"
backup=false
dry_run=false

usage() {
  cat <<\EOF
Usage:
  modify_setting.sh [-d DIR] [-p GLOB] [-b] [--dry-run] -- key1=val1 [key2=val2 ...]
  modify_setting.sh [-d DIR] [-p GLOB] [-b] [--dry-run]    key1=val1 [key2=val2 ...]

Options:
  -d DIR       Target directory (default: .)
  -p GLOB      Filename glob (default: submit_*)
  -b           Make .bak backups
  --dry-run    Show diff; do not modify files
  -h, --help   Show help

Notes:
  - Updates only assignment lines from BOL (allowing indent/export), e.g. "stage=3", "export x=1".
  - Skips lines whose value begins with '$' (e.g., stage=${stage}, export x=$Y).
  - Appends missing vars to EOF.
EOF
}

# --- parse options ---
while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    -d) shift; [[ $# -gt 0 ]] || { echo "Missing arg for -d" >&2; exit 1; }; dir="$1"; shift ;;
    -p) shift; [[ $# -gt 0 ]] || { echo "Missing arg for -p" >&2; exit 1; }; pattern="$1"; shift ;;
    -b) backup=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "Invalid option: $1" >&2; usage; exit 1 ;;
    *) break ;;
  esac
done

# Remaining args = updates (accept with or without `--`)
updates=( "$@" )
if [[ ${#updates[@]} -eq 0 ]]; then
  echo "No key=value updates provided." >&2
  usage; exit 1
fi

# Validate updates are key=value
bad=()
for kv in "${updates[@]}"; do
  [[ "$kv" == *=* ]] || bad+=( "$kv" )
done
if [[ ${#bad[@]} -gt 0 ]]; then
  echo "These arguments are not key=value pairs: ${bad[*]}" >&2
  echo "Tip: add -- before updates if any option is ambiguous." >&2
  usage; exit 1
fi

# Collect files (NUL-safe)
files=()
while IFS= read -r -d '' f; do files+=( "$f" ); done < <(find "$dir" -maxdepth 1 -type f -name "$pattern" -print0 | sort -z)
[[ ${#files[@]} -gt 0 ]] || { echo "No files matched: dir='$dir' pattern='$pattern'." >&2; exit 1; }

# Core updater: for one file apply many key=val edits
update_file() {
  local file="$1"; shift
  local tmp
  tmp="$(mktemp -t modset.XXXXXX)"
  cp "$file" "$tmp.work"

  local kv key val
  for kv in "$@"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    awk -v key="$key" -v newval="$val" '
      BEGIN{
        updated=0
        key_re=key
        gsub(/[][\\.^$*+?(){}|]/,"\\&", key_re)
        pattern="^([[:space:]]*(export[[:space:]]+)?(" key_re ")[[:space:]]*=[[:space:]]*)([^#]*)([[:space:]]*(#.*)?)$"
      }
      {
        line=$0
        if (match(line, pattern, m)) {
          # m[4] is the raw value (may include spaces). Determine first non-space char.
          valtxt=m[4]
          sub(/^[[:space:]]*/, "", valtxt)
          if (valtxt ~ /^\$/) {
            # Value starts with $, treat as reference => do NOT rewrite.
            print line
          } else {
            print m[1] newval m[5]
            updated=1
          }
        } else {
          print line
        }
      }
      END{
        if (!updated) {
          print ""
          print key "=" newval
        }
      }
    ' "$tmp.work" > "$tmp.next"
    mv "$tmp.next" "$tmp.work"
  done
  mv "$tmp.work" "$tmp"
  printf '%s\n' "$tmp"
}

# Run
changed_any=false
for f in "${files[@]}"; do
  new_file="$(update_file "$f" "${updates[@]}")"
  if $dry_run; then
    echo "==> $f"
    if command -v diff >/dev/null 2>&1; then
      diff -u "$f" "$new_file" || true
    else
      echo "(no diff available) updated copy: $new_file"
    fi
    rm -f "$new_file"
  else
    $backup && cp "$f" "$f.bak"
    mv "$new_file" "$f"
    changed_any=true
  fi
done

if ! $dry_run && $changed_any; then
  echo "Done: updated ${#updates[@]} variable(s) across ${#files[@]} file(s)."
fi

