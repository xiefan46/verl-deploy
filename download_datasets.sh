#!/bin/bash
# download_datasets.sh — fetch parquet datasets verl example scripts expect.
#
# Most verl example scripts hard-code paths like
#   /root/verl/data/DAPO-Math-17k/data/dapo-math-17k.parquet
#   /root/verl/data/AIME-2024/data/aime-2024.parquet
# These come from HF datasets but aren't bundled with model downloads.
#
# Usage:
#   bash download_datasets.sh                     # download defaults (DAPO + AIME)
#   bash download_datasets.sh <hf_dataset_repo>   # one specific dataset
#   DATA_ROOT=/somewhere bash download_datasets.sh
#
# Requires `hf` CLI in PATH (installed in the verl env).

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/verl/data}"

# (HF repo, local dir name under DATA_ROOT)
DEFAULT_DATASETS=(
    "BytedTsinghua-SIA/DAPO-Math-17k|DAPO-Math-17k"
    "BytedTsinghua-SIA/AIME-2024|AIME-2024"
)

log() { printf "[%(%H:%M:%S)T] %s\n" -1 "$*"; }

download_one() {
    local repo="$1" subdir="$2"
    local target="$DATA_ROOT/$subdir"
    if [[ -d "$target" ]] && [[ -n "$(ls -A "$target" 2>/dev/null)" ]]; then
        log "  $repo → $target (already populated, skipping)"
        return
    fi
    log "  $repo → $target"
    hf download --repo-type dataset "$repo" --local-dir "$target"
}

mkdir -p "$DATA_ROOT"

if [[ $# -ge 1 ]]; then
    # Single explicit repo. Subdir = last path component.
    repo="$1"
    subdir="${repo##*/}"
    log "Downloading 1 dataset to $DATA_ROOT"
    download_one "$repo" "$subdir"
else
    log "Downloading ${#DEFAULT_DATASETS[@]} default datasets to $DATA_ROOT"
    for entry in "${DEFAULT_DATASETS[@]}"; do
        repo="${entry%%|*}"
        subdir="${entry##*|}"
        download_one "$repo" "$subdir"
    done
fi

log "Done. Verify expected parquet paths exist:"
for f in \
    "$DATA_ROOT/DAPO-Math-17k/data/dapo-math-17k.parquet" \
    "$DATA_ROOT/AIME-2024/data/aime-2024.parquet"; do
    [[ -f "$f" ]] && echo "  ✓ $f" || echo "  ✗ $f (missing)"
done
