#!/bin/bash
#SBATCH --job-name=kol_pride_dl
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/kol_pride_%j.out
#SBATCH --error=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/kol_pride_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH --partition=noninterruptive
#
# Low-footprint download job (I/O bound). Total data ~1.5 TB across PXD014877 + PXD019483;
# extend --time if the cluster enforces shorter limits on this partition.
#
# Submit from anywhere:
#   sbatch /data/nasif12/home_if12/khsam/dnps_hybrid/scripts/download_kingdoms_of_life.slurm
#
# Optional environment overrides:
#   KOL_DEST=/path/to/kingdoms_of_life       (default below)
#   KOL_INCLUDE=RAW,SEARCH,OTHER             (comma-separated PRIDE categories)
#   KOL_PXD="PXD014877 PXD019483"           (space-separated; default both)
#   KOL_DRY_RUN=1                            (list only, no wget)
#   KOL_MAX_FILES=3                          (smoke test: stop after N downloads)

set -euo pipefail

DEST="${KOL_DEST:-/s/project/denovo-prosit/SamKhan/dnps_hybrid/kingdoms_of_life}"
INCLUDE="${KOL_INCLUDE:-RAW,SEARCH,OTHER}"
DRY=( )
if [[ -n "${KOL_DRY_RUN:-}" ]]; then
  DRY=( --dry-run )
fi
MAXF=( )
if [[ -n "${KOL_MAX_FILES:-}" ]]; then
  MAXF=( --max-files "${KOL_MAX_FILES}" )
fi

PXD_ARGS=( )
if [[ -n "${KOL_PXD:-}" ]]; then
  for a in ${KOL_PXD}; do
    PXD_ARGS+=( --pxd "$a" )
  done
fi

mkdir -p "${DEST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/pride_download_kingdoms_of_life.py"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "DEST=${DEST}"
echo "INCLUDE=${INCLUDE}"

if [[ ! -f "$PY" ]]; then
  echo "Missing downloader script: $PY" >&2
  exit 1
fi

python3 "$PY" \
  --dest "$DEST" \
  --include "$INCLUDE" \
  "${PXD_ARGS[@]}" \
  "${DRY[@]}" \
  "${MAXF[@]}"

echo "End: $(date -Is)"
