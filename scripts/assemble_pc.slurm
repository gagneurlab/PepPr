#!/bin/bash
#SBATCH --job-name=assemble_pc
#SBATCH --output=data/slurm/%j_assemble_pc.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga03,ouga08,ouga09,ouga05,ouga06,ouga26
#
# Assemble mAbs and plot 3-arm PC curves, for one species' mAb set.
#
#   sbatch scripts/assemble_pc.slurm <human|mouse>

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

case "${1:?usage: assemble_pc.slurm <human|mouse>}" in
  human) SP=human; MABS="S2P6 IgG1_Human_H IgG1_Human_L" ;;
  mouse) SP=mouse; MABS="2B4 36H6 85F7" ;;
  *) echo "unknown species '$1' (expected human | mouse)" >&2; exit 2 ;;
esac

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi

echo "=== Assembly ($SP mAbs: $MABS) ==="
echo "Host:  $(hostname)"
echo "Start: $(date -Is)"
"$PYTHON_BIN" -m dnps_hybrid.assembly --mabs $MABS

echo
echo "=== PC curves ($SP mAbs) ==="
"$PYTHON_BIN" xa_novo/plot_pc_3arm_mabs.py \
    --mabs $MABS \
    --out "xa_novo/pc_3arm_${SP}_mabs.png"

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh xa_novo/alps_2arm_all/2arm_summary.tsv "xa_novo/pc_3arm_${SP}_mabs.png"
