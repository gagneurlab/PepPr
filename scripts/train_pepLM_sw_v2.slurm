#!/bin/bash
#SBATCH --job-name=train_pepLM_sw_v2
#SBATCH --output=data/slurm/%j_train_pepLM_sw_v2.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Train a germline sliding-window pepLM (v2: unique-first 2M, I/L disambig) for
# one species, from data built by gen_plm_data_sw_v2.slurm.
#
#   sbatch scripts/train_pepLM_sw_v2.slurm <human|mouse>
#
# Reads:  germline_<species>/plm_seq_x_il_sw_v2.pt
# Writes: germline_<species>/plm_ckpt_il_sw_v2.pt

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -eo pipefail
cd "$REPO_ROOT"

case "${1:?usage: train_pepLM_sw_v2.slurm <human|mouse>}" in
  human|mouse) SPECIES="germline_$1" ;;
  *) echo "unknown species '$1' (expected human | mouse)" >&2; exit 2 ;;
esac

export DNPS_SPECIES="$SPECIES"
export DNPS_PLM_SPECIES="$SPECIES"
export DNPS_PLM_DATA_SUFFIX=_sw_v2
export DNPS_PLM_DISTINGUISH_IL=1
export DNPS_PLM_CKPT_PATH="$DNPS_DATA_PATH/$SPECIES/plm_ckpt_il_sw_v2.pt"
export DNPS_PLM_MAX_ITERS=100000

echo "=== Pre-flight ==="
echo "Host:    $(hostname)"
echo "Start:   $(date -Is)"
echo "Species: $SPECIES"
echo "Data:    $("$PYTHON_BIN" -c 'from dnps_hybrid import const; print(const.PLM_SEQ_X_PATH)')"
echo "Ckpt:    $DNPS_PLM_CKPT_PATH"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
"$PYTHON_BIN" dnps_hybrid/train_peptide_prior_model.py

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh "$DNPS_PLM_CKPT_PATH"
