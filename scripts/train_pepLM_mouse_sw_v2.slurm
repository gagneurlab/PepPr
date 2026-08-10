#!/bin/bash
#SBATCH --job-name=train_pepLM_mouse_sw_v2
#SBATCH --output=data/slurm/%j_train_pepLM_mouse_sw_v2.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Train mouse sw pepLM v2 from the unique-first 2M training data (I/L disambig).
# Reads:   germline_mouse/plm_seq_x_il_sw_v2.pt
# Writes:  germline_mouse/plm_ckpt_il_sw_v2.pt
#
# Submit after gen_plm_data_mouse_sw_v2.slurm completes:
#   sbatch --dependency=afterok:<data_job_id> train_pepLM_mouse_sw_v2.slurm

set -eo pipefail
cd "$REPO_ROOT"

export DNPS_SPECIES=germline_mouse
export DNPS_PLM_SPECIES=germline_mouse
export DNPS_PLM_DATA_SUFFIX=_sw_v2
export DNPS_PLM_DISTINGUISH_IL=1
export DNPS_PLM_CKPT_PATH=$DNPS_DATA_PATH/germline_mouse/plm_ckpt_il_sw_v2.pt
export DNPS_PLM_MAX_ITERS=100000

echo "=== Pre-flight ==="
echo "Host:   $(hostname)"
echo "Start:  $(date -Is)"
echo "Data:   $(python3 -c 'import sys; sys.path.insert(0,"."); from dnps_hybrid import const; print(const.PLM_SEQ_X_PATH)')"
echo "IL:     ${DNPS_PLM_DISTINGUISH_IL:-0}"
echo "Ckpt:   $DNPS_PLM_CKPT_PATH"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
python3 dnps_hybrid/train_peptide_prior_model.py

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh "$DNPS_PLM_CKPT_PATH"
