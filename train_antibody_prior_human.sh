#!/bin/bash
#SBATCH --job-name=plm_antibody_human
#SBATCH --output=data/slurm/%j_plm_antibody_human.out
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi
#
# Germline human pepLM trained on the clean IMGT corpus.
# See train_antibody_prior_mouse.sh for design notes.
# Total human records: ~1,274.

set -eo pipefail
set -x

export DNPS_PLM_DISTINGUISH_IL=0
export DNPS_SPECIES=antibody_human
export DNPS_PLM_SPECIES=antibody_human
export DNPS_PLM_PROTEASES=sliding
export DNPS_PLM_MAX_PEP_LEN=25
export DNPS_PLM_MAX_ITERS=100000
export DNPS_PLM_INIT_FROM_CHECKPOINT=0
export DNPS_PLM_RAND_SUFFIX_FULL_LEN=1

DATA_DIR=$DNPS_DATA_PATH/antibody_human
mkdir -p "$DATA_DIR"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi

"$PYTHON_BIN" -c "
from dnps_hybrid import const
from dnps_hybrid.prepare_data import generate_plm_training_data
print(f'FASTA_PATH={const.FASTA_PATH}')
print(f'PLM_SEQ_X_PATH={const.PLM_SEQ_X_PATH}')
generate_plm_training_data()
"

"$PYTHON_BIN" dnps_hybrid/train_peptide_prior_model.py

ls -lh "$DATA_DIR"
echo "Done. Human antibody pepLM checkpoint: $DATA_DIR/plm_ckpt.pt"
