#!/bin/bash
#SBATCH --job-name=plm_human_iso
#SBATCH --output=data/slurm/%j.out
#SBATCH --time=120:00:00
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


set -e
set -x

# I=L (merged tokens); SPECIES=human_iso routes FASTA_PATH to
# DATA_PATH/human_iso.fasta and all PLM artifacts to DATA_PATH/human_iso/.
export DNPS_PLM_DISTINGUISH_IL=0
export DNPS_SPECIES=human_iso
export DNPS_PLM_SPECIES=human_iso

"$PYTHON_BIN" -c "
import os
from dnps_hybrid import const
from dnps_hybrid.prepare_data import generate_plm_training_data
print(f'FASTA_PATH={const.FASTA_PATH}')
print(f'PLM_SEQ_X_PATH={const.PLM_SEQ_X_PATH}')
print(f'PLM_SEQ_Y_PATH={const.PLM_SEQ_Y_PATH}')
print(f'PLM_SEQ_COUNTS_PATH={const.PLM_SEQ_COUNTS_PATH}')
print(f'PLM_CHECKPOINT_PATH={const.PLM_CHECKPOINT_PATH}')
print(f'len(VOCAB)={len(const.VOCAB)} (expect 23 for I=L)')
if not (os.path.exists(const.PLM_SEQ_X_PATH) and os.path.exists(const.PLM_SEQ_Y_PATH)):
    generate_plm_training_data()
else:
    print('PLM training data already exists, skipping prep')
"

"$PYTHON_BIN" dnps_hybrid/train_peptide_prior_model.py
