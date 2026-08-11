#!/bin/bash
#SBATCH --job-name=plm
#SBATCH --output=data/slurm/%j_plm.out
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Train a peptide-prior (pepLM) model for one species. Generates the tryptic
# training corpus from that species' reference proteome (if not already on disk)
# and then trains the model. Tokens are I=L merged (DNPS_PLM_DISTINGUISH_IL=0).
#
# Usage:  sbatch run_plm.sh <species>          # e.g. human_iso, yeast, mouse, ...
#
# Species -> FASTA + artifact paths are resolved by dnps_hybrid.const from
# DNPS_SPECIES; see the SPECIES dict there.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -e
set -x
cd "$REPO_ROOT"

SP="${1:?usage: run_plm.sh <species>}"
export DNPS_PLM_DISTINGUISH_IL=0
export DNPS_SPECIES="$SP"
export DNPS_PLM_SPECIES="$SP"

"$PYTHON_BIN" -c "
import os
from dnps_hybrid import const
from dnps_hybrid.prepare_data import generate_plm_training_data
os.makedirs(const.PLM_RUN_PATH, exist_ok=True)
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
