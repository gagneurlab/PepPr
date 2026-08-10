#!/bin/bash
#SBATCH --job-name=plm_antibody_mouse
#SBATCH --output=data/slurm/%j_plm_antibody_mouse.out
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
# Germline mouse pepLM trained on the clean IMGT corpus.
#
# Corpus: antibody_mouse.fasta — HC emits V-REGION (FR1→FR3-end Cys)
# and J·C (FR4-start→end of constant) as separate records. NO V·D·J·C
# recombination, NO N-additions, NO CDR3. LC kept as V·J·C since LC CDR3 is
# conserved. Built by dnps_hybrid/build_antibody_db.py.
# Total mouse records: ~818 (vs 1.2M in the full V·D·J·C corpus).
#
# add_rand_suffix uses per-sample uniform rand_N in [0, pep_lens) so the model
# learns to recover from any-length OOD prefix, not just the last 5 tokens.

set -eo pipefail
set -x

export DNPS_PLM_DISTINGUISH_IL=0
export DNPS_SPECIES=antibody_mouse
export DNPS_PLM_SPECIES=antibody_mouse
export DNPS_PLM_PROTEASES=sliding
export DNPS_PLM_MAX_PEP_LEN=25
export DNPS_PLM_MAX_ITERS=100000
export DNPS_PLM_INIT_FROM_CHECKPOINT=0
export DNPS_PLM_RAND_SUFFIX_FULL_LEN=1

DATA_DIR=$DNPS_DATA_PATH/antibody_mouse
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
echo "Done. Mouse antibody pepLM checkpoint: $DATA_DIR/plm_ckpt.pt"
