#!/bin/bash
#SBATCH --job-name=gen_plm_sw_v2
#SBATCH --output=data/slurm/%j_gen_plm_sw_v2.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga03,ouga08,ouga09,ouga05,ouga06,ouga26
#
# Generate germline sliding-window pepLM training data (v2: unique-first
# sampling, 2M buffer, I/L disambiguation) for one species.
#
#   sbatch scripts/gen_plm_data_sw_v2.slurm <human|mouse>
#
# Writes germline_<species>/plm_seq_{x,y}_il_sw_v2.pt + plm_seq_counts_il_sw_v2.pkl
# Followed by: train_pepLM_sw_v2.slurm <species>

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

case "${1:?usage: gen_plm_data_sw_v2.slurm <human|mouse>}" in
  human|mouse) SPECIES="germline_$1" ;;
  *) echo "unknown species '$1' (expected human | mouse)" >&2; exit 2 ;;
esac

export DNPS_SPECIES="$SPECIES"
export DNPS_PLM_SPECIES="$SPECIES"
export DNPS_PLM_PROTEASES=sliding
export DNPS_PLM_BUFFER_SIZE=2000000
export DNPS_PLM_DATA_SUFFIX=_sw_v2
export DNPS_PLM_DISTINGUISH_IL=1

echo "=== Pre-flight ==="
echo "Host:    $(hostname)"
echo "Start:   $(date -Is)"
echo "Species: $SPECIES"
echo "FASTA:   $("$PYTHON_BIN" -c 'from dnps_hybrid import const; print(const.FASTA_PATH)')"
echo "Output:  $("$PYTHON_BIN" -c 'from dnps_hybrid import const; print(const.PLM_SEQ_X_PATH)')"

if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
"$PYTHON_BIN" -c "from dnps_hybrid.prepare_data import generate_plm_training_data; generate_plm_training_data()"

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh "$DNPS_DATA_PATH/$SPECIES"/plm_seq_{x,y}_il_sw_v2.pt \
       "$DNPS_DATA_PATH/$SPECIES/plm_seq_counts_il_sw_v2.pkl"
