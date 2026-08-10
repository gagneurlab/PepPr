#!/bin/bash
#SBATCH --job-name=gen_plm_human_sw_v2
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_gen_plm_human_sw_v2.out
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga03,ouga08,ouga09,ouga05,ouga06,ouga26
#
# Generate germline_human sw pepLM training data (v2: unique-first sampling,
# 2M buffer, I/L disambiguation).
# Writes:
#   germline_human/plm_seq_x_il_sw_v2.pt
#   germline_human/plm_seq_y_il_sw_v2.pt
#   germline_human/plm_seq_counts_il_sw_v2.pkl
#
# Followed by: train_pepLM_human_sw_v2.slurm

source /data/nasif12/home_if12/khsam/.bashrc
set -eo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

export DNPS_SPECIES=germline_human
export DNPS_PLM_SPECIES=germline_human
export DNPS_PLM_PROTEASES=sliding
export DNPS_PLM_BUFFER_SIZE=2000000
export DNPS_PLM_DATA_SUFFIX=_sw_v2
export DNPS_PLM_DISTINGUISH_IL=1

echo "=== Pre-flight ==="
echo "Host:   $(hostname)"
echo "Start:  $(date -Is)"
echo "FASTA:  $(python3 -c 'import sys; sys.path.insert(0,"."); from dnps_hybrid import const; print(const.FASTA_PATH)')"
echo "Output: $(python3 -c 'import sys; sys.path.insert(0,"."); from dnps_hybrid import const; print(const.PLM_SEQ_X_PATH)')"

conda activate khsam
python3 -c "
from dnps_hybrid.prepare_data import generate_plm_training_data
generate_plm_training_data()
"

echo "=== Done ==="
echo "End: $(date -Is)"
ls -lh /s/project/denovo-prosit/SamKhan/dnps_hybrid/germline_human/plm_seq_{x,y}_il_sw_v2.pt \
        /s/project/denovo-prosit/SamKhan/dnps_hybrid/germline_human/plm_seq_counts_il_sw_v2.pkl
