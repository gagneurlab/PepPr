#!/bin/bash
#SBATCH --job-name=regen_fusion_artifacts
#SBATCH --output=/data/nasif12/home_if12/khsam/dnps_hybrid/data/slurm/%j_regen_fusion_artifacts.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Rebuild MassiveKB fusion-training tensors consistent with the existing
# casanovo/casanovo_teacher_*_torch_data.pt (276,950 train / 69,372 test rows):
#   - casanovo/casanovo_teacher_scores_{train,test}.pt
#   - casanovo/fusion_y_{train,test}.pt          (was a stale symlink to ood/)
#   - casanovo/plm_psm_x_{train,test}.pt         (translated Y -> pepLM vocab)
#
# GPU is reserved because const.py's import touches torch.cuda; the work itself
# is mostly disk I/O.

export TMUX="${TMUX:-noninteractive}"
source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
cd /data/nasif12/home_if12/khsam/dnps_hybrid

export DNPS_SPECIES=human
export DNPS_PLM_SPECIES=human

conda activate khsam
python -c "from dnps_hybrid.prepare_data import generate_fusion_files; generate_fusion_files()"

echo "=== Sizes after regen ==="
ls -lh /s/project/denovo-prosit/SamKhan/dnps_hybrid/casanovo/{fusion_y_train,fusion_y_test,casanovo_teacher_scores_train,casanovo_teacher_scores_test,plm_psm_x_train,plm_psm_x_test}.pt
