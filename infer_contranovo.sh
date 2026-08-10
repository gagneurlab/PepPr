#!/bin/bash
#SBATCH --job-name=infer_contranovo
#SBATCH --output=data/slurm/%j_infer_contranovo.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#
# ContraNovo +/- pepLM (PepPr) inference on a nine-species subset, using the
# asymbnln + swap50 ContraNovo fusion head (see train_fusion_contranovo_swap50.sh).
# The fusion head is pepLM/species-independent, so the same head is used for every
# species; only DNPS_SPECIES changes.
#
# Usage: sbatch infer_contranovo.sh <species>     # e.g. human, mouse, ...

source /data/nasif12/home_if12/khsam/.bashrc
set -euo pipefail
set -x

SP="${1:?usage: infer_contranovo.sh <species>}"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
SHARED_DIR=$DNPS_DATA_PATH/casanovo

export DNPS_SPECIES=$SP
export DNPS_CONTRANOVO_FUSION_MODEL_PATH=$SHARED_DIR/contranovo_fusion_asymbnln_swap50.pth
export DNPS_CONTRANOVO_NULL_MODEL_PATH=$SHARED_DIR/contranovo_null_asymbnln_swap50.pth
ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"

conda activate khsam
cd /data/nasif12/home_if12/khsam/dnps_hybrid

echo "=== ContraNovo +/- pepLM for species: $SP (asymbnln+swap50 fusion head) ==="
python dnps_hybrid/inference.py contranovo
echo "=== Done ==="
