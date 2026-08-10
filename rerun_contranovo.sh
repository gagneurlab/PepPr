#!/bin/bash
#SBATCH --job-name=contranovo_plm
#SBATCH --output=data/slurm/%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26

source /data/nasif12/home_if12/khsam/.bashrc

set -e
set -x

# Sub-modes:
#   $1 = "infer" (default) : run de novo +/- pepLM on the active species
#   $1 = "teacher"         : teacher-force ContraNovo on MassiveKB train+val
#                            (writes contranovo_teacher_{train,val}_torch_data.pt)
#   $1 = "fusion_data"     : run prepare_data step that consumes those .pt files
#                            and emits PLM X tensors + fusion targets
#   $1 = "plm_teacher"     : run pepLM teacher on the ContraNovo-vocab inputs
#                            (must run inside the dnps env, not khsam_contranovo)
#   $1 = "train_fusion"    : train the (ContraNovo + pepLM) fusion model
#
# The full pipeline order is: teacher -> fusion_data -> plm_teacher -> train_fusion -> infer

MODE=${1:-infer}
export DNPS_SPECIES=${2:-human}

CONTRANOVO_ROOT=/data/nasif12/home_if12/khsam/dnps_hybrid/ContraNovo
CONTRANOVO_PYTHON=/opt/modules/i12g/anaconda/envs/khsam_contranovo/bin/python
CONTRANOVO_CKPT=$CONTRANOVO_ROOT/ContraNovo/ContraNovo.ckpt
CONTRANOVO_CONFIG=$CONTRANOVO_ROOT/ContraNovo/config.yaml
SHARED_DIR=/s/project/denovo-prosit/SamKhan/dnps_hybrid/casanovo
TRAIN_MGF_DIR=$SHARED_DIR/fusion_train_set
VAL_MGF_DIR=$SHARED_DIR/fusion_val_set

case "$MODE" in
  teacher)
    # MassiveKB is pre-split into one or more MGF files per directory.
    # ContraNovo loads one MGF at a time, so process each split separately.
    for split in train val; do
      out_pt=$SHARED_DIR/contranovo_teacher_${split}_torch_data.pt
      if [ "$split" = "train" ]; then mgf_dir=$TRAIN_MGF_DIR; else mgf_dir=$VAL_MGF_DIR; fi
      # We assume one .mgf in each split dir; if there are multiple, run
      # run_teacher.py separately and merge by concatenation. For now, just
      # take the first one.
      mgf=$(ls $mgf_dir/*.mgf | head -n1)
      echo "=== Teacher-forcing ContraNovo on $mgf -> $out_pt ==="
      PYTHONPATH=/data/nasif12/home_if12/khsam/dnps_hybrid:$PYTHONPATH \
        $CONTRANOVO_PYTHON $CONTRANOVO_ROOT/run_teacher.py \
          --peak_path=$mgf --model=$CONTRANOVO_CKPT \
          --config=$CONTRANOVO_CONFIG --out=$out_pt
    done
    # CONTRANOVO_TEACHER_*_PT_PATH in const.py expects {train,test}_torch_data.pt
    # filenames; rename val -> test to match.
    if [ -f $SHARED_DIR/contranovo_teacher_val_torch_data.pt ]; then
      mv -f $SHARED_DIR/contranovo_teacher_val_torch_data.pt \
            $SHARED_DIR/contranovo_teacher_test_torch_data.pt
    fi
    ;;

  fusion_data)
    python -c "from dnps_hybrid.prepare_data import generate_contranovo_fusion_files; generate_contranovo_fusion_files()"
    ;;

  plm_teacher)
    # Re-route plm_teacher so it consumes the ContraNovo-vocab X tensors and
    # writes the matching score tensors. inference.py's plm_teacher path uses
    # const.PLM_PSM_X_*_PATH and const.PLM_PSM_TEACHER_SCORES_*_PATH; for the
    # ContraNovo run we override via env vars (or just point a wrapper).
    python -c "
import torch, os
from dnps_hybrid import const
from dnps_hybrid.model import load_plm_model
from tqdm import tqdm

torch.manual_seed(const.SEED); torch.cuda.manual_seed(const.SEED)
torch.set_float32_matmul_precision('high')
model = load_plm_model()
batch_size = 4096

def run(X_path, out_path):
    X = torch.load(X_path, map_location=const.DEVICE)
    scores = torch.zeros(X.shape[0], const.PLM_BLOCK_SIZE, len(const.VOCAB), device=const.DEVICE)
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), total=len(X)//batch_size):
            end = min(i+batch_size, len(X))
            logits, *_ = model(X[i:end])
            scores[i:end] = logits
    torch.save(scores, out_path)
    print('Saved', out_path)

run(const.CONTRANOVO_PLM_PSM_X_TRAIN_PATH, const.CONTRANOVO_PLM_PSM_TEACHER_SCORES_TRAIN_PATH)
run(const.CONTRANOVO_PLM_PSM_X_TEST_PATH,  const.CONTRANOVO_PLM_PSM_TEACHER_SCORES_TEST_PATH)
"
    ;;

  train_fusion)
    DNPS_FUSION_BACKBONE=contranovo python -m dnps_hybrid.train_fusion_head
    ;;

  infer)
    echo "=== ContraNovo +/- pepLM for species: $DNPS_SPECIES ==="
    python dnps_hybrid/inference.py contranovo
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
