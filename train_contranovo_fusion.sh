#!/bin/bash
#SBATCH --job-name=train_contranovo_fusion
#SBATCH --output=data/slurm/%j_train_contranovo_fusion.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga25,ouga26
#
# ContraNovo + pepLM (PepPr) fusion-head pipeline. Produces the asymbnln + swap50
# ContraNovo fusion head used by `run_baseline.sh contranovo`.
#
# Usage:  sbatch train_contranovo_fusion.sh <mode>
#
#   teacher      teacher-force ContraNovo on MassiveKB train+val -> teacher .pt
#   fusion_data  build PLM X tensors + fusion targets from the teacher .pt
#   plm_teacher  run the pepLM teacher on the ContraNovo-vocab inputs
#   train_fusion train the (ContraNovo + pepLM) fusion head          [default]
#
# Full order: teacher -> fusion_data -> plm_teacher -> train_fusion.
# (De novo inference with the trained head lives in run_baseline.sh contranovo.)
#
# The teacher pass must run under the ContraNovo conda env; override its
# interpreter with CONTRANOVO_PYTHON (default: python).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
set -x
cd "$REPO_ROOT"

MODE=${1:-train_fusion}

SHARED_DIR=$DNPS_DATA_PATH/casanovo
CONTRANOVO_ROOT=$REPO_ROOT/ContraNovo
CONTRANOVO_PYTHON=${CONTRANOVO_PYTHON:-python}
CONTRANOVO_CKPT=$CONTRANOVO_ROOT/ContraNovo/ContraNovo.ckpt
CONTRANOVO_CONFIG=$CONTRANOVO_ROOT/ContraNovo/config.yaml
TRAIN_MGF_DIR=$SHARED_DIR/fusion_train_set
VAL_MGF_DIR=$SHARED_DIR/fusion_val_set

case "$MODE" in
  teacher)
    # MassiveKB is pre-split into MGF file(s) per directory; ContraNovo loads one
    # MGF at a time, so process each split separately (first MGF per split dir).
    for split in train val; do
      out_pt=$SHARED_DIR/contranovo_teacher_${split}_torch_data.pt
      if [ "$split" = "train" ]; then mgf_dir=$TRAIN_MGF_DIR; else mgf_dir=$VAL_MGF_DIR; fi
      mgf=$(ls "$mgf_dir"/*.mgf | head -n1)
      echo "=== Teacher-forcing ContraNovo on $mgf -> $out_pt ==="
      PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-} \
        "$CONTRANOVO_PYTHON" "$CONTRANOVO_ROOT/run_teacher.py" \
          --peak_path="$mgf" --model="$CONTRANOVO_CKPT" \
          --config="$CONTRANOVO_CONFIG" --out="$out_pt"
    done
    # const.CONTRANOVO_TEACHER_*_PT_PATH expects {train,test} filenames; val -> test.
    if [ -f "$SHARED_DIR/contranovo_teacher_val_torch_data.pt" ]; then
      mv -f "$SHARED_DIR/contranovo_teacher_val_torch_data.pt" \
            "$SHARED_DIR/contranovo_teacher_test_torch_data.pt"
    fi
    ;;

  fusion_data)
    "$PYTHON_BIN" -c "from dnps_hybrid.prepare_data import generate_contranovo_fusion_files; generate_contranovo_fusion_files()"
    ;;

  plm_teacher)
    # Consume the ContraNovo-vocab X tensors and write the matching score tensors.
    "$PYTHON_BIN" -c "
import torch
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
    # Retrain the ContraNovo fusion head with the asymbnln + swap50 recipe
    # (matches the Casanovo asymbnln methodology). Reuses the teacher artifacts
    # under casanovo/. Outputs:
    #   casanovo/contranovo_{fusion,null}_asymbnln_swap50.pth
    export DNPS_FUSION_BACKBONE=contranovo
    export DNPS_CONTRANOVO_FUSION_MODEL_PATH=$SHARED_DIR/contranovo_fusion_asymbnln_swap50.pth
    export DNPS_CONTRANOVO_NULL_MODEL_PATH=$SHARED_DIR/contranovo_null_asymbnln_swap50.pth
    export DNPS_FUSION_PLM_TOP2_SWAP_FRAC=0.5
    export DNPS_FUSION_PLM_RAND_SWAP_FRAC=0.0
    export DNPS_FUSION_EPOCHS=16
    unset DNPS_FUSION_OUTPUT_IL || true

    for f in \
      contranovo_teacher_scores_train.pt contranovo_teacher_scores_test.pt \
      contranovo_fusion_y_train.pt contranovo_fusion_y_test.pt \
      contranovo_plm_psm_teacher_scores_train.pt contranovo_plm_psm_teacher_scores_test.pt
    do
      if [ ! -s "$SHARED_DIR/$f" ]; then
        echo "[fatal] missing $SHARED_DIR/$f (run the teacher/fusion_data/plm_teacher modes first)" >&2
        exit 2
      fi
    done

    echo "=== Train ContraNovo+pepLM fusion (asymbnln, swap50, 16 epochs) ==="
    "$PYTHON_BIN" -m dnps_hybrid.train_fusion_head
    ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    exit 2 ;;
esac

echo "=== Done ==="
