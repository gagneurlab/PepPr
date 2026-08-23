#!/bin/bash
#SBATCH --job-name=dnps_train
#SBATCH --output=data/slurm/%j_train.out
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# One driver for training the method's models. The model code lives in
# peppr/ (train_peptide_prior_model.py, train_fusion_head.py, prepare_data.py);
# this script sets the per-target environment and generates any missing training data.
#
# Usage:
#   sbatch experiments/train.sh plm <species>            # proteome pepLM (e.g. human, yeast, human_iso)
#   sbatch experiments/train.sh antibody-plm <human|mouse>   # germline antibody pepLM (sliding-window)
#   sbatch experiments/train.sh fusion                   # Casanovo + pepLM fusion head (asymbnln, swap50)
#   sbatch experiments/train.sh contranovo-fusion [mode] # ContraNovo fusion pipeline (default: train_fusion)
#
# The header requests a GPU + 128G (a superset covering data-gen and training);
# override at submit time for lighter targets if desired.

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

TARGET="${1:?usage: train.sh <plm|antibody-plm|fusion|contranovo-fusion> [args]}"
shift || true

activate_env() { if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi; }
require_species() {
  case "${1:-}" in human|mouse) ;; *) echo "need species: human | mouse" >&2; exit 2 ;; esac
}
gen_if_missing() {
  # Generate pepLM training data only if the X/Y tensors are absent.
  "$PYTHON_BIN" -c "
import os
from peppr import const
from peppr.prepare_data import generate_plm_training_data
os.makedirs(const.PLM_RUN_PATH, exist_ok=True)
print(f'FASTA_PATH={const.FASTA_PATH}')
print(f'PLM_SEQ_X_PATH={const.PLM_SEQ_X_PATH}')
if os.path.exists(const.PLM_SEQ_X_PATH) and os.path.exists(const.PLM_SEQ_Y_PATH):
    print('pepLM training data already present, skipping generation')
else:
    generate_plm_training_data()
"
}

case "$TARGET" in
  plm)
    SP="${1:?usage: train.sh plm <species>}"
    export DNPS_SPECIES="$SP" DNPS_PLM_SPECIES="$SP"
    activate_env
    gen_if_missing
    "$PYTHON_BIN" peppr/train_peptide_prior_model.py
    ;;

  antibody-plm)
    require_species "${1:-}"; SPECIES="antibody_$1"
    export DNPS_SPECIES="$SPECIES" DNPS_PLM_SPECIES="$SPECIES"
    export DNPS_PLM_PROTEASES=sliding
    export DNPS_PLM_MAX_PEP_LEN=25
    export DNPS_PLM_MAX_ITERS=100000
    export DNPS_PLM_INIT_FROM_CHECKPOINT=0
    export DNPS_PLM_RAND_SUFFIX_FULL_LEN=1
    mkdir -p "$DNPS_DATA_PATH/$SPECIES"
    activate_env
    gen_if_missing
    "$PYTHON_BIN" peppr/train_peptide_prior_model.py
    echo "Done. $SPECIES pepLM checkpoint: $DNPS_DATA_PATH/$SPECIES/plm_ckpt.pt"
    ;;

  fusion)
    # Casanovo + pepLM fusion head: asymmetric BN(cas)+LN(plm), swap50, 16 epochs.
    export DNPS_SPECIES=human DNPS_PLM_SPECIES=human
    export DNPS_FUSION_PLM_TOP2_SWAP_FRAC=0.50
    export DNPS_FUSION_EPOCHS="${DNPS_FUSION_EPOCHS:-16}"
    export DNPS_FUSION_MODEL_PATH="$DNPS_DATA_PATH/models/casanovo/fusion_model_asymbnln.pth"
    export DNPS_NULL_MODEL_PATH="$DNPS_DATA_PATH/models/casanovo/null_model_asymbnln.pth"
    # Teacher tensors (Casanovo + pepLM logits, fusion targets) must be generated
    # first via `python peppr/prepare_data.py`; they are not in the archive.
    # train_fusion_head.py reads them from their const-defined paths and errors if absent.
    activate_env
    "$PYTHON_BIN" peppr/train_fusion_head.py
    ls -lh "$DNPS_FUSION_MODEL_PATH" "$DNPS_NULL_MODEL_PATH"
    ;;

  contranovo-fusion)
    # ContraNovo + pepLM fusion pipeline (baseline arm). Modes:
    #   teacher | fusion_data | plm_teacher | train_fusion (default).
    # The teacher pass needs the ContraNovo conda env; override CONTRANOVO_PYTHON.
    MODE="${1:-train_fusion}"
    SHARED_DIR="$DNPS_DATA_PATH/casanovo"
    CONTRANOVO_ROOT="$REPO_ROOT/ContraNovo"
    CONTRANOVO_PYTHON="${CONTRANOVO_PYTHON:-python}"
    case "$MODE" in
      teacher)
        for split in train val; do
          out_pt="$SHARED_DIR/contranovo_teacher_${split}_torch_data.pt"
          if [ "$split" = train ]; then mgf_dir="$SHARED_DIR/fusion_train_set"; else mgf_dir="$SHARED_DIR/fusion_val_set"; fi
          mgf=$(ls "$mgf_dir"/*.mgf | head -n1)
          echo "=== Teacher-forcing ContraNovo on $mgf -> $out_pt ==="
          PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$CONTRANOVO_PYTHON" "$CONTRANOVO_ROOT/run_teacher.py" \
            --peak_path="$mgf" --model="$CONTRANOVO_ROOT/ContraNovo/ContraNovo.ckpt" \
            --config="$CONTRANOVO_ROOT/ContraNovo/config.yaml" --out="$out_pt"
        done
        [ -f "$SHARED_DIR/contranovo_teacher_val_torch_data.pt" ] && \
          mv -f "$SHARED_DIR/contranovo_teacher_val_torch_data.pt" "$SHARED_DIR/contranovo_teacher_test_torch_data.pt"
        ;;
      fusion_data)
        activate_env
        "$PYTHON_BIN" -c "from peppr.prepare_data import generate_contranovo_fusion_files; generate_contranovo_fusion_files()"
        ;;
      plm_teacher)
        activate_env
        "$PYTHON_BIN" -c "
import torch
from peppr import const
from peppr.model import load_plm_model
from tqdm import tqdm
torch.manual_seed(const.SEED); torch.cuda.manual_seed(const.SEED)
torch.set_float32_matmul_precision('high')
model = load_plm_model(); batch_size = 4096
def run(X_path, out_path):
    X = torch.load(X_path, map_location=const.DEVICE)
    scores = torch.zeros(X.shape[0], const.PLM_BLOCK_SIZE, len(const.VOCAB), device=const.DEVICE)
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), total=len(X)//batch_size):
            end = min(i+batch_size, len(X)); logits, *_ = model(X[i:end]); scores[i:end] = logits
    torch.save(scores, out_path); print('Saved', out_path)
run(const.CONTRANOVO_PLM_PSM_X_TRAIN_PATH, const.CONTRANOVO_PLM_PSM_TEACHER_SCORES_TRAIN_PATH)
run(const.CONTRANOVO_PLM_PSM_X_TEST_PATH,  const.CONTRANOVO_PLM_PSM_TEACHER_SCORES_TEST_PATH)
"
        ;;
      train_fusion)
        export DNPS_FUSION_BACKBONE=contranovo
        export DNPS_CONTRANOVO_FUSION_MODEL_PATH="$DNPS_DATA_PATH/models/casanovo/contranovo_fusion_model.pth"
        export DNPS_CONTRANOVO_NULL_MODEL_PATH="$DNPS_DATA_PATH/models/casanovo/contranovo_null_model.pth"
        export DNPS_FUSION_PLM_TOP2_SWAP_FRAC=0.5 DNPS_FUSION_PLM_RAND_SWAP_FRAC=0.0 DNPS_FUSION_EPOCHS=16
        activate_env
        "$PYTHON_BIN" -m peppr.train_fusion_head
        ls -lh "$DNPS_CONTRANOVO_FUSION_MODEL_PATH" "$DNPS_CONTRANOVO_NULL_MODEL_PATH"
        ;;
      *) echo "unknown contranovo-fusion mode '$MODE'" >&2; exit 2 ;;
    esac
    ;;

  *)
    echo "unknown target '$TARGET' (plm|antibody-plm|fusion|contranovo-fusion)" >&2
    exit 2 ;;
esac

echo "=== Done ($(date -Is)) ==="
