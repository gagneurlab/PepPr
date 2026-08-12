#!/bin/bash
#SBATCH --job-name=9s_inference
#SBATCH --output=data/slurm/%A_%a_9s_inference.out
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# Casanovo + PepPr (pepLM) inference on the nine-species benchmark, with the
# asymbnln fusion head. One driver for the same-species and cross-species arms.
#
# Usage:
#   sbatch experiments/fig2/run_nine_species_inference.slurm <same|cross> [species ...]
#
#   same   -- each species evaluated against ITS OWN pepLM, using the
#             pepLM-independent MassiveKB asymbnln fusion head.
#             default species: human mouse honeybee tomato cowpea archaeon
#                              endoloripes yeast bacillus
#   cross  -- every species evaluated against the human_iso pepLM + its matching
#             human_iso asymbnln fusion head (human is excluded — it would
#             collapse to the same-species arm).
#             default species: mouse tomato cowpea archaeon endoloripes yeast bacillus
#
# Large species (yeast, bacillus) can be sharded across a Slurm array; the shard
# is taken from the array task id automatically:
#   sbatch --array=0-3 experiments/fig2/run_nine_species_inference.slurm same yeast
#
# Env overrides (all optional):
#   DNPS_FUSION_MODEL_PATH / DNPS_NULL_MODEL_PATH  -- fusion head (default per mode)
#   DNPS_PLM_SPECIES                               -- pepLM species (default per mode)
#   DNPS_SPECIES_LIST                              -- alternative to positional species
#   DNPS_NINE_SPECIES_PATH                         -- external ProForma tree root

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
# External nine-species ProForma tree; intentionally excluded from Zenodo.
export DNPS_NINE_SPECIES_PATH="${DNPS_NINE_SPECIES_PATH:-$DNPS_DATA_PATH/external/nine_species}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

export TMUX="${TMUX:-noninteractive}"
set -euo pipefail
cd "$REPO_ROOT"

MODE="${1:?usage: run_nine_species_inference.slurm <same|cross> [species ...]}"
shift || true

DATA_ROOT="$DNPS_DATA_PATH"
case "$MODE" in
  same)
    export DNPS_FUSION_MODEL_PATH="${DNPS_FUSION_MODEL_PATH:-$DATA_ROOT/models/casanovo/fusion_model_asymbnln.pth}"
    export DNPS_NULL_MODEL_PATH="${DNPS_NULL_MODEL_PATH:-$DATA_ROOT/models/casanovo/null_model_asymbnln.pth}"
    DEFAULT_SPECIES=(human mouse honeybee tomato cowpea archaeon endoloripes yeast bacillus)
    PLM_MODE=per_species
    ;;
  cross)
    export DNPS_FUSION_MODEL_PATH="${DNPS_FUSION_MODEL_PATH:-$DATA_ROOT/models/human_iso_asymbnln/fusion_model.pth}"
    export DNPS_NULL_MODEL_PATH="${DNPS_NULL_MODEL_PATH:-$DATA_ROOT/models/human_iso_asymbnln/null_model.pth}"
    export DNPS_PLM_SPECIES="${DNPS_PLM_SPECIES:-human_iso}"
    DEFAULT_SPECIES=(mouse tomato cowpea archaeon endoloripes yeast bacillus)
    PLM_MODE=fixed
    ;;
  *)
    echo "unknown mode '$MODE' (expected same | cross)" >&2; exit 2 ;;
esac

if [[ ! -s "$DNPS_FUSION_MODEL_PATH" ]]; then
  echo "[fatal] missing fusion head $DNPS_FUSION_MODEL_PATH" >&2
  exit 2
fi

# Species list: positional args > DNPS_SPECIES_LIST > mode default.
if [[ "$#" -gt 0 ]]; then
  SPECIES_LIST=( "$@" )
elif [[ -n "${DNPS_SPECIES_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  SPECIES_LIST=( ${DNPS_SPECIES_LIST} )
else
  SPECIES_LIST=( "${DEFAULT_SPECIES[@]}" )
fi

# Per-job isolated Lance cache (avoids commit conflicts between parallel jobs).
export DNPS_LANCE_DIR="${TMPDIR:-/tmp}/casanovo_lance_${SLURM_JOB_ID:-$$}"
# Optional array sharding of a single large species' MGF set.
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  export DNPS_MGF_SHARD="${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
fi

echo "=== Pre-flight ==="
echo "mode:      $MODE"
echo "fusion:    $DNPS_FUSION_MODEL_PATH ($(stat -c %y "$DNPS_FUSION_MODEL_PATH"))"
echo "null:      $DNPS_NULL_MODEL_PATH"
echo "pepLM:     ${PLM_MODE/per_species/per-species}${DNPS_PLM_SPECIES:+ ($DNPS_PLM_SPECIES)}"
echo "species:   ${SPECIES_LIST[*]}"
echo "lance_dir: $DNPS_LANCE_DIR  (per-job, isolated)"
[[ -n "${DNPS_MGF_SHARD:-}" ]] && echo "shard:     $DNPS_MGF_SHARD"
echo

for sp in "${SPECIES_LIST[@]}"; do
  if [[ "$MODE" == cross && "$sp" == human ]]; then
    echo "[human] skipped in cross mode (== same-species arm)"; continue
  fi
  if [[ "$PLM_MODE" == per_species ]]; then
    export DNPS_PLM_SPECIES="$sp"
  fi
  echo "================================================"
  echo "[$sp] DNPS_SPECIES=$sp  DNPS_PLM_SPECIES=$DNPS_PLM_SPECIES"
  echo "================================================"
  DNPS_SPECIES="$sp" \
    "$PYTHON_BIN" peptide_priors/inference.py auto \
      --dataset NINE_SPECIES_DATASET --use-plm true
  echo "[$sp] done"
  echo
done

echo "=== All species done ==="
