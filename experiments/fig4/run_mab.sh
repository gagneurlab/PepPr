#!/bin/bash
#SBATCH --job-name=run_mab
#SBATCH --output=data/slurm/%A_%a_run_mab.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=noninterruptive
#SBATCH --exclude=ouga08,ouga09,ouga05,ouga06,ouga26
#
# One driver for the antibody (mAb) benchmark pipeline.
#
#   run_mab.sh fragpipe <beslic <MAB> | nontryp>
#         FragPipe 23.0 headless -> 1% FDR psm.tsv. CPU only.
#         nontryp resolves RUNS[SLURM_ARRAY_TASK_ID]; submit as an array.
#
#   run_mab.sh annotate <beslic [MAB...] | nontryp [--idx N ...]>
#         Insert SEQ=<ProForma> into the spectra (python experiments/fig4/benchmark_prep.py).
#
#   run_mab.sh casanovo <beslic_H|beslic_L|herceptin|trastuzumab|nontryp> <noplm|human_iso|germline>
#         Casanovo v5 de novo, one dataset x one arm. NEEDS A GPU:
#           sbatch --gres=gpu:1 experiments/fig4/run_mab.sh casanovo beslic_H noplm
#         nontryp resolves RUNS[SLURM_ARRAY_TASK_ID]; submit as an array.
#
#   run_mab.sh submit [idx ...] [--skip-fragpipe]
#         Orchestrate the full non-tryptic array pipeline on Slurm
#         (fragpipe -> annotate -> casanovo) with dependencies. Run on a login node.
#
#   run_mab.sh assemble [--mabs MAB...] [--k K...]
#         ALPS-assemble each mAb (baseline vs +PP) from its Casanovo mzTabs and
#         write the Figure 4 Panel E summaries (const.FIGURE_4_ASSEMBLY_TSV_PATHS).
#         Needs ALPS.jar (set DNPS_ALPS_JAR) + npysearch.
#
# The #SBATCH header omits --gres so the FragPipe/annotate stages don't tie up a
# GPU; the casanovo stage must be submitted with --gres=gpu:1 (submit does this).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
: "${DNPS_DATA_PATH:?Set DNPS_DATA_PATH to the workflow data root}"
export DNPS_DATA_PATH
PYTHON_BIN="${PYTHON_BIN:-python}"
CASANOVO_BIN="${CASANOVO_BIN:-casanovo}"
if [[ -n "${DNPS_ENV_SETUP:-}" ]]; then
  source "$DNPS_ENV_SETUP"
fi

set -euo pipefail
cd "$REPO_ROOT"

SUBCMD="${1:?usage: run_mab.sh <fragpipe|annotate|casanovo|submit|assemble> ...}"
shift || true


# ══════════════════════════════════════════════════════════════════════════
fragpipe_stage() {
  local mode="${1:?usage: run_mab.sh fragpipe <beslic <MAB> | nontryp>}"
  local workdir label
  case "$mode" in
    beslic)
      local mab="${2:?mab id required (IgG1_Human_H | IgG1_Human_L)}"
      workdir="$DNPS_DATA_PATH/beslic_mab/$mab/fragpipe_workdir"; label="$mab" ;;
    nontryp)
      local idx="${SLURM_ARRAY_TASK_ID:?nontryp needs an --array task id (RUNS index)}"
      # Stagger array startup so concurrent FragPipe instances don't race on the
      # shared ~/.config/FragPipe workflows cache during GUI-tab init.
      echo "Staggering startup by $((idx * 30))s (anti-race for FragPipe config dir)"
      sleep $((idx * 30))
      local mab protease
      read mab protease workdir < <("$PYTHON_BIN" - <<PYEOF
from experiments.fig4.benchmark_registry import RUNS
r = RUNS[$idx]
print(r.mab_id, r.protease, r.workdir)
PYEOF
)
      label="${mab}_${protease}" ;;
    *) echo "unknown fragpipe mode '$mode'" >&2; exit 2 ;;
  esac

  # FragPipe 23.0 needs Java 11 on PATH; point FRAGPIPE_JAVA_BIN at its bin dir
  # (or have java11 already active). FRAGPIPE overrides the executable location.
  local fragpipe="${FRAGPIPE:-$HOME/local/fragpipe-23.0/bin/fragpipe}"
  [[ -n "${FRAGPIPE_JAVA_BIN:-}" ]] && export PATH="$FRAGPIPE_JAVA_BIN:$PATH"

  echo "=== FragPipe: $label ==="
  echo "workdir:  $workdir"
  echo "manifest: $(cat "$workdir/fragpipe-files.fp-manifest")"
  echo "java:     $(java -version 2>&1 | head -1)"

  cd "$workdir"
  "$fragpipe" --headless \
    --workflow "$workdir/fragpipe.workflow" \
    --manifest "$workdir/fragpipe-files.fp-manifest" \
    --workdir "$workdir" \
    --ram 24 --threads 8 \
    2>&1 | tee "$workdir/fragpipe_run_${SLURM_JOB_ID:-$$}.log"

  local psm
  psm=$(find "$workdir" -maxdepth 2 -name psm.tsv | head -1)
  [[ -n "$psm" ]] || { echo "No psm.tsv found!" >&2; exit 1; }
  echo "psm.tsv: $psm"; wc -l "$psm"
}


# ══════════════════════════════════════════════════════════════════════════
casanovo_stage() {
  local dataset="${1:?usage: run_mab.sh casanovo <dataset> <arm>}"
  local arm="${2:?arm required: noplm | human_iso | germline}"
  local data=$DNPS_DATA_PATH
  local mab_id sample dir src_mgf res_dir tag norm_cys=""

  case "$dataset" in
    beslic_H|beslic_L)
      if [[ "$dataset" == beslic_H ]]; then mab_id=IgG1_Human_H; sample=Heavy-Chain-Trypsin-1
      else                                  mab_id=IgG1_Human_L; sample=Light-Chain-Trypsin-1; fi
      dir="$data/beslic_mab/$mab_id"
      src_mgf="$dir/${sample}_annotated.mgf"; res_dir="$dir/casanovo_results"
      tag="$mab_id"; norm_cys="+57.021" ;;
    herceptin)
      mab_id=Herceptin; dir="$data/PXD023419"
      src_mgf="$dir/Peng2021_Herceptin_tryp_annotated.mgf"; res_dir="$dir/casanovo_results"
      tag="herceptin"; norm_cys="+58.005" ;;   # Peng 2021 iodoacetic acid (Carboxymethyl, +58)
    trastuzumab)
      mab_id=Trastuzumab; dir="$data/PXD057525"
      src_mgf="$dir/PXD057525_trastuzumab_annotated.mgf"; res_dir="$dir/casanovo_results"
      tag="pxd057525"; norm_cys="" ;;           # chloroacetamide -> Carbamidomethyl, native vocab
    nontryp)
      local idx="${SLURM_ARRAY_TASK_ID:?nontryp needs an --array task id (RUNS index)}"
      local protease
      read mab_id protease src_mgf res_dir < <("$PYTHON_BIN" - <<PYEOF
from experiments.fig4.benchmark_registry import RUNS
r = RUNS[$idx]
print(r.mab_id, r.protease, r.annotated_mgf, r.res_dir)
PYEOF
)
      tag="${mab_id}_${protease}"; norm_cys="" ;;
    *) echo "unknown dataset '$dataset'" >&2; exit 2 ;;
  esac

  [[ -f "$src_mgf" ]] || { echo "[fatal] missing source MGF $src_mgf" >&2; exit 1; }
  mkdir -p "$res_dir"

  # Normalise ground-truth mods to Casanovo's vocab when the Cys mass differs.
  local eval_mgf
  if [[ -n "$norm_cys" ]]; then
    eval_mgf="$res_dir/$(basename "${src_mgf%.mgf}")_v5norm.mgf"
    NORM_CYS="$norm_cys" SRC_MGF="$src_mgf" EVAL_MGF="$eval_mgf" "$PYTHON_BIN" - <<'PYEOF'
import os, re
src, out, cys = os.environ["SRC_MGF"], os.environ["EVAL_MGF"], os.environ["NORM_CYS"]
with open(src) as f, open(out, "w") as o:
    for line in f:
        if line.startswith("SEQ="):
            s = line[4:].rstrip()
            s = re.sub(r'^Q\[\-17\.027\]', '[Ammonia-loss]-Q', s)   # N-term pyro-Glu
            s = s.replace(f'C[{cys}]', 'C[Carbamidomethyl]')
            s = s.replace('M[+15.995]', 'M[Oxidation]')
            s = s.replace('N[+0.984]', 'N[Deamidated]')
            s = s.replace('Q[+0.984]', 'Q[Deamidated]')
            s = s.replace('W[+15.995]', 'W')                        # Casanovo can't predict W-ox
            o.write(f"SEQ={s}\n")
        else:
            o.write(line)
PYEOF
  else
    eval_mgf="$src_mgf"
  fi

  local ckpt="${CASANOVO_CKPT:-https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt}"
  local job_tag="${tag}_${SLURM_JOB_ID:-$$}"
  local config="$res_dir/casanovo_config_${arm}_${job_tag}.yaml"
  sed "s|^lance_dir:.*|lance_dir: $res_dir/lance_${arm}_${job_tag}|" \
      "$REPO_ROOT/casanovo/casanovo/config.yaml" > "$config"

  local use_peppr=false out_tag
  case "$arm" in
    noplm)
      out_tag="casanovo_${tag}_noplm_${job_tag}" ;;
    human_iso)
      use_peppr=true
      export DNPS_PLM_CKPT_PATH="$data/models/human_iso/plm_ckpt.pt"
      export DNPS_FUSION_MODEL_PATH="$data/models/casanovo/fusion_model.pth"
      export DNPS_NULL_MODEL_PATH="$data/models/casanovo/null_model.pth"
      out_tag="casanovo_${tag}_humaniso_plm_${job_tag}" ;;
    germline)
      use_peppr=true
      local sp
      case "$mab_id" in
        IgG1_Human*|Herceptin|Trastuzumab) sp=human ;;
        *)                                 sp=mouse ;;
      esac
      # germline pepLM checkpoints were renamed to models/antibody_<sp>/ for the archive.
      export DNPS_PLM_CKPT_PATH="$data/models/antibody_${sp}/plm_ckpt.pt"
      export DNPS_FUSION_MODEL_PATH="$data/models/casanovo/fusion_model_asymbnln.pth"
      export DNPS_NULL_MODEL_PATH="$data/models/casanovo/null_model_asymbnln.pth"
      out_tag="casanovo_${tag}_germline_${sp}_sw_clean_${job_tag}" ;;
    *) echo "unknown arm '$arm'" >&2; exit 2 ;;
  esac

  echo "=== Casanovo: $dataset / $arm (use_peppr=$use_peppr) ==="
  echo "eval MGF: $eval_mgf ($(grep -c '^BEGIN IONS' "$eval_mgf") spectra)"
  [[ "$use_peppr" == true ]] && echo "pepLM:    $DNPS_PLM_CKPT_PATH"

  if declare -F conda >/dev/null; then conda activate "${DNPS_CONDA_ENV:-khsam}"; fi
  cd "$res_dir"
  local base; base=$(basename "$eval_mgf" .mgf)
  rm -rf "${base}__lance.lance" "${base}__lance" "lance_${arm}_${job_tag}" 2>/dev/null || true

  "$CASANOVO_BIN" sequence \
    -m "$ckpt" -c "$config" -d "$res_dir" -o "$out_tag" \
    --teacher_forcing false --use_peppr "$use_peppr" -f \
    "$eval_mgf" 2>&1 | tee "$res_dir/${out_tag}.log"
  ls -lh "$res_dir/${out_tag}".mztab 2>/dev/null
}


# ══════════════════════════════════════════════════════════════════════════
submit_pipeline() {
  # Orchestrate the non-tryptic array pipeline. Caps concurrency at 5.
  mkdir -p data/slurm
  local skip_fragpipe=0 idx_list=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-fragpipe) skip_fragpipe=1; shift ;;
      --) shift; idx_list="$*"; break ;;
      *) idx_list="$idx_list $1"; shift ;;
    esac
  done

  local array_spec
  if [[ -z "${idx_list// }" ]]; then
    array_spec="0-25%5"
  else
    array_spec="$(echo "$idx_list" | tr -s ' ' | sed 's/^ //; s/ /,/g')%5"
  fi
  echo "Using --array=$array_spec"

  local self="experiments/fig4/run_mab.sh" dep=""
  if [[ "$skip_fragpipe" -eq 0 ]]; then
    local fp_jid build_jid
    fp_jid=$(sbatch --parsable --array="$array_spec" \
               --time=3:00:00 --mem=32G "$self" fragpipe nontryp)
    echo "FragPipe array job: $fp_jid"
    build_jid=$(sbatch --parsable --dependency=afterok:"$fp_jid" \
                  --time=0:30:00 --mem=8G "$self" annotate nontryp)
    echo "Annotate job: $build_jid (after $fp_jid)"
    dep="--dependency=afterok:$build_jid"
  fi
  local cas_jid
  cas_jid=$(sbatch --parsable $dep --array="$array_spec" \
              --gres=gpu:1 "$self" casanovo nontryp noplm)
  echo "Casanovo array job: $cas_jid${dep:+ ($dep)}"
  echo
  echo "Monitor: squeue -u $USER -o '%.12i %.30j %.2t %.10M %R'"
}


# ══════════════════════════════════════════════════════════════════════════
case "$SUBCMD" in
  fragpipe) fragpipe_stage "$@" ;;
  casanovo) casanovo_stage "$@" ;;
  annotate) "$PYTHON_BIN" experiments/fig4/benchmark_prep.py annotate "$@" ;;
  submit)   submit_pipeline "$@" ;;
  assemble) "$PYTHON_BIN" experiments/fig4/assembly.py "$@" ;;
  *) echo "unknown subcommand '$SUBCMD' (fragpipe|annotate|casanovo|submit|assemble)" >&2; exit 2 ;;
esac

echo "=== Done ($(date -Is)) ==="
