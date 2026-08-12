# Peptide priors

Resolving spectral ambiguity in *de novo* peptide sequencing using peptide priors.

## Installation

```bash
MY_NEW_ENV=          # your new environment name
conda create --name "$MY_NEW_ENV" -y python=3.11
conda activate "$MY_NEW_ENV"
pip install -r requirements.txt
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Fetch model submodules and apply their integration patches.
git submodule update --init
cd casanovo
git apply ../casanovo_integration.patch
pip install -e .
cd ..
cd ContraNovo
git apply ../contranovo_integration.patch
cd ..
```

### The Casanovo integration patch

`casanovo_integration.patch` adds the fusion hooks to the Casanovo v5 submodule:

- a `--use_plm` flag on `casanovo sequence` (`casanovo.py`),
- pepLM + fusion-head loading and fusion re-scoring inside beam search
(`denovo/model.py`),
- teacher-score dumping and PSM/mzTab plumbing used to build fusion training data
(`denovo/model_runner.py`, `data/psm.py`, `data/ms_io.py`).

The patch depends only on `dnps_hybrid` being importable (`from dnps_hybrid.model import load_plm_model, load_fusion_model`). To adapt the fusion head to a
**different DNPS backbone**, apply the analogous hooks to that model's beam search
and point `DNPS_FUSION_BACKBONE` at it (see below).

### The ContraNovo integration patch

`contranovo_integration.patch` adds pepLM fusion during ContraNovo beam search,
propagates the `--use_plm` option, and provides prediction and teacher-forcing
runner scripts used by the fusion pipeline.

## Configuration

Download and extract the companion Zenodo data archive anywhere outside the
Git checkout, then point the code at its root.

### Download the data archive

The archive ships as a single gzip tarball, `dnps_hybrid_zenodo.tar.gz`
(13.2 GB compressed; SHA-256
`589cd2638972978c69d3de04b04848646c36af4ad4ba36a9f24f1be7f7190091`), published on
Zenodo. Download it from the record's DOI page (`https://doi.org/<DOI>`, filled in
on publication) or on the command line:

```bash
curl -L -o dnps_hybrid_zenodo.tar.gz \
  "https://zenodo.org/records/21869640/files/dnps_hybrid_zenodo.tar.gz?download=1"
```

During peer review the Zenodo record is private; reviewers are given a temporary
preview link separately (it is not committed here).

### Extract and configure

```bash
tar xzf dnps_hybrid_zenodo.tar.gz
sha256sum dnps_hybrid_zenodo.tar.gz   # expect 589cd263…190091
export DNPS_DATA_PATH="$PWD/dnps_hybrid_zenodo"
```

> The nine-species MGF benchmark (~28 GB) is **not** bundled; obtain it separately
> and set `DNPS_NINE_SPECIES_PATH` (see the archive's `ARCHIVE_README.md`).

`DNPS_DATA_PATH` is required. The archive contains the final annotated inputs,
checkpoints, and result tables used by the maintained workflows; no
machine-specific filesystem layout is assumed. Repository paths (the Casanovo
and ContraNovo submodules, configs, and source code) are resolved from the
checkout itself.

The archive's `ARCHIVE_README.md`, `MANIFEST.tsv`, and `MANIFEST.json` record the
role, provenance, size, and SHA-256 checksum of every archived file.

Archive layout:

- `fastas/`: maintained proteome and antibody training FASTAs
- `training/massivekb/`: final annotated fusion train/validation MGFs
- `benchmarks/`: ProteomeTools SAAV, mAb, and fixed KoL benchmark MGFs
- `external/nine_species/`: Noble nine-species ProForma MGF benchmark
- `models/`: canonical pepLM, fusion, null, antibody, and baseline checkpoints
- `results/`: canonical mzTabs, logs, summaries, and external-baseline outputs
- `metadata/`: mAb references, regions, and final assembly summaries

The KoL payload is under 1 GiB: it contains one fixed 10,000-spectrum MGF for
each of the 15 maintained species. Each human- or mouse-prior cross-species
comparison uses 14 of these after excluding the matching species.

The Noble nine-species benchmark (MassIVE
[`MSV000090982`](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=MSV000090982))
is included loose under `external/nine_species/<Species>/*.mgf`. That path is
the default for `DNPS_NINE_SPECIES_PATH`.
The selected MGFs must have valid ProForma `SEQ=` annotations. If a downloaded
distribution uses leading numeric mass shifts (for example
`SEQ=+43.006PEPTIDE`), convert it before inference:

```bash
python dnps_hybrid/prepare_data.py convert_proforma \
    "/path/to/downloaded/species/*.mgf" \
    "/path/to/proforma/species"
```

Paths and experiment choices are selected by environment variables:


| Env var                       | Meaning                                                     | Default          |
| ----------------------------- | ----------------------------------------------------------- | ---------------- |
| `DNPS_SPECIES`                | species of the sample being evaluated                       | `human`          |
| `DNPS_PLM_SPECIES`            | which species' pepLM to use (cross-species if ≠ target)     | `human_iso` for human; otherwise target species |
| `DNPS_PLM_CKPT_PATH`          | pepLM checkpoint to load                                    | under `models/`  |
| `DNPS_FUSION_MODEL_PATH`      | trained fusion head                                         | under `models/`  |
| `DNPS_NULL_MODEL_PATH`        | null (backbone-only) baseline head                          | under `models/`  |
| `DNPS_FUSION_BACKBONE`        | `casanovo` (default) or `contranovo`                        | `casanovo`       |
| `DNPS_EXP_DIR`                | redirect intermediate fusion artifacts to an experiment dir | unset            |
| `DNPS_DATA_PATH`              | extracted Zenodo archive root                               | required         |
| `DNPS_NINE_SPECIES_PATH`      | nine-species ProForma MGF root                              | `$DNPS_DATA_PATH/external/nine_species` |
| `DNPS_STAGING_DIR`            | writable temporary directory for data preparation           | system temp dir  |
| `DNPS_LANCE_DIR`              | writable Casanovo Lance cache                               | archive `work/`  |
| `DNPS_CONTRANOVO_PYTHON`      | Python executable for the optional ContraNovo environment   | current Python   |
| `DNPS_THERMO_RAW_FILE_PARSER` | optional ThermoRawFileParser executable                     | unset            |


Per-species proteome FASTAs are listed in the `SPECIES` dict in `const.py`.

## 1. Train a peptide language model (pepLM)

Generate the pepLM training corpus (in-silico digest of the FASTA file →
`plm_seq_x.pt` / `plm_seq_y.pt`), then train:

```bash
# Build training data for the species' proteome.
DNPS_SPECIES=human python dnps_hybrid/prepare_data.py

# Train the pepLM (writes the checkpoint under models/; override with
# DNPS_PLM_CKPT_PATH). Hyperparameters live in const.py (PLM_* knobs).
DNPS_SPECIES=human python dnps_hybrid/train_peptide_prior_model.py
```

On SLURM these two steps are wrapped by `scripts/train.sh`:
`train.sh plm <species>` (proteome), `train.sh germline-plm <human|mouse>`
(germline sliding-window), or `train.sh antibody-plm <human|mouse>`.

To train a pepLM on something other than a UniProt proteome (e.g. an antibody
germline repertoire), build `antibody/antibody_{human,mouse}.fasta` with
`dnps_hybrid/build_antibody_db.py` (which produces the clean V-REGION + J·C
germline corpus), add it to `SPECIES` in `const.py`, and run the two commands
above.

## 2. Train a fusion head (Casanovo example)

The fusion head learns to combine a **frozen** Casanovo backbone with the
pepLM. It is trained from teacher scores—the per-residue logits each model
assigns to the ground-truth peptides of the archived MassIVE-KB corpus.

```bash
# (a) Generate fusion training tensors: Casanovo teacher logits + pepLM teacher
#     logits over the training corpus (MassIVE-KB by default). This runs the
#     patched Casanovo to dump teacher scores, then assembles the fusion inputs.
DNPS_SPECIES=human python dnps_hybrid/prepare_data.py

# (b) Train the fusion head + the null (backbone-only) baseline.
#     DNPS_FUSION_BACKBONE=casanovo (default) writes DNPS_FUSION_MODEL_PATH and
#     DNPS_NULL_MODEL_PATH.
DNPS_FUSION_BACKBONE=casanovo python dnps_hybrid/train_fusion_head.py
```

On SLURM: `scripts/train.sh fusion` (Casanovo head) or
`scripts/train.sh contranovo-fusion` (ContraNovo baseline head).

The fusion head is **pepLM-independent** by design: once trained it can be reused
with any pepLM of the same backbone/vocab — you do not retrain it per species.
Swap the pepLM by setting `DNPS_PLM_CKPT_PATH` (or `DNPS_PLM_SPECIES`) at inference.

## 3. Run inference (Casanovo + pepLM)

With the pepLM and fusion-head checkpoints in place, run the patched Casanovo with
`--use_plm`:

```bash
DNPS_PLM_CKPT_PATH="$DNPS_DATA_PATH/models/human_iso/plm_ckpt.pt" \
DNPS_FUSION_MODEL_PATH="$DNPS_DATA_PATH/models/human_iso_asymbnln/fusion_model.pth" \
casanovo sequence \
    -m https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt \
    -c casanovo/casanovo/config.yaml \
    --teacher_forcing false --use_plm true \
    -d output_dir -o results \
    -f spectra.mgf
```

`--use_plm false` reproduces the plain Casanovo baseline. The full evaluation
pipeline (both arms over all benchmark datasets) is orchestrated by
`dnps_hybrid/inference.py auto`; `scripts/run_nine_species_inference.slurm` is the
SLURM entry point for the nine-species benchmark (`same`/`cross` arms), and
`scripts/run_kol_eval.sh` for the Kingdoms-of-Life generalization sweep.

## 4. Reproduce the paper figures

The archive stores canonical mzTabs, final ProForma MGFs, baseline outputs, and
small metadata tables. Regenerate derived plotting summaries before composing
the figures:

```bash
python scripts/plot_scatter_precision.py
python scripts/plot_supp_fig_1.py
python scripts/plot_supp_fig_2.py
python scripts/plot_benchmark.py --species mouse
python scripts/plot_kol_overlap_vs_pp_gain.py

python scripts/plot_figure_2.py      # nine-species benchmark
python scripts/plot_figure_3.py      # cross-species, SAAV, KoL generalization
python scripts/plot_figure_4.py      # mAb assembly (V+C)
for script in scripts/plot_supp_fig_*.py; do python "$script"; done
```

Benchmark baselines (PowerNovo, ContraNovo, InstaNovo) are launched via
`scripts/run_baseline.sh <tool> <species>` and compared in `scripts/plot_benchmark.py`.

The spectral-angle summary uses Koina/Prosit when it is regenerated and
therefore requires network access to the Koina service. ThermoRawFileParser is
needed only to recreate final ProForma MGFs from vendor RAW files; those final
MGFs are already included in the archive.

## Files

All maintained data paths are defined in `dnps_hybrid/const.py` relative to
`DNPS_DATA_PATH`. The archive intentionally excludes vendor RAW files,
unannotated or pre-ProForma MGFs, mzML files, teacher-score tensors, Lance and
FragPipe workspaces, generated configs, plotting caches, and scheduler logs.
These are either regenerable from archived final inputs or unrelated to the
published workflows.

Run the portability and archive checks with:

```bash
python tools/check_absolute_paths.py
```