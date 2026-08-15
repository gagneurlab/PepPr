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

# Fetch the model submodules, then apply the Casanovo integration patch.
git submodule update --init
cd casanovo
git apply ../casanovo_integration.patch
# editable_mode=compat is required: setuptools' default (strict) editable install
# resolves casanovo's nested casanovo/ package as a namespace, breaking `import casanovo`.
pip install -e . --config-settings editable_mode=compat
cd ..
```

### Integration patches

The install step applies `casanovo_integration.patch`, which adds the integration
hooks to Casanovo:

- a `--use_plm` flag on `casanovo sequence` (`casanovo.py`),
- pepLM + fusion-head loading and fusion re-scoring inside beam search
(`denovo/model.py`),
- teacher-score dumping and PSM/mzTab plumbing used to build fusion training data
(`denovo/model_runner.py`, `data/psm.py`, `data/ms_io.py`).

Casanovo is the worked example here, but the same recipe can be used for other
DNPS backbones: add the pepLM/fusion hooks to that model's beam
search, have it `from peptide_priors.model import load_plm_model, load_fusion_model`,
and point `DNPS_FUSION_BACKBONE` at it. A ready-made ContraNovo integration,
`contranovo_integration.patch`, is included — apply it the same way
(`cd ContraNovo && git apply ../contranovo_integration.patch`) to use the
`contranovo` backbone.

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

### Extract and configure

```bash
tar xzf dnps_hybrid_zenodo.tar.gz
sha256sum dnps_hybrid_zenodo.tar.gz   # expect 589cd263…190091
export DNPS_DATA_PATH="$PWD/dnps_hybrid_zenodo"
```

Archive layout:

- `fastas/`: maintained proteome and antibody training FASTAs
- `training/massivekb/`: final annotated fusion train/validation MGFs
- `benchmarks/`: ProteomeTools SAAV, mAb, and fixed KoL benchmark MGFs
- `external/nine_species/`: Noble nine-species ProForma MGF benchmark
- `models/`: canonical pepLM, fusion, null, antibody, and baseline checkpoints
- `results/`: canonical mzTabs, logs, summaries, and external-baseline outputs
- `metadata/`: mAb references, regions, and final assembly summaries

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

## 1. Train a peptide prior model

Generate the training corpus (in-silico digest of the FASTA file →
`plm_seq_x.pt` / `plm_seq_y.pt`), then train:

```bash
# Build training data for the species' proteome.
DNPS_SPECIES=human python peptide_priors/prepare_data.py

# Train the pepLM (writes the checkpoint under models/; override with
# DNPS_PLM_CKPT_PATH). Hyperparameters live in const.py (PLM_* knobs).
DNPS_SPECIES=human python peptide_priors/train_peptide_prior_model.py
```

On SLURM these two steps are wrapped by `experiments/train.sh`:
`train.sh plm <species>` (proteome) or `train.sh antibody-plm <human|mouse>`
(antibody).

To train a pepLM on something other than a UniProt proteome (e.g. an antibody
germline repertoire), build the antibody germline corpus
(`fastas/antibody_{human,mouse}.fasta`) with `experiments/fig4/germline_corpus.py`,
add it to `SPECIES` in `const.py`, and run the two commands above.

## 2. Train a fusion head (Casanovo example)

The fusion head learns to combine a **frozen** Casanovo backbone with the
pepLM. It is trained from teacher scores—the per-residue logits each model
assigns to the ground-truth peptides of the archived MassIVE-KB corpus.

```bash
# (a) Generate fusion training tensors: Casanovo teacher logits + pepLM teacher
#     logits over the training corpus (MassIVE-KB by default). This runs the
#     patched Casanovo to dump teacher scores, then assembles the fusion inputs.
DNPS_SPECIES=human python peptide_priors/prepare_data.py

# (b) Train the fusion head + the null (backbone-only) baseline.
#     DNPS_FUSION_BACKBONE=casanovo (default) writes DNPS_FUSION_MODEL_PATH and
#     DNPS_NULL_MODEL_PATH.
DNPS_FUSION_BACKBONE=casanovo python peptide_priors/train_fusion_head.py
```

On SLURM: `experiments/train.sh fusion` (Casanovo head) or
`experiments/train.sh contranovo-fusion` (ContraNovo baseline head).

The fusion head is **prior-independent** by design: once trained it can be reused
with any prior of the same backbone.
Swap the prior by setting `DNPS_PLM_CKPT_PATH` (or `DNPS_PLM_SPECIES`) at inference.

## 3. Run inference (Casanovo + pepLM)

With the prior model and fusion-head checkpoints in place, run the patched Casanovo with
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
`peptide_priors/inference.py auto`; `experiments/fig2/run_nine_species_inference.slurm` is the
SLURM entry point for the nine-species benchmark (`same`/`cross` arms).

## 4. Reproduce the paper figures

The archive stores canonical mzTabs, final ProForma MGFs, baseline outputs, and
small metadata tables. Regenerate derived plotting summaries before composing
the figures:

```bash
python experiments/fig3/plot_scatter_precision.py
python experiments/supp/plot_supp_fig_1.py
python experiments/supp/plot_supp_fig_2.py
python experiments/fig3/plot_kol_overlap_vs_pp_gain.py

python experiments/fig2/plot_figure_2.py      # nine-species benchmark
python experiments/fig3/plot_figure_3.py      # cross-species, SAAV, KoL generalization
python experiments/fig4/plot_figure_4.py      # mAb assembly (V+C)
for script in experiments/supp/plot_supp_fig_*.py; do python "$script"; done
```

Benchmark baselines (PowerNovo, ContraNovo, InstaNovo) are launched via
`experiments/fig2/run_baseline.sh <tool> <species>` and compared in `experiments/fig2/plot_figure_2.py`.

Figure 4 Panel E reads the mAb assembly summaries the archive ships under
`metadata/mabs/assembly/`. To regenerate them from the Casanovo mzTabs, run
`experiments/fig4/run_mab.sh assemble` (i.e. `experiments/fig4/assembly.py`),
which ALPS-assembles each mAb's baseline and +PepPr arms and rewrites those
TSVs. It requires the third-party ALPS assembler (`ALPS.jar`; set
`DNPS_ALPS_JAR`) and `npysearch`.

The spectral-angle summary uses Koina/Prosit when it is regenerated and
therefore requires network access to the Koina service. ThermoRawFileParser is
needed only to recreate final ProForma MGFs from vendor RAW files; those final
MGFs are already included in the archive.
