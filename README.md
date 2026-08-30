# PepPr — peptide priors

Resolving spectral ambiguity in *de novo* peptide sequencing using peptide
priors.

A peptide language model (pepLM) trained on a reference proteome is fused with
a frozen *de novo* sequencing backbone through a small fusion head, which acts
as a drop-in replacement for the backbone's output layer. Casanovo is the
worked example; the same recipe applies to other backbones.

The code reproducing the paper's benchmarks and figures lives separately in
[gagneurlab/PepPr_paper](https://github.com/gagneurlab/PepPr_paper). This
repository is the tool alone.

## Installation

```bash
MY_NEW_ENV=          # your new environment name
conda create --name "$MY_NEW_ENV" -y python=3.11
conda activate "$MY_NEW_ENV"
pip install -r requirements.txt
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
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

`casanovo_integration.patch` adds the integration hooks to Casanovo. It depends
only on `peppr` being importable
(`from peppr.model import load_plm_model, load_fusion_model`):

- a `--use_peppr` flag on `casanovo sequence` (`casanovo.py`),
- pepLM + fusion-head loading and fusion re-scoring inside beam search
  (`denovo/model.py`),
- teacher-score dumping and PSM/mzTab plumbing used to build fusion training
  data (`denovo/model_runner.py`, `data/psm.py`, `data/ms_io.py`).

The same recipe can be used for other backbones: add the pepLM/fusion hooks to
that model's beam search, have it import the two loaders, and point
`DNPS_FUSION_BACKBONE` at it. A ready-made ContraNovo integration,
`contranovo_integration.patch`, is included — apply it the same way
(`cd ContraNovo && git apply ../contranovo_integration.patch`).

## Configuration

Inference needs only the checkpoints. Point the three path variables at them and
`peppr` works with no data archive present:

```bash
export DNPS_PLM_CKPT_PATH=/path/to/plm_ckpt.pt
export DNPS_FUSION_MODEL_PATH=/path/to/fusion_model.pth
export DNPS_NULL_MODEL_PATH=/path/to/null_model.pth
```

`DNPS_DATA_PATH` is only required for training and data preparation, where it
selects the archive holding the FASTAs, training corpora and checkpoints. When
it is unset, archive-derived paths resolve to `None` and any operation that
genuinely needs them fails with a message naming the variable to set.

| Env var                       | Meaning                                                     | Default                            |
| ----------------------------- | ----------------------------------------------------------- | ---------------------------------- |
| `DNPS_PLM_CKPT_PATH`          | pepLM checkpoint to load                                    | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_FUSION_MODEL_PATH`      | trained fusion head                                         | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_NULL_MODEL_PATH`        | null (backbone-only) baseline head                          | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_DATA_PATH`              | data archive root                                           | required for training / data prep  |
| `DNPS_SPECIES`                | species of the sample being evaluated                       | `human`                            |
| `DNPS_PLM_SPECIES`            | which species' pepLM to use (cross-species if ≠ target)     | `human_iso` for human, else target |
| `DNPS_FUSION_BACKBONE`        | `casanovo` (default) or `contranovo`                        | `casanovo`                         |
| `DNPS_DATASETS_MODULE`        | module defining named datasets for `inference.py auto`      | `peppr.const`                      |
| `DNPS_EXP_DIR`                | redirect intermediate fusion artifacts to an experiment dir | unset                              |
| `DNPS_LANCE_DIR`              | writable Casanovo Lance cache                               | archive `work/`                    |
| `DNPS_CONTRANOVO_PYTHON`      | Python executable for the optional ContraNovo environment   | current Python                     |
| `DNPS_THERMO_RAW_FILE_PARSER` | optional ThermoRawFileParser executable                     | unset                              |

## 1. Train a peptide prior model

Generate the training corpus (in-silico digest of the FASTA file →
`plm_seq_x.pt` / `plm_seq_y.pt`), then train:

```bash
# Build training data for the species' proteome.
DNPS_SPECIES=human python peppr/prepare_data.py

# Train the pepLM (writes the checkpoint under models/; override with
# DNPS_PLM_CKPT_PATH). Hyperparameters live in const.py (PLM_* knobs).
DNPS_SPECIES=human python peppr/train_peptide_prior_model.py
```

To train a pepLM on something other than a UniProt proteome (e.g. an antibody
germline repertoire), build the corpus FASTA, add it to `SPECIES` in
`const.py`, and run the two commands above.

## 2. Train a fusion head (Casanovo example)

The fusion head learns to combine a **frozen** Casanovo backbone with the
pepLM. It is trained from teacher scores — the per-residue logits each model
assigns to the ground-truth peptides of a training corpus.

```bash
# (a) Generate fusion training tensors: Casanovo teacher logits + pepLM teacher
#     logits over the training corpus (MassIVE-KB by default). This runs the
#     patched Casanovo to dump teacher scores, then assembles the fusion inputs.
DNPS_SPECIES=human python peppr/prepare_data.py

# (b) Train the fusion head + the null (backbone-only) baseline.
#     DNPS_FUSION_BACKBONE=casanovo (default) writes DNPS_FUSION_MODEL_PATH and
#     DNPS_NULL_MODEL_PATH.
DNPS_FUSION_BACKBONE=casanovo python peppr/train_fusion_head.py
```

The fusion head is **prior-independent** by design: once trained it can be
reused with any prior of the same backbone. Swap the prior by setting
`DNPS_PLM_CKPT_PATH` (or `DNPS_PLM_SPECIES`) at inference.

## 3. Run inference (Casanovo + pepLM)

With the pepLM and fusion-head checkpoints in place, run the patched Casanovo
with `--use_peppr`:

```bash
DNPS_PLM_CKPT_PATH=/path/to/plm_ckpt.pt \
DNPS_FUSION_MODEL_PATH=/path/to/fusion_model.pth \
casanovo sequence \
    -m https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt \
    -c casanovo/casanovo/config.yaml \
    --teacher_forcing false --use_peppr true \
    -d output_dir -o results \
    -f spectra.mgf
```

`--use_peppr false` reproduces the plain Casanovo baseline.

`peppr/inference.py auto` orchestrates both arms over a set of named datasets.
The datasets themselves are supplied by whoever drives inference: point
`DNPS_DATASETS_MODULE` (or `--datasets-module`) at a module exposing
`DatasetPaths` attributes. The paper analyses register theirs in
`experiments.paths`.
