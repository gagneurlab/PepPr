# PepPr — peptide priors

Resolving spectral ambiguity in *de novo* peptide sequencing using peptide
priors.

A peptide language model (pepLM) trained on a reference proteome is fused with
a frozen *de novo* sequencing backbone through a small fusion head, which
replaces the backbone's output layer. Casanovo is the worked example.

Code for the paper's benchmarks and figures lives in
[gagneurlab/PepPr_paper](https://github.com/gagneurlab/PepPr_paper).

## Install

```bash
conda create --name peppr -y python=3.11
conda activate peppr
pip install -r requirements.txt
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Patch Casanovo with the PepPr hooks.
git submodule update --init
cd casanovo
git apply ../casanovo_integration.patch
pip install -e . --config-settings editable_mode=compat
cd ..
```

`editable_mode=compat` is required — the default strict editable install
resolves Casanovo's nested `casanovo/` package as a namespace and breaks
`import casanovo`.

A ContraNovo integration is included too: `cd ContraNovo && git apply
../contranovo_integration.patch`.

## Run

Point at your checkpoints and run the patched Casanovo with `--use_peppr`:

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

`--use_peppr false` gives the plain Casanovo baseline.

Inference needs only the checkpoint paths — no data archive.

## Train

Set `DNPS_DATA_PATH` to a data archive holding the FASTAs and training corpora
(required for training and data preparation only).

```bash
# pepLM: digest the proteome, then train.
DNPS_SPECIES=human python peppr/prepare_data.py
DNPS_SPECIES=human python peppr/train_peptide_prior_model.py

# Fusion head: teacher scores from the frozen backbone + the pepLM.
DNPS_FUSION_BACKBONE=casanovo python peppr/train_fusion_head.py
```

The fusion head is prior-independent: once trained, swap priors at inference
with `DNPS_PLM_CKPT_PATH`.

## Environment variables

| Variable                 | Meaning                                       | Default                            |
| ------------------------ | --------------------------------------------- | ---------------------------------- |
| `DNPS_PLM_CKPT_PATH`     | pepLM checkpoint                              | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_FUSION_MODEL_PATH` | fusion head                                   | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_NULL_MODEL_PATH`   | null (backbone-only) head                     | under `$DNPS_DATA_PATH/models/`    |
| `DNPS_DATA_PATH`         | data archive root                             | required for training / data prep  |
| `DNPS_SPECIES`           | species being evaluated                       | `human`                            |
| `DNPS_PLM_SPECIES`       | which species' pepLM to use                   | `human_iso` for human, else target |
| `DNPS_FUSION_BACKBONE`   | `casanovo` or `contranovo`                    | `casanovo`                         |
| `DNPS_DATASETS_MODULE`   | module defining datasets for `inference.py`   | `peppr.const`                      |

With `DNPS_DATA_PATH` unset, archive-derived paths resolve to `None` and any
operation that needs them fails with a message naming the variable to set.
