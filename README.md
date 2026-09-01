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
PEPPR_PRIOR_PATH=/path/to/plm_ckpt.pt \
PEPPR_FUSION_PATH=/path/to/fusion_model.pth \
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

Training writes its intermediates under `PEPPR_WORK_DIR` and reads the proteome
from `PEPPR_FASTA`. No data archive is involved — point them wherever you like.

```bash
export PEPPR_WORK_DIR=/path/to/work
export PEPPR_FASTA=/path/to/proteome.fasta

# Prior: digest the proteome, then train.
python peppr/prepare_data.py
PEPPR_PRIOR_PATH=/path/to/prior.pt python peppr/train_peptide_prior_model.py

# Fusion head: backbone teacher scores + prior teacher scores over one corpus.
PEPPR_FUSION_PATH=/path/to/fusion.pth python peppr/train_fusion_head.py
```

Everything lands under `PEPPR_WORK_DIR`. The one exception worth knowing about
is `fusion/`, holding the backbone teacher scores and fusion targets: those
depend only on the training corpus, not on the prior, so point
`PEPPR_FUSION_WORK_DIR` at a shared location to avoid recomputing them for
every prior.

The fusion head is prior-independent: once trained, swap priors at inference
with `PEPPR_PRIOR_PATH`.

## Environment variables

| Variable                 | Meaning                                                   | Default        |
| ------------------------ | --------------------------------------------------------- | -------------- |
| `PEPPR_PRIOR_PATH`       | prior (pepLM) checkpoint                                  | required       |
| `PEPPR_FUSION_PATH`      | fusion head                                               | required       |
| `PEPPR_BACKBONE`         | `casanovo` or `contranovo`                                | `casanovo`     |
| `PEPPR_WORK_DIR`         | root for training intermediates                           | training only  |
| `PEPPR_FASTA`            | proteome to digest for prior training                     | training only  |
| `PEPPR_FUSION_WORK_DIR`  | shared fusion-corpus directory                            | `$WORK/fusion` |
| `PEPPR_DATASETS_MODULE`  | module defining named datasets for `inference.py auto`    | `peppr.const`  |

Inference needs only the two checkpoint paths. Anything that genuinely requires
a training directory fails with a message naming the variable to set.
operation that needs them fails with a message naming the variable to set.
