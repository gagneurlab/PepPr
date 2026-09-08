# PepPr

Resolving spectral ambiguity in *de novo* peptide sequencing using peptide
priors.

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

## Inference

Download and extract the model archive from Zenodo (DOI filled in on
publication), then point at the checkpoints inside it:

```bash
tar xzf peppr_models.tar.gz
export PEPPR_MODELS="$PWD/peppr_models"

PEPPR_PRIOR_PATH="$PEPPR_MODELS/models/human/prior_model.pt" \
PEPPR_FUSION_PATH="$PEPPR_MODELS/models/human/fusion_model.pt" \
casanovo sequence \
    -m https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt \
    -c casanovo/casanovo/config.yaml \
    --teacher_forcing false --use_peppr true \
    -d output_dir -o results \
    -f spectra.mgf
```

The archive holds one prior per species under `models/<species>/prior_model.pt`;
swap `PEPPR_PRIOR_PATH` to use a different one, and the same fusion head still
applies. `--use_peppr false` gives the plain Casanovo baseline.

## Train

Priors and fusion heads are trained separately and don't need to be retrained
together. A fusion head fuses a prior's scores with one specific backbone's
score distribution (`PEPPR_BACKBONE`, e.g. Casanovo or ContraNovo), so it's
reusable across any prior trained for that backbone: swap species by pointing
`PEPPR_PRIOR_PATH` at a different prior and keep the same fusion head. Only
changing the backbone itself requires training a new fusion head, since that
changes the score distribution the fusion head was fit to.

Training writes its intermediates into the current directory. Set
`PEPPR_WORK_DIR` to write elsewhere.

### Train a new prior

Reads the proteome from `PEPPR_FASTA`. No data archive is involved.

```bash
export PEPPR_FASTA=/path/to/proteome.fasta

# Digest the proteome, then train.
python peppr/prepare_data.py
PEPPR_PRIOR_PATH=/path/to/prior.pt python peppr/train_peptide_prior_model.py
```

### Train a new fusion head

Needed only when switching `PEPPR_BACKBONE`; a fusion head trained for one
backbone works with every prior trained for that same backbone.

```bash
# Backbone teacher scores + prior teacher scores over one corpus.
# PEPPR_BACKBONE selects which backbone's teacher scores to compute
# ('casanovo', the default, or 'contranovo').
PEPPR_FUSION_PATH=/path/to/fusion.pth python peppr/train_fusion_head.py
```

The part worth relocating deliberately is `fusion/`, holding the backbone
teacher scores and fusion targets: those depend only on the training corpus,
not on the prior, so point `PEPPR_FUSION_WORK_DIR` at a shared location to
avoid recomputing them for every prior.

## Environment variables

| Variable            | Meaning                  | Default  |
| ------------------- | ------------------------ | -------- |
| `PEPPR_PRIOR_PATH`  | filepath of prior model  | required |
| `PEPPR_FUSION_PATH` | filepath of fusion model | required |

Inference needs only these two — which backbone runs is decided by which CLI
you invoke (`casanovo sequence`, ContraNovo's equivalent, ...), not by an env
var. `PEPPR_BACKBONE` is a training-only variable; see Train a new fusion
head above. Anything that genuinely requires a training directory fails with
a message naming the variable to set.
