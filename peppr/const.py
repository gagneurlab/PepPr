import os
import torch
from pathlib import Path
from dataclasses import dataclass

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent
PROJECT_ROOT = str(current_dir)

# Data-archive contract. PEPPR_DATA_PATH points at the extracted archive root;
# code and bundled configuration continue to resolve relative to PROJECT_ROOT.
#
# It is OPTIONAL. Training, data preparation and the paper-reproduction
# workflows all need it, but plain inference does not: point
# PEPPR_PRIOR_PATH / PEPPR_FUSION_PATH / PEPPR_NULL_PATH at
# checkpoints directly and `import peppr` works with no archive present.
# Paths derived from the archive are None when it is unconfigured, so an
# unset PEPPR_DATA_PATH surfaces where the path is actually used rather than
# at import time.
_data_path = os.environ.get("PEPPR_DATA_PATH")
DATA_PATH = os.path.abspath(os.path.expanduser(_data_path)) if _data_path else None


def require_data_path(what: str = "this operation") -> str:
    """Return DATA_PATH, or raise with an actionable message if it is unset."""
    if DATA_PATH is None:
        raise RuntimeError(
            f"PEPPR_DATA_PATH is required for {what}; set it to the extracted "
            "data archive root. Inference alone does not need it if "
            "PEPPR_PRIOR_PATH, PEPPR_FUSION_PATH and "
            "PEPPR_NULL_PATH are set directly."
        )
    return DATA_PATH


def _under_data(*parts: str) -> str | None:
    """Join under the archive root, or None when it is not configured."""
    return os.path.join(DATA_PATH, *parts) if DATA_PATH else None


def _under(base: str | None, *parts: str) -> str | None:
    """Join under an archive-derived base that may itself be None."""
    return os.path.join(base, *parts) if base else None


FASTAS_DIR = _under_data("fastas")
TRAINING_DIR = _under_data("training")
MASSIVEKB_TRAINING_DIR = _under_data("training", "massivekb")
MODELS_DIR = _under_data("models")
RESULTS_DIR = _under_data("results")
WORK_DIR = _under_data("work")


def model_run_path(run_name: str) -> str:
    return os.path.join(require_data_path("model paths"), "models", run_name)


def result_run_path(run_name: str) -> str:
    return os.path.join(require_data_path("result paths"), "results", run_name)


def work_run_path(run_name: str) -> str:
    return os.path.join(require_data_path("work paths"), "work", run_name)


SPECIES = {
    "human":       {"benchmark_dir": "H.-sapiens",              "fasta": "UP000005640_9606.fasta",        "run_name": "casanovo", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"},
    "mouse":       {"benchmark_dir": "Mus-musculus",            "fasta": "mus_musculus.fasta",               "run_name": "mus_musculus", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000000589/UP000000589_10090.fasta.gz"},
    "yeast":       {"benchmark_dir": "Saccharomyces-cerevisiae","fasta": "saccharomyces_cerevisiae.fasta",   "run_name": "saccharomyces_cerevisiae", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000002311/UP000002311_559292.fasta.gz"},
    "bacillus":    {"benchmark_dir": "Bacillus-subtilis",       "fasta": "bacillus_subtilis.fasta",          "run_name": "bacillus_subtilis", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000001570/UP000001570_224308.fasta.gz"},
    "honeybee":    {"benchmark_dir": "Apis-mellifera",          "fasta": "apis_mellifera.fasta",             "run_name": "apis_mellifera", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005203/UP000005203_7460.fasta.gz"},
    "tomato":      {"benchmark_dir": "Solanum-lycopersicum",    "fasta": "solanum_lycopersicum.fasta",       "run_name": "solanum_lycopersicum", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000004994/UP000004994_4081.fasta.gz"},
    "cowpea":      {"benchmark_dir": "Vigna-mungo",             "fasta": "vigna_mungo.fasta",                "run_name": "vigna_mungo", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP001374535/UP001374535_3915.fasta.gz"},
    "archaeon":    {"benchmark_dir": "Methanosarcina-mazei",    "fasta": "methanosarcina_mazei.fasta",       "run_name": "methanosarcina_mazei", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Archaea/UP000034578/UP000034578_2209.fasta.gz"},
    "endoloripes": {"benchmark_dir": "Candidatus-endoloripes",  "fasta": "candidatus_endoloripes.fasta",     "run_name": "candidatus_endoloripes", "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Bacteria/UP000094849/UP000094849_1818881.fasta.gz"},
    "human_iso":   {"benchmark_dir": "H.-sapiens",              "fasta": "human_iso.fasta",                  "run_name": "human_iso", "url": "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&includeIsoform=true&query=reviewed%3Atrue+AND+organism_id%3A9606"},
    "antibody_human": {"benchmark_dir": "H.-sapiens",           "fasta": "antibody_human.fasta",             "run_name": "antibody_human", "url": ""},
    "antibody_mouse": {"benchmark_dir": "Mus-musculus",         "fasta": "antibody_mouse.fasta",             "run_name": "antibody_mouse", "url": ""},
}

ACTIVE_SPECIES = os.environ.get("PEPPR_SPECIES", "human")
_species_cfg = SPECIES[ACTIVE_SPECIES]
THERMO_RAW_FILE_PARSER = os.environ.get("PEPPR_THERMO_RAW_FILE_PARSER")
CONTRANOVO_PYTHON = os.environ.get("PEPPR_CONTRANOVO_PYTHON")
CASANOVO_CONFIG_YAML = os.path.join(current_dir, "casanovo", "casanovo", "config.yaml")
CASANOVO_DEFAULT_CHECKPOINT = "https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt"
CONTRANOVO_CONFIG_YAML = os.path.join(
    PROJECT_ROOT, "ContraNovo", "ContraNovo", "config.yaml"
)
RUN_NAME = _species_cfg["run_name"]
RUN_PATH = _under_data("work", RUN_NAME)
MODEL_RUN_PATH = _under_data("models", RUN_NAME)
RESULT_RUN_PATH = _under_data("results", RUN_NAME)
FASTA_PATH = _under(FASTAS_DIR, _species_cfg["fasta"])
# Human PepPr always uses the isoform-inclusive pepLM; nine-species benchmark
# species "human" keeps the canonical proteome FASTA above.
HUMAN_PEPPR_PRIOR = "human_iso"
HUMAN_PEPPR_FUSION_RUN = "human_iso_asymbnln"
_default_prior_species = HUMAN_PEPPR_PRIOR if ACTIVE_SPECIES == "human" else ACTIVE_SPECIES
PRIOR_SPECIES = os.environ.get("PEPPR_PRIOR_SPECIES", _default_prior_species)
_prior_species_cfg = SPECIES[PRIOR_SPECIES]
PRIOR_RUN_PATH = _under_data("work", _prior_species_cfg["run_name"])
PRIOR_MODEL_RUN_PATH = _under_data("models", _prior_species_cfg["run_name"])
SHARED_RUN_PATH = _under_data("work", "massivekb")
SHARED_MODEL_RUN_PATH = _under_data("models", "casanovo")
_EXP_DIR = os.environ.get("PEPPR_EXP_DIR")
def _exp(default_path: str | None) -> str | None:
    if _EXP_DIR is None or default_path is None:
        return default_path
    return os.path.join(_EXP_DIR, os.path.basename(default_path))

@dataclass(frozen=True)
class DatasetPaths:
    rawfile_glob: str | None
    mgf_unprocessed_dir: str | None
    final_mgf_glob: str
    msms_glob: str | None
    name: str
    mztab_path_dnps: str
    mztab_path_fusion: str


# --- Shared files (same for all species, live under casanovo/) ---
FUSION_TRAINING_TRAIN_SET = _under(MASSIVEKB_TRAINING_DIR, "fusion_train_set")
FUSION_TRAINING_VAL_SET = _under(MASSIVEKB_TRAINING_DIR, "fusion_val_set")

CASANOVO_TEACHER_TRAIN_MZTAB_PATH = _exp(_under(SHARED_RUN_PATH, "casanovo_teacher_train.mztab"))
CASANOVO_TEACHER_TEST_MZTAB_PATH = _exp(_under(SHARED_RUN_PATH, "casanovo_teacher_test.mztab"))
CASANOVO_TEACHER_SCORES_TRAIN_PATH = _exp(_under(SHARED_RUN_PATH, "casanovo_teacher_scores_train.pt"))
CASANOVO_TEACHER_SCORES_TEST_PATH = _exp(_under(SHARED_RUN_PATH, "casanovo_teacher_scores_test.pt"))
FUSION_Y_TRAIN_PATH = _exp(_under(SHARED_RUN_PATH, "fusion_y_train.pt"))
FUSION_Y_TEST_PATH = _exp(_under(SHARED_RUN_PATH, "fusion_y_test.pt"))

# --- ContraNovo teacher / fusion artifacts (separate from casanovo's because
# the score tensors live in a different vocab even though the dim happens to
# also be 29).
CONTRANOVO_TEACHER_TRAIN_PT_PATH = _under(SHARED_RUN_PATH, "contranovo_teacher_train_torch_data.pt")
CONTRANOVO_TEACHER_TEST_PT_PATH = _under(SHARED_RUN_PATH, "contranovo_teacher_test_torch_data.pt")
CONTRANOVO_TEACHER_SCORES_TRAIN_PATH = _under(SHARED_RUN_PATH, "contranovo_teacher_scores_train.pt")
CONTRANOVO_TEACHER_SCORES_TEST_PATH = _under(SHARED_RUN_PATH, "contranovo_teacher_scores_test.pt")
CONTRANOVO_FUSION_Y_TRAIN_PATH = _under(SHARED_RUN_PATH, "contranovo_fusion_y_train.pt")
CONTRANOVO_FUSION_Y_TEST_PATH = _under(SHARED_RUN_PATH, "contranovo_fusion_y_test.pt")
CONTRANOVO_PRIOR_PSM_X_TRAIN_PATH = _under(SHARED_RUN_PATH, "contranovo_plm_psm_x_train.pt")
CONTRANOVO_PRIOR_PSM_X_TEST_PATH = _under(SHARED_RUN_PATH, "contranovo_plm_psm_x_test.pt")
CONTRANOVO_PRIOR_PSM_TEACHER_SCORES_TRAIN_PATH = _under(SHARED_RUN_PATH, "contranovo_plm_psm_teacher_scores_train.pt")
CONTRANOVO_PRIOR_PSM_TEACHER_SCORES_TEST_PATH = _under(SHARED_RUN_PATH, "contranovo_plm_psm_teacher_scores_test.pt")
CONTRANOVO_FUSION_MODEL_PATH = os.environ.get(
    "PEPPR_CONTRANOVO_FUSION_PATH"
) or _under(SHARED_MODEL_RUN_PATH, "contranovo_fusion_model.pth")
CONTRANOVO_NULL_MODEL_PATH = os.environ.get(
    "PEPPR_CONTRANOVO_NULL_PATH"
) or _under(SHARED_MODEL_RUN_PATH, "contranovo_null_model.pth")

# --- Per-species files (live under RUN_PATH, different for each species) ---
# PEPPR_PRIOR_DATA_SUFFIX lets experiments write versioned training data (e.g.
# "_sw_v2") without overwriting the baseline files.
_prior_data_suffix = os.environ.get("PEPPR_PRIOR_DATA_SUFFIX", "")
PRIOR_SEQ_X_PATH = _under(PRIOR_RUN_PATH, f'plm_seq_x{_prior_data_suffix}.pt')
PRIOR_SEQ_Y_PATH = _under(PRIOR_RUN_PATH, f'plm_seq_y{_prior_data_suffix}.pt')
PRIOR_SEQ_COUNTS_PATH = _under(PRIOR_RUN_PATH, f'plm_seq_counts{_prior_data_suffix}.pkl')
PRIOR_SEQ_TEACHER_SCORES_PATH = _under(PRIOR_RUN_PATH, f'plm_seq_teacher_scores.pt')
PRIOR_PSM_X_TRAIN_PATH = _exp(_under(RUN_PATH, f'plm_psm_x_train.pt'))
PRIOR_PSM_X_TEST_PATH = _exp(_under(RUN_PATH, f'plm_psm_x_test.pt'))
PRIOR_PSM_TEACHER_SCORES_TRAIN_PATH = _exp(_under(RUN_PATH, f'plm_psm_teacher_scores_train.pt'))
PRIOR_PSM_TEACHER_SCORES_TEST_PATH = _exp(_under(RUN_PATH, f'plm_psm_teacher_scores_test.pt'))
PRIOR_CHECKPOINT_PATH = os.environ.get("PEPPR_PRIOR_PATH") or _under(
    PRIOR_MODEL_RUN_PATH, "plm_ckpt.pt"
)

# Fusion head is trained from Casanovo + pepLM teacher scores on the same PSM
# rows; prior scores live under RUN_PATH (per ACTIVE_SPECIES).  Default the
# checkpoint next to null_model.pth under RUN_PATH so e.g. PEPPR_SPECIES=mouse
# loads mus_musculus/fusion_model.pth instead of silently using the shared
# human checkpoint under casanovo/.  Human PepPr uses the asymmetric head
# trained against the human_iso pepLM (human_iso_asymbnln/).
_fusion_model_run = (
    HUMAN_PEPPR_FUSION_RUN if PRIOR_SPECIES == HUMAN_PEPPR_PRIOR else RUN_NAME
)
FUSION_MODEL_PATH = os.environ.get("PEPPR_FUSION_PATH") or _under_data(
    "models", _fusion_model_run, "fusion_model.pth"
)
FUSION_SCORES_TEACHER_TEST_PATH = _under(RUN_PATH, 'fusion_scores_teacher_test.pt')
NULL_MODEL_PATH = os.environ.get("PEPPR_NULL_PATH") or _under_data(
    "models", _fusion_model_run, "null_model.pth"
)
NULL_SCORES_TEACHER_TEST_PATH = _under(RUN_PATH, 'null_scores_teacher_test.pt')

PRIOR_INIT_FROM_CHECKPOINT = os.environ.get("PEPPR_PRIOR_INIT_FROM_CHECKPOINT", "0").lower() in ("1", "true", "yes")
PRIOR_RAND_SUFFIX_FULL_LEN = os.environ.get("PEPPR_PRIOR_RAND_SUFFIX_FULL_LEN", "0").lower() in ("1", "true", "yes")
PRIOR_EVAL_INTERVAL = 1000
PRIOR_LOG_INTERVAL = 50
PRIOR_EVAL_ITERS = 10
PRIOR_EVAL_ONLY = False # if True, script exits right after the first eval
PRIOR_N_LAYER = 12
PRIOR_N_HEAD = 12
PRIOR_N_EMBD = 768
PRIOR_LEARNING_RATE = 2e-4
PRIOR_MAX_ITERS = int(os.environ.get("PEPPR_PRIOR_MAX_ITERS", 750_000))
PRIOR_WEIGHT_DECAY = 0.0
PRIOR_BETA1 = 0.9
PRIOR_BETA2 = 0.95
PRIOR_GRAD_CLIP = 1.0
PRIOR_WARMUP_ITERS = 50_000
RAND_LOSS_WEIGHT = 0.05
PRIOR_MIN_LR = PRIOR_LEARNING_RATE / 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 # float16 if CPU
WANDB_PROJECT = 'dnps'
SEED=420
PRIOR_BATCH_SIZE = 384
PRIOR_BLOCK_SIZE = 100
VOCAB = ['-', # padding
         '$', # stop
         '.', # N-terminus
         'H', 'Y', 'A', 'C', 'Q', 'M', 'P', 'F', 'W', 'E', 'S', 'T', 'D', 'V', 'R', 'N', 'K', 'L', 'G',
         '#']
START_TOKEN = len(VOCAB)
CASANOVO_TOKENIZER_INDEX = {
    "$": 1, "A": 2, "C": 3, "C[Carbamidomethyl]": 4, "D": 5, "E": 6,
    "F": 7, "G": 8, "H": 9, "K": 10, "L": 11, "M": 12, "M[Oxidation]": 13,
    "N": 14, "N[Deamidated]": 15, "P": 16, "Q": 17, "Q[Deamidated]": 18,
    "R": 19, "S": 20, "T": 21, "V": 22, "W": 23, "Y": 24,
    "[+25.980265]-": 25, "[Acetyl]-": 26, "[Ammonia-loss]-": 27,
    "[Carbamyl]-": 28,
}
CASANOVO_TRANSLATION = torch.zeros(max(CASANOVO_TOKENIZER_INDEX.values()) + 1, dtype=torch.int64, device=DEVICE)
for _cas_str, _cas_idx in CASANOVO_TOKENIZER_INDEX.items():
    _aa = _cas_str[0]
    if _aa == '[':
        _aa = '.'
    elif _aa == 'I':
        _aa = 'L'
    CASANOVO_TRANSLATION[_cas_idx] = VOCAB.index(_aa)

# ContraNovo's tokenizer derives indices from its config residues dict, so build
# the translation lazily once we know the residue list at model init time.
def build_contranovo_translation(residues):
    """Build a translation tensor mapping ContraNovo token indices to pepLM
    VOCAB indices.

    ContraNovo assigns ids 1..N to residues and N+1 to the stop token "$".
    Index 0 is the padding token.
    """
    amino_acids = list(residues.keys()) + ["$"]
    translation = torch.zeros(len(amino_acids) + 1, dtype=torch.int64, device=DEVICE)
    for i, aa in enumerate(amino_acids):
        token_idx = i + 1
        first = aa[0]
        if first in ("+", "-", "["):
            mapped = "."
        elif first == "I":
            mapped = "L"
        else:
            mapped = first
        translation[token_idx] = VOCAB.index(mapped)
    return translation