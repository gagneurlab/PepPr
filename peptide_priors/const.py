import os
import torch
from pathlib import Path
from dataclasses import dataclass

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent.parent
PROJECT_ROOT = str(current_dir)

# Portable Zenodo archive contract. DNPS_DATA_PATH points at the extracted
# archive root; code and bundled configuration continue to resolve relative to
# PROJECT_ROOT.
_data_path = os.environ.get("DNPS_DATA_PATH")
if not _data_path:
    raise RuntimeError(
        "DNPS_DATA_PATH is required; set it to the extracted Zenodo archive root."
    )
DATA_PATH = os.path.abspath(os.path.expanduser(_data_path))
FASTAS_DIR = os.path.join(DATA_PATH, "fastas")
TRAINING_DIR = os.path.join(DATA_PATH, "training")
MASSIVEKB_TRAINING_DIR = os.path.join(TRAINING_DIR, "massivekb")
BENCHMARKS_DIR = os.path.join(DATA_PATH, "benchmarks")
# Nine-species ProForma MGFs are included loose under external/nine_species/
# in the extracted Zenodo release tarball.
NINE_SPECIES_PATH = os.path.abspath(os.path.expanduser(
    os.environ.get(
        "DNPS_NINE_SPECIES_PATH",
        os.path.join(DATA_PATH, "external", "nine_species"),
    )
))
# Compatibility alias retained for callers that treated this as a benchmark
# root before the external-data boundary was made explicit.
NINE_SPECIES_BENCHMARK_DIR = NINE_SPECIES_PATH
PROTEOMETOOLS_SAAV_DIR = os.path.join(BENCHMARKS_DIR, "proteometools_saav")
KOL_DATA_ROOT = os.path.join(BENCHMARKS_DIR, "kingdoms_of_life")
KOL_SUBSETS_DIR = os.path.join(KOL_DATA_ROOT, "subsets")
MABS_BENCHMARK_DIR = os.path.join(BENCHMARKS_DIR, "mabs")
MODELS_DIR = os.path.join(DATA_PATH, "models")
RESULTS_DIR = os.path.join(DATA_PATH, "results")
KOL_RESULTS_DIR = os.path.join(RESULTS_DIR, "kingdoms_of_life")
MABS_RESULTS_DIR = os.path.join(RESULTS_DIR, "mabs")
METADATA_DIR = os.path.join(DATA_PATH, "metadata")
MABS_METADATA_DIR = os.path.join(METADATA_DIR, "mabs")
MABS_REFERENCES_DIR = os.path.join(MABS_METADATA_DIR, "references")
MABS_ASSEMBLY_DIR = os.path.join(MABS_METADATA_DIR, "assembly")
WORK_DIR = os.path.join(DATA_PATH, "work")


def model_run_path(run_name: str) -> str:
    return os.path.join(MODELS_DIR, run_name)


def result_run_path(run_name: str) -> str:
    return os.path.join(RESULTS_DIR, run_name)


def work_run_path(run_name: str) -> str:
    return os.path.join(WORK_DIR, run_name)


def nine_species_benchmark_dir(species: str) -> str:
    return os.path.join(NINE_SPECIES_BENCHMARK_DIR, SPECIES[species]["benchmark_dir"])


# Compatibility names used by the maintained antibody workflows.
XA_NOVO_DIR = os.path.join(MABS_BENCHMARK_DIR, "xa_novo")
NONTRYP_DIR = os.path.join(MABS_BENCHMARK_DIR, "nontryp")
BESLIC_DIR = os.path.join(MABS_BENCHMARK_DIR, "beslic")

# Shared plotting palette. Keep method colors stable across all figures.
COLOR_ORANGE = "#ff6f30"
COLOR_NAVY = "#004562"
COLOR_LIGHT_BLUE = "#5891ad"
COLOR_GREEN = "#1f7a3a"
COLOR_BLUE = "#1f77b4"
COLOR_RED = "#d62728"
COLOR_PURPLE = "#7a4ca8"
COLOR_PLUM = "#9b59b6"
COLOR_PEACH = "#ffb38a"
COLOR_LIGHT_GRAY = "#cccccc"
COLOR_HISTOGRAM_GRAY = "#aaaaaa"
COLOR_TEXT_GRAY = "#444444"
COLOR_MID_GRAY = "#6a6a6a"
COLOR_DARK_GRAY = "#2c2c2c"

COLOR_CASANOVO = COLOR_ORANGE
COLOR_PP = COLOR_NAVY
# User-facing peptide-prior method name in figures (formerly "+PP").
LABEL_CASANOVO_PEPPR = "Casanovo + PepPr"
LABEL_CASANOVO_PEPPR_COMPACT = "Casanovo+PepPr"
LABEL_CONTRANOVO_PEPPR = "ContraNovo + PepPr"
LABEL_CONTRANOVO_PEPPR_COMPACT = "ContraNovo+PepPr"
COLOR_XANOVO = COLOR_LIGHT_BLUE
COLOR_DATABASE_SEARCH = COLOR_GREEN

FIGURE_4_PATH = os.path.join(PROJECT_ROOT, "figure_4.png")
FIGURE_4_EXAMPLE_MGF = os.path.join(
    MABS_BENCHMARK_DIR, "xa_novo", "PXD060500_36H6", "mgf",
    "36H6-pepsin-HCD-20240524.mgf"
)
FIGURE_4_ASSEMBLY_TSV_PATHS = (
    os.path.join(MABS_ASSEMBLY_DIR, "3arm_summary_solo_k7to11.tsv"),
    os.path.join(MABS_ASSEMBLY_DIR, "3arm_summary_beslic_k7to11.tsv"),
)
SUPP_FIGURE_5_PATH = os.path.join(PROJECT_ROOT, "supp_fig_5.png")
FIGURE_4_MAB_REFERENCE_PATHS = {
    "2B4": os.path.join(MABS_REFERENCES_DIR, "2B4_ref.fasta"),
    "36H6": os.path.join(MABS_REFERENCES_DIR, "36H6_ref.fasta"),
    "85F7": os.path.join(MABS_REFERENCES_DIR, "85F7_ref.fasta"),
    "S2P6": os.path.join(MABS_REFERENCES_DIR, "S2P6_ref.fasta"),
    "IgG1_Human_H": os.path.join(MABS_REFERENCES_DIR, "IgG1_Human_H_ref.fasta"),
    "IgG1_Human_L": os.path.join(MABS_REFERENCES_DIR, "IgG1_Human_L_ref.fasta"),
    "Herceptin": os.path.join(MABS_REFERENCES_DIR, "Herceptin_ref.fasta"),
    "anti-FLAG-M2": os.path.join(MABS_REFERENCES_DIR, "anti-FLAG-M2_ref.fasta"),
    "WIgG1_H": os.path.join(MABS_REFERENCES_DIR, "WIgG1_mouse_H_ref.fasta"),
    "WIgG1_L": os.path.join(MABS_REFERENCES_DIR, "WIgG1_mouse_L_ref.fasta"),
}


def figure_4_xanovo_results_dir(mab: str) -> str:
    return os.path.join(
        MABS_RESULTS_DIR, "xa_novo", f"PXD060500_{mab}", "casanovo_results"
    )


def figure_4_nontryp_results_dir(mab: str, protease: str) -> str:
    return os.path.join(
        MABS_RESULTS_DIR, "nontryp", mab, protease, "casanovo_results"
    )

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

ACTIVE_SPECIES = os.environ.get("DNPS_SPECIES", "human")
_species_cfg = SPECIES[ACTIVE_SPECIES]
THERMO_RAW_FILE_PARSER = os.environ.get("DNPS_THERMO_RAW_FILE_PARSER")
CONTRANOVO_PYTHON = os.environ.get("DNPS_CONTRANOVO_PYTHON")
SMSNET_ROOT = os.environ.get(
    "DNPS_SMSNET_ROOT", os.path.join(RESULTS_DIR, "baselines", "smsnet")
)
CASANOVO_CONFIG_YAML = os.path.join(current_dir, "casanovo", "casanovo", "config.yaml")
CASANOVO_DEFAULT_CHECKPOINT = "https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt"
CONTRANOVO_CONFIG_YAML = os.path.join(
    PROJECT_ROOT, "ContraNovo", "ContraNovo", "config.yaml"
)
KOL_STAGING_DIR = os.environ.get(
    "DNPS_STAGING_DIR", os.path.join(WORK_DIR, "kingdoms_of_life", "staging")
)
RUN_NAME = _species_cfg["run_name"]
RUN_PATH = work_run_path(RUN_NAME)
MODEL_RUN_PATH = model_run_path(RUN_NAME)
RESULT_RUN_PATH = result_run_path(RUN_NAME)
FASTA_PATH = os.path.join(FASTAS_DIR, _species_cfg["fasta"])
# Human PepPr always uses the isoform-inclusive pepLM; nine-species benchmark
# species "human" keeps the canonical proteome FASTA above.
HUMAN_PEPPR_PLM = "human_iso"
HUMAN_PEPPR_FUSION_RUN = "human_iso_asymbnln"
_default_plm_species = HUMAN_PEPPR_PLM if ACTIVE_SPECIES == "human" else ACTIVE_SPECIES
PLM_SPECIES = os.environ.get("DNPS_PLM_SPECIES", _default_plm_species)
_plm_species_cfg = SPECIES[PLM_SPECIES]
PLM_RUN_PATH = work_run_path(_plm_species_cfg["run_name"])
PLM_MODEL_RUN_PATH = model_run_path(_plm_species_cfg["run_name"])
SHARED_RUN_PATH = work_run_path("massivekb")
SHARED_MODEL_RUN_PATH = model_run_path("casanovo")
_EXP_DIR = os.environ.get("DNPS_EXP_DIR")
def _exp(default_path: str) -> str:
    if _EXP_DIR is None:
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

# De novo runs on the GT-search 1%-FDR spectra (mgf_gt), which are labelled with the
# ground-truth variant peptide and stored in ProForma form (Casanovo-tokenizable).
PROTEOMETOOLS_SAAV_MGF_GLOB = os.path.join(PROTEOMETOOLS_SAAV_DIR, "mgf_gt", "*.mgf")
PROTEOMETOOLS_SAAV_DATASET = DatasetPaths(
    name="ProteomeTools_SAAV",
    rawfile_glob=None,
    msms_glob=None,
    mgf_unprocessed_dir=None,
    final_mgf_glob=PROTEOMETOOLS_SAAV_MGF_GLOB,
    mztab_path_dnps=os.path.join(RESULT_RUN_PATH, "proteometools_saav_dnps.mztab"),
    mztab_path_fusion=os.path.join(RESULT_RUN_PATH, "proteometools_saav_hybrid.mztab"),
)
_plm_tag = f"_plm{PLM_SPECIES}" if PLM_SPECIES != ACTIVE_SPECIES else ""
NINE_SPECIES_PROFORMA_DIR = nine_species_benchmark_dir(ACTIVE_SPECIES)
NINE_SPECIES_DATASET = DatasetPaths(
    name=f"9S_{ACTIVE_SPECIES}",
    rawfile_glob=None,
    msms_glob=None,
    mgf_unprocessed_dir=None,
    final_mgf_glob=os.path.join(NINE_SPECIES_PROFORMA_DIR, "*.mgf"),
    mztab_path_dnps=os.path.join(RESULT_RUN_PATH, f"9s_{ACTIVE_SPECIES}{_plm_tag}_dnps.mztab"),
    mztab_path_fusion=os.path.join(RESULT_RUN_PATH, f"9s_{ACTIVE_SPECIES}{_plm_tag}_hybrid.mztab"),
)
# Raw KoL preparation is optional and not part of the portable archive. A local
# source tree can be supplied explicitly when regenerating the final MGFs.
_KOL_SOURCE_ROOT = os.environ.get("DNPS_KOL_SOURCE_ROOT")
KOL_SEARCH_RESULTS_ROOT = (
    os.path.join(_KOL_SOURCE_ROOT, "PXD014877", "search_results")
    if _KOL_SOURCE_ROOT else None
)
KOL_RAW_DIRS = tuple(
    os.path.join(_KOL_SOURCE_ROOT, accession, "raw")
    for accession in ("PXD014877", "PXD019483")
) if _KOL_SOURCE_ROOT else ()
KOL_SPECIES_DIRS = {
    'bos_tauros'                                  : 'Bos_tauros'                                  ,
    'canaerohabidis_elegans'                      : 'Canaerohabidis_elegans'                      ,
    'cricetulus_griseus'                          : 'Cricetulus_griseus'                          ,
    'danio_rerio'                                 : 'Danio rerio'                                 ,
    'didelphis_didelphinae'                       : 'Didelphis didelphinae'                       ,
    'dog'                                         : 'Canis_lupus'                                 ,
    'drosophila_melanogaster'                     : 'Drosophila_melanogaster'                     ,
    'gallus_gallus'                               : 'Gallus_gallus'                               ,
    'human'                                       : 'Homo_sapiens'                                ,
    'mouse'                                       : 'Mus_musculus'                                ,
    'oryctolagus_cuniculus'                       : 'Oryctolagus cuniculus'                       ,
    'oryzias_melastigma'                          : 'Oryzias_melastigma'                          ,
    'rattus_norvegicus'                           : 'Rattus Norvegicus'                           ,
    'sus_scrofa'                                  : 'Sus scrofa'                                  ,
    'tardigrade'                                  : 'Tardigrade'                                  ,
}
KOL_SPECIES_RAW_PATTERNS = {
    'bos_tauros'                                  : r'BtaurosiRT'                                  ,
    'canaerohabidis_elegans'                      : r'C-elegansiRT'                                ,
    'cricetulus_griseus'                          : r'HamsteriRT'                                  ,
    'danio_rerio'                                 : r'DaniorerioiRT'                               ,
    'didelphis_didelphinae'                       : r'DidelphisiRT'                                ,
    'dog'                                         : r'20181130.*DogiRT'                            ,
    'drosophila_melanogaster'                     : r'DrosophilaiRT'                               ,
    'gallus_gallus'                               : r'GallusiRT'                                   ,
    'human'                                       : r'HeLaiRT'                                     ,
    'mouse'                                       : r'MusmusculusiRT'                              ,
    'oryctolagus_cuniculus'                       : r'RabbitiRT'                                   ,
    'oryzias_melastigma'                          : r'MedakaiRT'                                   ,
    'rattus_norvegicus'                           : r'RatiRT'                                      ,
    'sus_scrofa'                                  : r'PigiRT'                                      ,
    'tardigrade'                                  : r'TardigradeiRT'                               ,
}
KOL_NAME_REPLACEMENTS = {
    "Maus":          "Musmusculus", # mouse
    "Cow":           "Btauros",     # bos_tauros
    "ZebraFisch":    "Daniorerio",  # danio_rerio
    "Opossum":       "Didelphis",   # didelphis_didelphinae
    "Chicken":       "Gallus",      # gallus_gallus
    "Baertierchen":  "Tardigrade",  # tardigrade
}

def kol_species_msms_path(species: str) -> str:
    if KOL_SEARCH_RESULTS_ROOT is None:
        raise RuntimeError(
            "KoL raw-data preparation requires DNPS_KOL_SOURCE_ROOT; the "
            "portable archive contains final ProForma MGFs only."
        )
    return os.path.join(KOL_SEARCH_RESULTS_ROOT, KOL_SPECIES_DIRS[species], "msms.txt")

def kol_species_output_root(species: str) -> str:
    return os.path.join(WORK_DIR, "kingdoms_of_life", species)


def kol_species_subset_path(species: str) -> str:
    return os.path.join(KOL_SUBSETS_DIR, f"{species}.mgf")

# --- Shared files (same for all species, live under casanovo/) ---
FUSION_TRAINING_TRAIN_SET = os.path.join(MASSIVEKB_TRAINING_DIR, "fusion_train_set")
FUSION_TRAINING_VAL_SET = os.path.join(MASSIVEKB_TRAINING_DIR, "fusion_val_set")

CASANOVO_TEACHER_TRAIN_MZTAB_PATH = _exp(os.path.join(SHARED_RUN_PATH, "casanovo_teacher_train.mztab"))
CASANOVO_TEACHER_TEST_MZTAB_PATH = _exp(os.path.join(SHARED_RUN_PATH, "casanovo_teacher_test.mztab"))
CASANOVO_TEACHER_SCORES_TRAIN_PATH = _exp(os.path.join(SHARED_RUN_PATH, "casanovo_teacher_scores_train.pt"))
CASANOVO_TEACHER_SCORES_TEST_PATH = _exp(os.path.join(SHARED_RUN_PATH, "casanovo_teacher_scores_test.pt"))
FUSION_Y_TRAIN_PATH = _exp(os.path.join(SHARED_RUN_PATH, "fusion_y_train.pt"))
FUSION_Y_TEST_PATH = _exp(os.path.join(SHARED_RUN_PATH, "fusion_y_test.pt"))

# --- ContraNovo teacher / fusion artifacts (separate from casanovo's because
# the score tensors live in a different vocab even though the dim happens to
# also be 29).
CONTRANOVO_TEACHER_TRAIN_PT_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_teacher_train_torch_data.pt")
CONTRANOVO_TEACHER_TEST_PT_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_teacher_test_torch_data.pt")
CONTRANOVO_TEACHER_SCORES_TRAIN_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_teacher_scores_train.pt")
CONTRANOVO_TEACHER_SCORES_TEST_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_teacher_scores_test.pt")
CONTRANOVO_FUSION_Y_TRAIN_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_fusion_y_train.pt")
CONTRANOVO_FUSION_Y_TEST_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_fusion_y_test.pt")
CONTRANOVO_PLM_PSM_X_TRAIN_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_plm_psm_x_train.pt")
CONTRANOVO_PLM_PSM_X_TEST_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_plm_psm_x_test.pt")
CONTRANOVO_PLM_PSM_TEACHER_SCORES_TRAIN_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_plm_psm_teacher_scores_train.pt")
CONTRANOVO_PLM_PSM_TEACHER_SCORES_TEST_PATH = os.path.join(SHARED_RUN_PATH, "contranovo_plm_psm_teacher_scores_test.pt")
CONTRANOVO_FUSION_MODEL_PATH = os.environ.get(
    "DNPS_CONTRANOVO_FUSION_MODEL_PATH",
    os.path.join(SHARED_MODEL_RUN_PATH, "contranovo_fusion_model.pth"),
)
CONTRANOVO_NULL_MODEL_PATH = os.environ.get(
    "DNPS_CONTRANOVO_NULL_MODEL_PATH",
    os.path.join(SHARED_MODEL_RUN_PATH, "contranovo_null_model.pth"),
)

# --- Per-species files (live under RUN_PATH, different for each species) ---
# DNPS_PLM_DATA_SUFFIX lets experiments write versioned training data (e.g.
# "_sw_v2") without overwriting the baseline files.
_plm_data_suffix = os.environ.get("DNPS_PLM_DATA_SUFFIX", "")
PLM_SEQ_X_PATH = os.path.join(PLM_RUN_PATH, f'plm_seq_x{_plm_data_suffix}.pt')
PLM_SEQ_Y_PATH = os.path.join(PLM_RUN_PATH, f'plm_seq_y{_plm_data_suffix}.pt')
PLM_SEQ_COUNTS_PATH = os.path.join(PLM_RUN_PATH, f'plm_seq_counts{_plm_data_suffix}.pkl')
PLM_SEQ_TEACHER_SCORES_PATH = os.path.join(PLM_RUN_PATH, f'plm_seq_teacher_scores.pt')
PLM_PSM_X_TRAIN_PATH = _exp(os.path.join(RUN_PATH, f'plm_psm_x_train.pt'))
PLM_PSM_X_TEST_PATH = _exp(os.path.join(RUN_PATH, f'plm_psm_x_test.pt'))
PLM_PSM_TEACHER_SCORES_TRAIN_PATH = _exp(os.path.join(RUN_PATH, f'plm_psm_teacher_scores_train.pt'))
PLM_PSM_TEACHER_SCORES_TEST_PATH = _exp(os.path.join(RUN_PATH, f'plm_psm_teacher_scores_test.pt'))
PLM_CHECKPOINT_PATH = os.environ.get(
    "DNPS_PLM_CKPT_PATH",
    os.path.join(PLM_MODEL_RUN_PATH, f'plm_ckpt.pt'),
)

# Fusion head is trained from Casanovo + pepLM teacher scores on the same PSM
# rows; PLM scores live under RUN_PATH (per ACTIVE_SPECIES).  Default the
# checkpoint next to null_model.pth under RUN_PATH so e.g. DNPS_SPECIES=mouse
# loads mus_musculus/fusion_model.pth instead of silently using the shared
# human checkpoint under casanovo/.  Human PepPr uses the asymmetric head
# trained against the human_iso pepLM (human_iso_asymbnln/).
_fusion_model_run = (
    HUMAN_PEPPR_FUSION_RUN if PLM_SPECIES == HUMAN_PEPPR_PLM else RUN_NAME
)
FUSION_MODEL_PATH = os.environ.get(
    "DNPS_FUSION_MODEL_PATH",
    os.path.join(model_run_path(_fusion_model_run), "fusion_model.pth"),
)
FUSION_SCORES_TEACHER_TEST_PATH = os.path.join(RUN_PATH, 'fusion_scores_teacher_test.pt')
NULL_MODEL_PATH = os.environ.get(
    "DNPS_NULL_MODEL_PATH",
    os.path.join(model_run_path(_fusion_model_run), "null_model.pth"),
)
NULL_SCORES_TEACHER_TEST_PATH = os.path.join(RUN_PATH, 'null_scores_teacher_test.pt')

PLM_INIT_FROM_CHECKPOINT = os.environ.get("DNPS_PLM_INIT_FROM_CHECKPOINT", "0").lower() in ("1", "true", "yes")
PLM_RAND_SUFFIX_FULL_LEN = os.environ.get("DNPS_PLM_RAND_SUFFIX_FULL_LEN", "0").lower() in ("1", "true", "yes")
PLM_EVAL_INTERVAL = 1000
PLM_LOG_INTERVAL = 50
PLM_EVAL_ITERS = 10
PLM_EVAL_ONLY = False # if True, script exits right after the first eval
PLM_N_LAYER = 12
PLM_N_HEAD = 12
PLM_N_EMBD = 768
PLM_LEARNING_RATE = 2e-4
PLM_MAX_ITERS = int(os.environ.get("DNPS_PLM_MAX_ITERS", 750_000))
PLM_WEIGHT_DECAY = 0.0
PLM_BETA1 = 0.9
PLM_BETA2 = 0.95
PLM_GRAD_CLIP = 1.0
PLM_WARMUP_ITERS = 50_000
RAND_LOSS_WEIGHT = 0.05
PLM_MIN_LR = PLM_LEARNING_RATE / 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 # float16 if CPU
WANDB_PROJECT = 'dnps'
SEED=420
PLM_BATCH_SIZE = 384
PLM_BLOCK_SIZE = 100
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