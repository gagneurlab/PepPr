"""Paths, palette and datasets for the paper-reproduction figures.

Everything defined here is specific to reproducing the paper: where the
benchmarks, results and metadata live inside the data archive, the shared
figure palette, and the per-figure output paths. The installable ``peppr``
package deliberately knows nothing about any of it -- this module is the paper
side of that boundary.

Tool-level constants that the figure scripts also need (``VOCAB``, ``SPECIES``,
``CASANOVO_CONFIG_YAML``, ...) are re-exported here so a figure module has a
single import surface:

    from experiments.paths import COLOR_PP, SPECIES, result_run_path
"""

import os
from pathlib import Path

from peppr.const import (  # noqa: F401  (re-exported for figure modules)
    ACTIVE_SPECIES,
    CASANOVO_CONFIG_YAML,
    DATA_PATH,
    DatasetPaths,
    FASTAS_DIR,
    MODELS_DIR,
    PLM_SPECIES,
    SPECIES,
    VOCAB,
    WORK_DIR,
    require_data_path,
    result_run_path,
)

# Root of the paper checkout; rendered figures are written here.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _under_data(*parts: str) -> str | None:
    """Join under the archive root, or None when it is not configured."""
    return os.path.join(DATA_PATH, *parts) if DATA_PATH else None


def _under(base: str | None, *parts: str) -> str | None:
    """Join under an archive-derived base that may itself be None."""
    return os.path.join(base, *parts) if base else None


# ── Archive layout used by the paper analyses ─────────────────────────────
BENCHMARKS_DIR = _under_data("benchmarks")
RESULTS_DIR = _under_data("results")
METADATA_DIR = _under_data("metadata")

PROTEOMETOOLS_SAAV_DIR = _under_data("benchmarks", "proteometools_saav")

KOL_DATA_ROOT = _under_data("benchmarks", "kingdoms_of_life")
KOL_SUBSETS_DIR = _under_data("benchmarks", "kingdoms_of_life", "subsets")
KOL_RESULTS_DIR = _under_data("results", "kingdoms_of_life")

MABS_BENCHMARK_DIR = _under_data("benchmarks", "mabs")
MABS_RESULTS_DIR = _under_data("results", "mabs")
MABS_METADATA_DIR = _under_data("metadata", "mabs")
MABS_REFERENCES_DIR = _under_data("metadata", "mabs", "references")
MABS_ASSEMBLY_DIR = _under_data("metadata", "mabs", "assembly")

# Compatibility names used by the maintained antibody workflows.
XA_NOVO_DIR = _under(MABS_BENCHMARK_DIR, "xa_novo")
NONTRYP_DIR = _under(MABS_BENCHMARK_DIR, "nontryp")
BESLIC_DIR = _under(MABS_BENCHMARK_DIR, "beslic")

SMSNET_ROOT = os.environ.get("DNPS_SMSNET_ROOT") or _under(
    RESULTS_DIR, "baselines", "smsnet"
)

# Nine-species ProForma MGFs ship loose under external/nine_species/.
_nine_species_env = os.environ.get("DNPS_NINE_SPECIES_PATH")
NINE_SPECIES_PATH = (
    os.path.abspath(os.path.expanduser(_nine_species_env))
    if _nine_species_env
    else _under_data("external", "nine_species")
)
NINE_SPECIES_BENCHMARK_DIR = NINE_SPECIES_PATH


def nine_species_benchmark_dir(species: str) -> str:
    if NINE_SPECIES_BENCHMARK_DIR is None:
        require_data_path("the nine-species benchmark")
    return os.path.join(NINE_SPECIES_BENCHMARK_DIR, SPECIES[species]["benchmark_dir"])


# ── Shared plotting palette. Keep method colors stable across all figures. ─
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
COLOR_XANOVO = COLOR_LIGHT_BLUE
COLOR_DATABASE_SEARCH = COLOR_GREEN

# User-facing peptide-prior method name in figures (formerly "+PP").
LABEL_CASANOVO_PEPPR = "Casanovo + PepPr"
LABEL_CASANOVO_PEPPR_COMPACT = "Casanovo+PepPr"
LABEL_CONTRANOVO_PEPPR = "ContraNovo + PepPr"
LABEL_CONTRANOVO_PEPPR_COMPACT = "ContraNovo+PepPr"

# ── Figure outputs and their inputs ───────────────────────────────────────
FIGURE_4_PATH = os.path.join(PROJECT_ROOT, "figure_4.png")
SUPP_FIGURE_5_PATH = os.path.join(PROJECT_ROOT, "supp_fig_5.png")
FIGURE_4_EXAMPLE_MGF = _under(
    MABS_BENCHMARK_DIR, "xa_novo", "PXD060500_36H6", "mgf",
    "36H6-pepsin-HCD-20240524.mgf"
)
FIGURE_4_ASSEMBLY_TSV_PATHS = (
    _under(MABS_ASSEMBLY_DIR, "3arm_summary_solo_k7to11.tsv"),
    _under(MABS_ASSEMBLY_DIR, "3arm_summary_beslic_k7to11.tsv"),
)
FIGURE_4_MAB_REFERENCE_PATHS = {
    "2B4": _under(MABS_REFERENCES_DIR, "2B4_ref.fasta"),
    "36H6": _under(MABS_REFERENCES_DIR, "36H6_ref.fasta"),
    "85F7": _under(MABS_REFERENCES_DIR, "85F7_ref.fasta"),
    "S2P6": _under(MABS_REFERENCES_DIR, "S2P6_ref.fasta"),
    "IgG1_Human_H": _under(MABS_REFERENCES_DIR, "IgG1_Human_H_ref.fasta"),
    "IgG1_Human_L": _under(MABS_REFERENCES_DIR, "IgG1_Human_L_ref.fasta"),
    "Herceptin": _under(MABS_REFERENCES_DIR, "Herceptin_ref.fasta"),
    "anti-FLAG-M2": _under(MABS_REFERENCES_DIR, "anti-FLAG-M2_ref.fasta"),
    "WIgG1_H": _under(MABS_REFERENCES_DIR, "WIgG1_mouse_H_ref.fasta"),
    "WIgG1_L": _under(MABS_REFERENCES_DIR, "WIgG1_mouse_L_ref.fasta"),
}


def figure_4_xanovo_results_dir(mab: str) -> str:
    return os.path.join(
        MABS_RESULTS_DIR, "xa_novo", f"PXD060500_{mab}", "casanovo_results"
    )


def figure_4_nontryp_results_dir(mab: str, protease: str) -> str:
    return os.path.join(
        MABS_RESULTS_DIR, "nontryp", mab, protease, "casanovo_results"
    )


# ── Kingdoms-of-Life benchmark ────────────────────────────────────────────
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
KOL_STAGING_DIR = os.environ.get("DNPS_STAGING_DIR") or _under(
    WORK_DIR, "kingdoms_of_life", "staging"
)
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
    return os.path.join(require_data_path("KoL outputs"), "work",
                        "kingdoms_of_life", species)


def kol_species_subset_path(species: str) -> str:
    return os.path.join(KOL_SUBSETS_DIR, f"{species}.mgf")


# ── Benchmark datasets ────────────────────────────────────────────────────
# DatasetPaths itself is a generic container and stays in peppr; only these
# paper-specific instances live here.
_RESULT_RUN_PATH = _under_data("results", SPECIES[ACTIVE_SPECIES]["run_name"])

# De novo runs on the GT-search 1%-FDR spectra (mgf_gt), which are labelled with
# the ground-truth variant peptide and stored in ProForma form.
PROTEOMETOOLS_SAAV_MGF_GLOB = _under(PROTEOMETOOLS_SAAV_DIR, "mgf_gt", "*.mgf")
PROTEOMETOOLS_SAAV_DATASET = DatasetPaths(
    name="ProteomeTools_SAAV",
    rawfile_glob=None,
    msms_glob=None,
    mgf_unprocessed_dir=None,
    final_mgf_glob=PROTEOMETOOLS_SAAV_MGF_GLOB,
    mztab_path_dnps=_under(_RESULT_RUN_PATH, "proteometools_saav_dnps.mztab"),
    mztab_path_fusion=_under(_RESULT_RUN_PATH, "proteometools_saav_hybrid.mztab"),
)

_plm_tag = f"_plm{PLM_SPECIES}" if PLM_SPECIES != ACTIVE_SPECIES else ""
NINE_SPECIES_PROFORMA_DIR = _under(
    NINE_SPECIES_BENCHMARK_DIR, SPECIES[ACTIVE_SPECIES]["benchmark_dir"]
)
NINE_SPECIES_DATASET = DatasetPaths(
    name=f"9S_{ACTIVE_SPECIES}",
    rawfile_glob=None,
    msms_glob=None,
    mgf_unprocessed_dir=None,
    final_mgf_glob=_under(NINE_SPECIES_PROFORMA_DIR, "*.mgf"),
    mztab_path_dnps=_under(
        _RESULT_RUN_PATH, f"9s_{ACTIVE_SPECIES}{_plm_tag}_dnps.mztab"),
    mztab_path_fusion=_under(
        _RESULT_RUN_PATH, f"9s_{ACTIVE_SPECIES}{_plm_tag}_hybrid.mztab"),
)
