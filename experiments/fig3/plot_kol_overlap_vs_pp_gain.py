#!/usr/bin/env python3
"""Scatter the +PP precision lift (Casanovo+PepPr − Casanovo) against the
pepLM peptide overlap with each KoL target species' tryptic proteome.

Inputs:
  - kingdom_peptide_overlap.csv: `pct_plm_in_kol` for each
    (plm_species, kol_species) pair. Generated automatically from the proteome
    FASTAs and KoL subset MGFs when missing (or with --rebuild-overlap); see
    build_overlap_csv below.
  - <runs-dir>/<run_dir>/casanovo_only.log and
    casanovo_pp_<plm_species>.log: casanovo "Peptide Precision: X%" lines.

Run directories are discovered automatically: every subdirectory that has
both logs is used.  Folder names may be KoL short keys (``bos_tauros``),
Search.zip-style names (``Bos_tauros``, ``Homo_sapiens``), or legacy aliases
(``Caenorhabditis_elegans`` → Müller's ``Canaerohabidis_elegans``).  Mapping
uses ``peppr.const.KOL_SPECIES_DIRS`` plus a small list of cases where
the overlap CSV uses a short epithet (``rerio``, ``scrofa``, …) instead of the
underscored MaxQuant folder name.

Multiple pepLMs can be plotted on the same figure (default: human + mouse);
each gets its own marker and color, while a **single** linear fit is drawn
through **all** points pooled across pepLMs (after dropping same-species pairs:
human↔Homo_sapiens, mouse↔Mus_musculus).  Δ values are computed against
``casanovo_only.log`` in the same run dir, so the baseline is shared across
pepLMs (consistent x-projection: each KoL species' baseline is one horizontal
slice that all pepLMs lift from).

Point labels use common or abbreviated names (e.g. human, cow, c. elegans);
pass ``--scientific-labels`` to annotate raw ``kol_species`` strings instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from peppr.const import (  # noqa: E402
    COLOR_CASANOVO,
    COLOR_DARK_GRAY,
    COLOR_MID_GRAY,
    COLOR_PLUM,
    COLOR_PP,
    FASTAS_DIR,
    KOL_RESULTS_DIR,
    KOL_SPECIES_DIRS,
    KOL_SUBSETS_DIR,
    PROJECT_ROOT,
    SPECIES,
)

DEFAULT_RUNS_DIR = Path(KOL_RESULTS_DIR) / "runs"
DEFAULT_OVERLAP_CSV = Path(PROJECT_ROOT) / "kingdom_peptide_overlap.csv"
DEFAULT_PLM_SPECIES = ["human_iso", "mouse"]
DEFAULT_OUT_PNG = "kol_overlap_vs_pp_gain.png"


# ---------------------------------------------------------------------------
# Overlap CSV generation
#
# kingdom_peptide_overlap.csv gives, per (plm_species, kol_species) pair,
# ``pct_plm_in_kol`` = the percentage of the KoL target species' identified
# peptides that also occur in that pepLM's training set.
#
# The pepLM training set is the in-silico tryptic digest of the pepLM's
# proteome FASTA (peppr.const.SPECIES[plm]["fasta"]), reproduced here with the
# exact rules peppr.prepare_data uses in trypsin mode
# (generate_plm_training_data -> digest_protein(protein, 1, False)):
#   * cleave after K/R, up to 1 missed cleavage, no methionine excision;
#   * peptide length 4..PLM_BLOCK_SIZE-2 residues (the +'$' terminator must fit
#     in a PLM_BLOCK_SIZE row);
#   * isoleucine folded to leucine and selenocysteine to cysteine (the pepLM
#     vocabulary is I->L, U->C), peptides with any other non-standard residue
#     dropped (encode() would KeyError on them).
# Target peptides come from the KoL subset MGFs (KOL_SUBSETS_DIR/<key>.mgf)
# SEQ= lines, stripped to bare amino acids and folded the same way. Set
# membership is orientation-independent, so peptides are compared forward (N->C)
# rather than reversed as the model stores them.
# ---------------------------------------------------------------------------

_PLM_MAX_PEP_LEN = 100          # peppr.const.PLM_BLOCK_SIZE
# VOCAB amino acids after the I->L / U->C fold (peppr.const.VOCAB minus I).
_VOCAB_AA = frozenset("HYACQMPFWESTDVRNKLG")
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_KR_RE = re.compile(r"[KR]")


def _fold_il_uc(seq: str) -> str:
    """Fold isoleucine->leucine and selenocysteine->cysteine (pepLM vocab)."""
    return seq.replace("I", "L").replace("U", "C")


def _bare_peptide(seq: str) -> str:
    """SEQ= value (e.g. 'GDPEM[Oxidation]EQK') -> folded bare amino acids."""
    seq = _MOD_BRACKET_RE.sub("", seq)
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return _fold_il_uc(seq)


def _read_fasta_sequences(path: Path):
    """Yield raw protein sequences from a FASTA (no external deps)."""
    chunk: list[str] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if chunk:
                    yield "".join(chunk)
                    chunk = []
            else:
                chunk.append(line.strip())
    if chunk:
        yield "".join(chunk)


def _tryptic_training_peptides(fasta_path: Path) -> set[str]:
    """Reproduce the pepLM tryptic training peptide set for one proteome.

    Mirrors peppr.prepare_data.digest_protein(protein, 1, False) plus the
    length / vocabulary filters applied in generate_plm_training_data.
    """
    max_missed = 1
    max_pep = _PLM_MAX_PEP_LEN - 2   # + '$' terminator must fit in PLM_BLOCK_SIZE
    peptides: set[str] = set()
    for raw in _read_fasta_sequences(fasta_path):
        protein = "." + raw                 # N-terminus marker, as in training
        sites = sorted({0, len(protein),
                        *(m.start() + 1 for m in _KR_RE.finditer(protein))})
        for i in range(len(sites) - 1):
            for j in range(i + 1, min(i + 2 + max_missed, len(sites))):
                pep = protein[sites[i]:sites[j]]
                if not (3 < len(pep) <= max_pep):
                    continue
                aa = _fold_il_uc(pep.replace(".", ""))
                if aa and all(c in _VOCAB_AA for c in aa):
                    peptides.add(aa)
    return peptides


def _kol_target_peptides(subset_mgf: Path) -> set[str]:
    """Unique folded bare-AA peptides identified for one KoL species."""
    peptides: set[str] = set()
    with open(subset_mgf) as handle:
        for line in handle:
            if line.startswith("SEQ="):
                aa = _bare_peptide(line[4:].strip())
                if aa:
                    peptides.add(aa)
    return peptides


def build_overlap_csv(out_path: Path, plm_species: list[str]) -> None:
    """(Re)compute kingdom_peptide_overlap.csv from proteome FASTAs + KoL subsets.

    One row per (plm_species, kol_species): ``pct_plm_in_kol`` is the percentage
    of the KoL species' identified peptides that also appear in the pepLM's
    tryptic training set. kol_species uses the same labels the overlap consumers
    expect (via _overlap_kol_for_const_key).
    """
    subset_dir = Path(KOL_SUBSETS_DIR)
    subsets = sorted(subset_dir.glob("*.mgf"))
    if not subsets:
        raise FileNotFoundError(f"no KoL subset MGFs under {subset_dir}")

    # Target peptide sets are pepLM-independent — compute once.
    targets: dict[str, set[str]] = {}
    for mgf in subsets:
        key = mgf.stem
        if key not in KOL_SPECIES_DIRS:
            print(f"  ! subset {mgf.name} has no KOL_SPECIES_DIRS key — skipping",
                  file=sys.stderr)
            continue
        targets[key] = _kol_target_peptides(mgf)

    rows: list[dict] = []
    for plm in plm_species:
        cfg = SPECIES.get(plm)
        if cfg is None:
            print(f"  ! no SPECIES config for plm_species={plm!r} — skipping",
                  file=sys.stderr)
            continue
        fasta_path = Path(FASTAS_DIR) / cfg["fasta"]
        print(f"[overlap] digesting {plm} proteome {fasta_path.name} ...")
        train = _tryptic_training_peptides(fasta_path)
        print(f"[overlap]   {len(train):,} unique training peptides")
        for key, target in targets.items():
            if not target:
                continue
            n_in = sum(1 for p in target if p in train)
            rows.append({
                "plm_species": plm,
                "kol_species": _overlap_kol_for_const_key(key),
                "pct_plm_in_kol": round(100.0 * n_in / len(target), 6),
                "n_target_peptides": len(target),
                "n_in_training": n_in,
            })

    if not rows:
        raise RuntimeError("overlap build produced no rows")
    df = pd.DataFrame(rows).sort_values(["plm_species", "pct_plm_in_kol"])
    df.to_csv(out_path, index=False)
    print(f"[overlap] wrote {len(df)} rows to {out_path}")

# const short-key -> kol_species label in kingdom_peptide_overlap.csv when it
# is NOT simply search_dir.replace(" ", "_").  (Epithets / Norvegicus casing.)
_KEY_TO_OVERLAP_KOL: dict[str, str] = {
    "danio_rerio": "rerio",
    "didelphis_didelphinae": "didelphinae",
    "oryctolagus_cuniculus": "cuniculus",
    "rattus_norvegicus": "Norvegicus",
    "sus_scrofa": "scrofa",
}

# Old eval folder names / typos -> CSV kol_species (must exist for chosen plm).
_LEGACY_RUN_DIR_TO_KOL: dict[str, str] = {
    "Caenorhabditis_elegans": "Canaerohabidis_elegans",
}

# kingdom_peptide_overlap.csv ``kol_species`` -> short label for plot annotations.
# Epithets (rerio, scrofa, …) match CSV spelling.  Unknown keys fall back to
# ``_kol_display_fallback``.
_KOL_SPECIES_DISPLAY: dict[str, str] = {
    # --- animalia (common / familiar names) ---
    "Homo_sapiens": "human",
    "Mus_musculus": "mouse",
    "Bos_tauros": "cow",
    "Canis_lupus": "dog",
    "scrofa": "pig",
    "Gallus_gallus": "chicken",
    "rerio": "zebrafish",
    "cuniculus": "rabbit",
    "Norvegicus": "rat",
    "didelphinae": "opossum",
    "Canaerohabidis_elegans": "nematode",
    "Drosophila_melanogaster": "fruit fly",
    "Cricetulus_griseus": "hamster",
    "Oryzias_melastigma": "killifish",
    "Tardigrade": "tardigrade",
    # --- plantae ---
    "aestivum": "wheat",
    "Arabidopsis_thaliana_Callus": "thale cress (callus)",
    "Arabidopsis_thaliana_Root": "thale cress (root)",
    "Arabidopsis_thaliana_sprout": "thale cress (sprout)",
    "hirsicum": "cotton",
    "max": "soybean",
    "vinefera": "grape",
    # --- fungi ---
    "cerevisiae": "yeast",
    "crassa": "N. crassa",
    "Fusarium_oxisporum": "Fusarium",
    # --- bacteria (abbreviated binomials / short names) ---
    "Akkermansia_municiphila": "A. muciniphila",
    "Aminomonas_paucivorans": "A. paucivorans",
    "Bacillus_subtilis": "B. subtilis",
    "Bacteroides_fragilis": "B. fragilis",
    "Bacteroides_uniformis": "B. uniformis",
    "Bacteroides_vulgatus": "B. vulgatus",
    "Bacteroide_thetaiotaomicron": "B. thetaiotaomicron",
    "Bifidobacterium_adolescentis": "B. adolescentis",
    "Bifidobacterium_longum_subsp.longum": "B. longum",
    "Blautia_obeum": "B. obeum",
    "bromii": "R. bromii",
    "Caldisericum_exile": "C. exile",
    "Caldithryx_abyssi": "C. abyssi",
    "Clostridium_bolteae": "C. bolteae",
    "Clostridium_perfringens": "C. perfringens",
    "Clostridium_saccharolyticum": "C. saccharolyticum",
    "Collinsella_aerofaciens": "C. aerofaciens",
    "commune": "T. commune",
    "copri": "P. copri",
    "Coprococcus_comes": "C. comes",
    "Deinococcus_radiodurans": "D. radiodurans",
    "denticola": "T. denticola",
    "Dentriovibrio_acetiphilus": "D. acetiphilus",
    "Dethiosulfovibrio_peptidovorans": "D. peptidovorans",
    "Dictyoglomus_thermophilum": "D. thermophilum",
    "distasonis": "P. distasonis",
    "Dorea_formicigenerans": "D. formicigenerans",
    "Eggerthella_lenta": "E. lenta",
    "Escherichia_coli": "E. coli",
    "Eubacterium_rectale": "E. rectale",
    "extracellular": "Mycoplasma (extracellular)",
    "Fusobacterium_nucleatum_subspecies_nucleatum": "F. nucleatum",
    "gnavus": "R. gnavus",
    "Granulicella_tundricola": "G. tundricola",
    "halophila": "P. halophila",
    "hypogea": "P. hypogea",
    "indicus": "T. indicus",
    "intestinalis": "R. intestinalis",
    "intracellular": "Mycoplasma (intracellular)",
    "Lactobacillus_paracaesei": "L. paracasei",
    "marina": "P. marina",
    "maritima": "T. maritima",
    "merdae": "P. merdae",
    "parasanguinis": "S. parasanguinis",
    "parvual": "V. parvula",
    "splanchnicus": "O. splanchnicus",
    "thermophila": "S. thermophila",
    "torques": "R. torques",
    # --- archaea ---
    "acidocaldicarius": "S. acidocaldarius",
    "acidophilum": "T. acidophilum",
    "Archaeoglobus_fulgidus": "A. fulgidus",
    "Archaeoglobus_profundus": "A. profundus",
    "Ferroglobus_placidus": "F. placidus",
    "furiosus": "P. furiosus",
    "Haloarcula_marismortui": "H. marismortui",
    "Halobacterium_sp": "Halobacterium",
    "Haloferax_mediterranei": "H. mediterranei",
    "litoralis": "T. litoralis",
    "Methanocaldococcus_jannaschii": "M. jannaschii",
    "Methanopyrus_kandleri": "M. kandleri",
    "Methanosarcina_barkeri": "M. barkeri",
    "Methanothermobacter_marburgensis": "M. marburgensis",
    "Methanothermobacter_thermautotrophicus": "M. thermautotrophicus",
    "solfaticarius": "S. solfataricus",
    "tenax": "T. tenax",
    "torridus": "P. torridus",
    "volcanicum": "T. volcanium",
}


def _kol_display_fallback(kol: str) -> str:
    """Underscored token -> ``G. epithet`` when the first token looks like a genus."""
    parts = kol.split("_")
    if len(parts) >= 2 and parts[0] and parts[0][0].isupper():
        epithet = parts[1]
        return f"{parts[0][0]}. {epithet}"
    return kol.replace("_", " ")


def kol_display_name(kol: str, scientific: bool = False) -> str:
    """Label for a ``kol_species`` row (plot annotations and optional CSV column)."""
    if scientific:
        return kol
    return _KOL_SPECIES_DISPLAY.get(kol, _kol_display_fallback(kol))


# (color, marker) per pepLM species. Anything not listed falls back to the
# matplotlib default cycle.
_PLM_STYLE: dict[str, tuple[str, str]] = {
    "human":    (COLOR_PP, "o"),        # deep navy circles
    "human_iso": (COLOR_PP, "o"),
    "mouse":    (COLOR_CASANOVO, "s"),  # casanovo orange squares
    "honeybee": (COLOR_PLUM, "^"),  # plum triangles
    "tomato":   ("#27ae60", "D"),  # green diamonds
    "yeast":    ("#c0392b", "P"),  # red plus
}
_FALLBACK_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Legend label per pepLM species; anything not listed falls back to
# "{plm} prior".
_PLM_LEGEND: dict[str, str] = {
    "human": "human prior",
    "human_iso": "human prior",
    "mouse": "mouse prior",
}

# plm_species (CLI / overlap CSV) -> kingdom_peptide_overlap ``kol_species`` for
# the KoL run that uses the same species as the prior.  Those rows are dropped
# from the scatter and fit so cross-species overlap drives the relationship.
_PLM_SELF_KOL: dict[str, str] = {
    "human": "Homo_sapiens",
    "human_iso": "Homo_sapiens",
    "mouse": "Mus_musculus",
}

_PEP_PREC_RE = re.compile(r"Peptide Precision:\s*([0-9.]+)\s*%")


def _overlap_kol_for_const_key(key: str) -> str:
    if key in _KEY_TO_OVERLAP_KOL:
        return _KEY_TO_OVERLAP_KOL[key]
    return KOL_SPECIES_DIRS[key].replace(" ", "_")


def _build_run_dir_to_key_index() -> dict[str, str]:
    """Lowercased run-dir token -> const short key (many-to-one)."""
    idx: dict[str, str] = {}
    for key, search in KOL_SPECIES_DIRS.items():
        idx[key.lower()] = key
        su = search.replace(" ", "_")
        idx[su.lower()] = key
    return idx


_RUN_DIR_TO_KEY = _build_run_dir_to_key_index()


def resolve_run_dir_to_kol(run_dir: str, valid_kols: set[str]) -> str | None:
    """Map a runs/ subdir name to kingdom_peptide_overlap ``kol_species``."""
    if run_dir in valid_kols:
        return run_dir
    if run_dir in _LEGACY_RUN_DIR_TO_KOL:
        kol = _LEGACY_RUN_DIR_TO_KOL[run_dir]
        return kol if kol in valid_kols else None

    key = _RUN_DIR_TO_KEY.get(run_dir.lower())
    if key is not None:
        kol = _overlap_kol_for_const_key(key)
        if kol in valid_kols:
            return kol

    # e.g. run_dir already matches underscored search dir with different case
    for k, search in KOL_SPECIES_DIRS.items():
        su = search.replace(" ", "_")
        if run_dir == su or run_dir.lower() == su.lower():
            kol = _overlap_kol_for_const_key(k)
            if kol in valid_kols:
                return kol
    return None


def parse_peptide_precision(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    last: float | None = None
    text = log_path.read_text(errors="replace")
    for match in _PEP_PREC_RE.finditer(text):
        try:
            last = float(match.group(1))
        except ValueError:
            continue
    return last


def _load_overlap(overlap_csv: Path, plm_species: str) -> dict[str, float]:
    overlap = pd.read_csv(overlap_csv)
    overlap = overlap[overlap["plm_species"] == plm_species]
    if overlap.empty:
        return {}
    if overlap.duplicated(subset=["kol_species"]).any():
        print(
            f"  ! warning: duplicate kol_species rows in {overlap_csv} for "
            f"plm={plm_species}; using first occurrence",
            file=sys.stderr,
        )
        overlap = overlap.drop_duplicates(subset=["kol_species"], keep="first")
    return overlap.set_index("kol_species")["pct_plm_in_kol"].to_dict()


def collect_runs_for_plm(
    runs_dir: Path,
    plm_species: str,
    overlap_by_kol: dict[str, float],
) -> list[dict]:
    """One row per run dir that has casanovo_only.log + casanovo_pp_<plm>.log
    AND a resolvable kol_species in the overlap CSV.
    """
    valid_kols = set(overlap_by_kol)
    out: list[dict] = []
    cas_log_name = "casanovo_only.log"
    pp_log_name = f"casanovo_pp_{plm_species}.log"

    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        cas_log = sub / cas_log_name
        pp_log = sub / pp_log_name
        if not pp_log.exists():
            continue  # quiet skip — this run just didn't include this pepLM
        kol = resolve_run_dir_to_kol(sub.name, valid_kols)
        if kol is None:
            print(
                f"  ! [{plm_species}] cannot map run dir {sub.name!r} to a "
                f"kol_species in overlap CSV — skipping",
                file=sys.stderr,
            )
            continue
        cas_pct = parse_peptide_precision(cas_log)
        pp_pct = parse_peptide_precision(pp_log)
        if cas_pct is None or pp_pct is None:
            print(
                f"  ! [{plm_species}] missing metrics for {sub.name} "
                f"(cas={cas_pct}, pp={pp_pct}) — skipping",
                file=sys.stderr,
            )
            continue
        out.append({
            "plm_species": plm_species,
            "run_dir": sub.name,
            "kol_species": kol,
            "pct_plm_in_kol": overlap_by_kol[kol],
            "casanovo_pep_precision": cas_pct,
            "casanovo_pp_pep_precision": pp_pct,
            "delta_pep_precision": pp_pct - cas_pct,
        })
    return out


def _style_for(plm: str, idx: int) -> tuple[str, str]:
    if plm in _PLM_STYLE:
        return _PLM_STYLE[plm]
    return (
        _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
        _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--overlap-csv", type=Path, default=DEFAULT_OVERLAP_CSV)
    parser.add_argument(
        "--plm-species",
        nargs="+",
        default=DEFAULT_PLM_SPECIES,
        help=("pepLM species to plot (one or more). For each, looks for "
              "casanovo_pp_<name>.log in every run dir AND filters the "
              "overlap CSV to plm_species==<name>. "
              f"Default: {' '.join(DEFAULT_PLM_SPECIES)}."),
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PNG)
    parser.add_argument("--csv-out", default="kol_overlap_vs_pp_gain.csv")
    parser.add_argument(
        "--scientific-labels",
        action="store_true",
        help="Annotate points with overlap CSV kol_species (folder-style) "
        "instead of common / abbreviated names.",
    )
    parser.add_argument(
        "--rebuild-overlap",
        action="store_true",
        help="Recompute the overlap CSV from proteome FASTAs + KoL subset MGFs "
        "before plotting (otherwise it is built only when missing).",
    )
    args = parser.parse_args()

    if args.rebuild_overlap or not args.overlap_csv.exists():
        reason = "--rebuild-overlap" if args.overlap_csv.exists() else "not found"
        print(f"[overlap] {args.overlap_csv} {reason} — building")
        build_overlap_csv(args.overlap_csv, args.plm_species)

    all_rows: list[dict] = []
    runs_per_plm: dict[str, list[dict]] = {}
    for plm in args.plm_species:
        overlap_by_kol = _load_overlap(args.overlap_csv, plm)
        if not overlap_by_kol:
            print(
                f"  ! no overlap rows for plm_species={plm} — skipping",
                file=sys.stderr,
            )
            continue
        rows = collect_runs_for_plm(args.runs_dir, plm, overlap_by_kol)
        if not rows:
            print(f"  ! no usable runs for plm_species={plm}", file=sys.stderr)
            continue
        runs_per_plm[plm] = rows
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("no usable runs found for any requested pepLM")

    df = pd.DataFrame(all_rows).sort_values(["plm_species", "pct_plm_in_kol"])
    _self_kol = df["plm_species"].map(_PLM_SELF_KOL)
    df = df[~(_self_kol.notna() & (df["kol_species"] == _self_kol))]

    df["kol_label"] = df["kol_species"].map(
        lambda k: kol_display_name(k, scientific=args.scientific_labels),
    )
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote {len(df)} rows to {args.csv_out}\n")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Δ=0 reference (same stroke style the linear fit used previously).
    ax.axhline(
        0.0,
        color=COLOR_DARK_GRAY,
        lw=1.2,
        ls="--",
        alpha=0.85,
        zorder=2,
    )

    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]
    try:
        from adjustText import adjust_text  # noqa: F401
        _have_adjust = True
    except ImportError:
        _have_adjust = False

    x_pooled = df["pct_plm_in_kol"].to_numpy()
    y_pooled = df["delta_pep_precision"].to_numpy()
    if len(x_pooled) >= 2:
        slope, intercept = np.polyfit(x_pooled, y_pooled, 1)
        xs = np.linspace(x_pooled.min(), x_pooled.max(), 100)
        r_pooled = np.corrcoef(x_pooled, y_pooled)[0, 1]
        ax.plot(
            xs,
            slope * xs + intercept,
            color=COLOR_MID_GRAY,
            lw=1.0,
            ls="-",
            alpha=0.85,
            zorder=1,
            label=f"Linear fit (n={len(x_pooled)}, r={r_pooled:+.2f})",
        )

    all_texts = []
    for idx, plm in enumerate(args.plm_species):
        sub = df[df["plm_species"] == plm]
        if sub.empty:
            continue
        sub = sub.sort_values("pct_plm_in_kol")
        x = sub["pct_plm_in_kol"].to_numpy()
        y = sub["delta_pep_precision"].to_numpy()
        color, marker = _style_for(plm, idx)

        ax.scatter(
            x, y, s=70, color=color, marker=marker,
            edgecolors="white", linewidths=0.6, zorder=3,
            label=_PLM_LEGEND.get(plm, f"{plm} prior"),
        )

        point_labels = sub["kol_species"].map(
            lambda k: kol_display_name(k, scientific=args.scientific_labels),
        )
        if _have_adjust:
            for xi, yi, name in zip(x, y, point_labels):
                all_texts.append(
                    ax.text(xi, yi, name, fontsize=9, color="black",
                            path_effects=glow, zorder=5)
                )
        else:
            for xi, yi, name in zip(x, y, point_labels):
                ax.annotate(
                    name, (xi, yi), xytext=(4, 4),
                    textcoords="offset points", fontsize=9,
                    color="black", path_effects=glow,
                )

    if _have_adjust and all_texts:
        from adjustText import adjust_text
        adjust_text(
            all_texts, ax=ax, seed=42,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.6),
        )

    ax.set_xlabel(
        "% of prior peptides in target proteome (overlap)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Δ peptide precision",
        fontsize=12,
    )
    ax.set_title(
        "Human/mouse prior, KoL [Animal] data",
        fontsize=13,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="best", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
