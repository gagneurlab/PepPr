"""Antibody reference, region-classification, and Figure 4 result helpers."""

from __future__ import annotations

import glob
import os
import re

from dnps_hybrid import const


ARM_STYLE = {
    "vanilla": {"label": "Casanovo v5", "color": const.COLOR_CASANOVO},
    "xanovo_v3": {"label": "XA-Novo v3", "color": const.COLOR_XANOVO},
    "germline_mouse_sw": {
        "label": f"{const.LABEL_CASANOVO_PEPPR_COMPACT} (mouse)",
        "color": const.COLOR_PP,
    },
    "germline_human_sw": {
        "label": f"{const.LABEL_CASANOVO_PEPPR_COMPACT} (human)",
        "color": const.COLOR_PP,
    },
}

XANOVO_PROTEASES = ["aspn", "chymo", "elastase", "pepsin", "trypsin"]
BESLIC_IGG_H_PROTEASES = ["aspn", "chymo", "gluc", "lysc", "proteinasek"]
BESLIC_IGG_L_PROTEASES = ["aspn", "chymo", "gluc", "lysc", "proteinasek"]
HERCEPTIN_PROTEASES = ["elastase", "gluc", "lysc", "lysn", "thermolysin"]
FLAG_PROTEASES = [
    "aspn",
    "chymo",
    "elastase",
    "gluc",
    "lysc",
    "lysn",
    "thermolysin",
]
WIGG_PROTEASES = ["aspn", "chymo"]

MAB_SPECS = {
    "2B4": {
        "species": "mouse",
        "pp_arm": "germline_mouse_sw",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["2B4"],
        "proteases": XANOVO_PROTEASES,
        "source": "xa_novo",
    },
    "36H6": {
        "species": "mouse",
        "pp_arm": "germline_mouse_sw",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["36H6"],
        "proteases": XANOVO_PROTEASES,
        "source": "xa_novo",
    },
    "85F7": {
        "species": "mouse",
        "pp_arm": "germline_mouse_sw",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["85F7"],
        "proteases": XANOVO_PROTEASES,
        "source": "xa_novo",
    },
    "S2P6": {
        "species": "human",
        "pp_arm": "germline_human_sw",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["S2P6"],
        "proteases": XANOVO_PROTEASES,
        "source": "xa_novo",
    },
    "IgG1_Human_H": {
        "species": "human (HC)",
        "pp_arm": "antibody_human",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["IgG1_Human_H"],
        "proteases": BESLIC_IGG_H_PROTEASES,
        "source": "beslic",
    },
    "IgG1_Human_L": {
        "species": "human (LC)",
        "pp_arm": "antibody_human",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["IgG1_Human_L"],
        "proteases": BESLIC_IGG_L_PROTEASES,
        "source": "beslic",
    },
    "Herceptin": {
        "species": "human",
        "pp_arm": "antibody_human",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["Herceptin"],
        "proteases": HERCEPTIN_PROTEASES,
        "source": "beslic",
    },
    "anti-FLAG-M2": {
        "species": "mouse",
        "pp_arm": "antibody_mouse",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["anti-FLAG-M2"],
        "proteases": FLAG_PROTEASES,
        "source": "beslic",
    },
    "WIgG1_H": {
        "species": "mouse (HC)",
        "pp_arm": "antibody_mouse",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["WIgG1_H"],
        "proteases": WIGG_PROTEASES,
        "source": "beslic",
    },
    "WIgG1_L": {
        "species": "mouse (LC)",
        "pp_arm": "antibody_mouse",
        "vanilla_arm": "noplm",
        "ref": const.FIGURE_4_MAB_REFERENCE_PATHS["WIgG1_L"],
        "proteases": WIGG_PROTEASES,
        "source": "beslic",
    },
}


def parse_header(header_tail: str):
    """Return three CDR ranges and the V/C boundary from a FASTA header."""
    try:
        positions = [
            int(value)
            for value in header_tail.replace(",", " ").split()
            if value
        ]
    except ValueError:
        return None
    if len(positions) < 7:
        return None
    return [
        (positions[0], positions[1]),
        (positions[2], positions[3]),
        (positions[4], positions[5]),
    ], positions[6]


def read_fasta_with_regions(path: str) -> dict[str, dict]:
    """Read antibody chains and region coordinates from an annotated FASTA."""
    records = {}
    name = None
    header_tail = ""
    sequence = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    region_info = parse_header(header_tail)
                    records[name] = {
                        "seq": "".join(sequence),
                        "cdrs": region_info[0] if region_info else None,
                        "vc": region_info[1] if region_info else None,
                    }
                header = line[1:].split(None, 1)
                name = header[0]
                header_tail = header[1] if len(header) > 1 else ""
                sequence = []
            else:
                sequence.append(line.strip())
    if name is not None:
        region_info = parse_header(header_tail)
        records[name] = {
            "seq": "".join(sequence),
            "cdrs": region_info[0] if region_info else None,
            "vc": region_info[1] if region_info else None,
        }
    return records


def il_fold(sequence: str) -> str:
    """Fold isoleucine into leucine for mass-equivalent matching."""
    return sequence.replace("I", "L")


def strip_mods(sequence: str | None) -> str:
    """Remove modification annotations from a peptide sequence."""
    if sequence is None:
        return ""
    sequence = re.sub(r"\[[^\]]+\]-?", "", sequence)
    sequence = re.sub(r"[+\-][0-9]+(?:\.[0-9]+)?", "", sequence)
    return sequence.replace("-", "")


def find_all_matches(peptide: str, reference: str):
    """Yield 1-based inclusive matches after I/L folding."""
    peptide = il_fold(peptide)
    reference = il_fold(reference)
    if not peptide:
        return
    start = 0
    while True:
        index = reference.find(peptide, start)
        if index < 0:
            return
        yield index + 1, index + len(peptide)
        start = index + 1


def region_at(position: int, cdrs: list[tuple[int, int]], vc: int) -> str:
    """Return the antibody region containing a 1-based position."""
    if position > vc:
        return "C"
    if any(start <= position <= end for start, end in cdrs):
        return "CDR"
    return "FR"


def classify_alignment(
    alignment_start: int,
    alignment_end: int,
    cdrs: list[tuple[int, int]],
    vc: int,
):
    """Classify the antibody regions spanned by an aligned peptide."""
    regions = {
        region_at(position, cdrs, vc)
        for position in range(alignment_start, alignment_end + 1)
    }
    crosses_vc = "C" in regions and bool(regions & {"FR", "CDR"})
    crosses_fr_cdr = "CDR" in regions and "FR" in regions
    if crosses_fr_cdr and crosses_vc:
        category = "FR<->CDR + V<->C"
    elif crosses_vc:
        category = "V<->C"
    elif crosses_fr_cdr:
        category = "FR<->CDR"
    elif regions == {"C"}:
        category = "C-only"
    elif regions == {"FR"}:
        category = "FR-only"
    elif regions == {"CDR"}:
        category = "CDR-only"
    else:
        category = "?"
    return category, regions


def categorize_gt(gt_sequence: str, reference_chains: dict[str, dict]) -> str | None:
    """Return the antibody-region category of a ground-truth peptide."""
    peptide = strip_mods(gt_sequence)
    if len(peptide) < 5:
        return None
    for chain in reference_chains.values():
        matches = list(find_all_matches(peptide, chain["seq"]))
        if matches:
            start, end = matches[0]
            category, _ = classify_alignment(
                start, end, chain["cdrs"], chain["vc"]
            )
            return category
    return None


def _mztab_psm_count(path: str) -> int:
    """Count PSM rows without loading the full mzTab."""
    with open(path) as handle:
        return sum(line.startswith("PSM\t") for line in handle)


def find_mztab(mab: str, protease: str, arm: str) -> str | None:
    """Pick the latest valid mzTab for one mAb, protease, and arm."""
    spec = MAB_SPECS[mab]
    if spec["source"] == "beslic":
        results_dir = const.figure_4_nontryp_results_dir(mab, protease)
        patterns = [
            os.path.join(
                results_dir,
                f"casanovo_{mab}_{protease}_{arm}_{mab}_*.mztab",
            ),
            os.path.join(
                results_dir,
                f"casanovo_{mab}_{protease}_{arm}_[0-9]*.mztab",
            ),
        ]
        matches = [
            path
            for pattern in patterns
            for path in glob.glob(pattern)
        ]
    else:
        results_dir = const.figure_4_xanovo_results_dir(mab)
        matches = glob.glob(
            os.path.join(
                results_dir,
                f"casanovo_{mab}_{protease}_{arm}_{mab}_*.mztab",
            )
        )
    matches = sorted(matches, key=os.path.getmtime, reverse=True)
    return next(
        (path for path in matches if _mztab_psm_count(path) > 100),
        None,
    )
