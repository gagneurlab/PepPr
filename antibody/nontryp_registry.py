"""Registry of (mAb, non-tryptic-protease) runs.

Each Run describes a single FragPipe + Casanovo end-to-end pipeline:
  - source MGF (centroided) holding MS2 spectra for one protease's run
  - per-mAb target+decoy FASTA
  - MSFragger enzyme params for the protease
  - per-protease peer-tool overlays (figshare summary.csv etc.)

The registry is consumed by prepare_nontryp.py / SLURM array jobs / the
per-protease plotting script.  Tryptic runs live in plot_pc_curves_mab.MABS;
this module is strictly non-tryptic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dnps_hybrid.const import (
    BESLIC_DIR,
    MABS_BENCHMARK_DIR,
    MABS_REFERENCES_DIR,
    MABS_RESULTS_DIR,
    WORK_DIR,
)

BESLIC = BESLIC_DIR
_FIGSHARE_ROOT = os.path.join(MABS_BENCHMARK_DIR, "figshare_21394143")
FIGSHARE_BES = os.path.join(
    _FIGSHARE_ROOT, "01-RawData", "MSV000079801_CompleteAssembly", "Centroided"
)
FIGSHARE_RES_IGG = os.path.join(
    _FIGSHARE_ROOT, "02-Results", "IgG1_Human"
)
FIGSHARE_RES_HER = os.path.join(
    _FIGSHARE_ROOT, "02-Results", "Herceptin"
)
PXD023419_RAW = os.path.join(
    _FIGSHARE_ROOT, "01-RawData", "PXD023419_anti-FLAG-M2_antibody", "RAW"
)
PXD023419 = os.path.join(MABS_BENCHMARK_DIR, "PXD023419")
NT_ROOT = os.path.join(MABS_BENCHMARK_DIR, "nontryp")
NT_WORK_ROOT = os.path.join(WORK_DIR, "mabs", "nontryp")
NT_RESULTS_ROOT = os.path.join(MABS_RESULTS_DIR, "nontryp")


@dataclass
class EnzymeSpec:
    """MSFragger enzyme params (subset of fragpipe.workflow keys)."""
    name: str            # msfragger.search_enzyme_name_1
    cut: str             # msfragger.search_enzyme_cut_1
    sense: str           # 'C' or 'N'
    nocut: str = ""      # msfragger.search_enzyme_nocut_1
    missed_cleavage: int = 2


# Per-protease enzyme params (MSFragger built-in names).
ENZYMES = {
    "aspn":        EnzymeSpec("aspn",        "D",    "N", missed_cleavage=2),
    "chymo":       EnzymeSpec("chymotrypsin","FYWL", "C", nocut="P", missed_cleavage=2),
    "gluc":        EnzymeSpec("gluc",        "DE",   "C", missed_cleavage=2),
    "lysc":        EnzymeSpec("lysc",        "K",    "C", missed_cleavage=2),
    "lysn":        EnzymeSpec("lysn",        "K",    "N", missed_cleavage=2),
    "proteinasek": EnzymeSpec("nonspecific", "-",    "C", missed_cleavage=2),
    "elastase":    EnzymeSpec("elastase",    "AVST", "C", nocut="P", missed_cleavage=3),
    "thermolysin": EnzymeSpec("thermolysin", "FILMV","N", nocut="P", missed_cleavage=3),
}


@dataclass
class NonTrypRun:
    mab_id: str                  # IgG1_Human_H | IgG1_Human_L | Herceptin
    protease: str                # key into ENZYMES
    short_protease: str          # display label (e.g. 'AspN')
    sample: str                  # basename used everywhere (no extension)
    src_mgf: str                 # original MGF (centroided)
    fasta_td: str                # target+decoy fasta for MSFragger
    short_mab: str               # e.g. 'IgG1-Human Heavy'
    note: str = ""
    # Peer-tool overlays (per-protease versions):
    summary_csv: Optional[str] = None             # Beslic per-protease summary.csv
    summary_scan_remap_mgf: Optional[str] = None  # MGF to remap 0-indexed -> Thermo scans
    supernovo: Optional[tuple] = None             # (csv_path, raw_substring) - Herceptin only
    # Optional PEAKS overlays (not commonly available per-protease in figshare):
    peaks_db: Optional[tuple] = None
    peaks_denovo: Optional[tuple] = None
    peaks_spider: Optional[tuple] = None

    @property
    def workdir(self) -> str:
        return os.path.join(
            NT_WORK_ROOT, self.mab_id, self.protease, "fragpipe_workdir"
        )

    @property
    def res_dir(self) -> str:
        return os.path.join(
            NT_RESULTS_ROOT, self.mab_id, self.protease, "casanovo_results"
        )

    @property
    def prepared_mgf(self) -> str:
        """FragPipe-friendly MGF (TITLE = <sample>.<scan>.<scan>.<charge>, SCANS=<scan>)."""
        return f"{NT_ROOT}/{self.mab_id}/{self.protease}/{self.sample}.mgf"

    @property
    def annotated_mgf(self) -> str:
        return f"{NT_ROOT}/{self.mab_id}/{self.protease}/{self.sample}_annotated.mgf"

    @property
    def enzyme(self) -> EnzymeSpec:
        return ENZYMES[self.protease]


IGG_H_FASTA = os.path.join(MABS_REFERENCES_DIR, "Human_HL_td.fasta")
IGG_L_FASTA = IGG_H_FASTA
HER_FASTA = os.path.join(MABS_REFERENCES_DIR, "trastuzumab_td.fasta")
WIGG_FASTA = os.path.join(MABS_REFERENCES_DIR, "WIgG1_HL_td.fasta")
FLAG_FASTA = os.path.join(MABS_REFERENCES_DIR, "antiFLAGM2_td.fasta")

FIGSHARE_RES_WIGG = os.path.join(
    _FIGSHARE_ROOT, "02-Results", "IgG1_Waters_Mouse"
)


def _igg_run(mab_id, fasta, short_mab, hc_or_lc, protease, sample_stem, short_prot,
             figshare_subdir):
    """One IgG1 H or L non-tryptic Run."""
    return NonTrypRun(
        mab_id=mab_id,
        protease=protease,
        short_protease=short_prot,
        sample=sample_stem,
        src_mgf=f"{FIGSHARE_BES}/{sample_stem}.mgf",
        fasta_td=fasta,
        short_mab=short_mab,
        summary_csv=f"{FIGSHARE_RES_IGG}/{hc_or_lc}/{figshare_subdir}/summary.csv",
    )


def _flag_run(protease, sample_stem, short_prot):
    """anti-FLAG-M2 (PXD023419) non-tryptic Run. Same RAW dir as Herceptin;
    Supernovo per-mAb CSV is separate. Mouse mAb → +PP uses germline_mouse pepLM."""
    return NonTrypRun(
        mab_id="anti-FLAG-M2",
        protease=protease,
        short_protease=short_prot,
        sample=sample_stem,
        src_mgf=f"{PXD023419_RAW}/{sample_stem}.mgf",
        fasta_td=FLAG_FASTA,
        short_mab="anti-FLAG-M2",
        supernovo=(
            f"{PXD023419_RAW}/Peng2021_anti-FLAG-M2_SupernovoPeptideOverview.csv",
            f"_{protease}.raw",
        ),
    )


def _wigg_run(chain, protease, sample_stem, short_prot, figshare_subdir):
    """WIgG1 (MSV000079801, mouse) non-tryptic Run. HC/LC handled as separate
    mab_ids (WIgG1_H / WIgG1_L) to mirror the IgG1_Human_H/_L split."""
    mab_id  = f"WIgG1_{chain}"
    hc_or_lc = "HC" if chain == "H" else "LC"
    return NonTrypRun(
        mab_id=mab_id,
        protease=protease,
        short_protease=short_prot,
        sample=sample_stem,
        src_mgf=f"{FIGSHARE_BES}/{sample_stem}.mgf",
        fasta_td=WIGG_FASTA,
        short_mab=f"WIgG1 {'Heavy' if chain=='H' else 'Light'}",
        summary_csv=f"{FIGSHARE_RES_WIGG}/{hc_or_lc}/{figshare_subdir}/summary.csv",
    )


def _her_run(protease, sample_stem, short_prot):
    # Beslic's Herceptin per-protease results dirs use mixed case (lysC, gluC,
    # lysN, aspN) while our registry keys are all lowercase. Map at the boundary.
    her_dir = {
        "lysc": "lysC", "gluc": "gluC", "lysn": "lysN",
        "chymo": "chymo", "elastase": "elastase", "thermolysin": "thermolysin",
    }[protease]
    return NonTrypRun(
        mab_id="Herceptin",
        protease=protease,
        short_protease=short_prot,
        sample=sample_stem,
        src_mgf=f"{PXD023419_RAW}/{sample_stem}.mgf",
        fasta_td=HER_FASTA,
        short_mab="Herceptin",
        summary_csv=f"{FIGSHARE_RES_HER}/{her_dir}/summary.csv",
        supernovo=(
            f"{PXD023419_RAW}/Peng2021_Herceptin_SupernovoPeptideOverview.csv",
            f"_{protease}.raw",
        ),
    )


# Per-mAb non-tryptic availability (matches what's actually on disk):
#   IgG1_H: AspN, Chymotrypsin, GluC, LysC, ProteinaseK
#   IgG1_L: AspN, Chymotrypsin,        LysC, ProteinaseK   (no GluC)
#   Herceptin: elastase, gluC, lysC, lysN, thermolysin
RUNS = [
    # ----- IgG1 Human Heavy -----
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "IgG1-Human Heavy", "HC",
             "aspn",        "Heavy-Chain-Asp-N-1",        "AspN",        "HC_AspN"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "IgG1-Human Heavy", "HC",
             "chymo",       "Heavy-Chain-Chymotrypsin-1", "Chymotrypsin","HC_Chymotrypsin"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "IgG1-Human Heavy", "HC",
             "gluc",        "Heavy-Chain-Glu-C-1",        "GluC",        "HC_GluC"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "IgG1-Human Heavy", "HC",
             "lysc",        "Heavy-Chain-Lys-C-1",        "LysC",        "HC_LysC"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "IgG1-Human Heavy", "HC",
             "proteinasek", "Heavy-Chain-Proteinase-K-1", "ProteinaseK", "HC_ProteinaseK"),

    # ----- IgG1 Human Light (no GluC on disk) -----
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "IgG1-Human Light", "LC",
             "aspn",        "Light-Chain-Asp-N-1",        "AspN",        "LC_AspN"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "IgG1-Human Light", "LC",
             "chymo",       "Light-Chain-Chymotrypsin-1", "Chymotrypsin","LC_Chymotrypsin"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "IgG1-Human Light", "LC",
             "lysc",        "Light-Chain-Lys-C-1",        "LysC",        "LC_LysC"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "IgG1-Human Light", "LC",
             "proteinasek", "Light-Chain-Proteinase-K-1", "ProteinaseK", "LC_ProteinaseK"),

    # ----- IgG1_Human_L (fill in missing GluC — Light-Chain-Glu-C-1.mgf exists) -----
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "IgG1-Human Light", "LC",
             "gluc",        "Light-Chain-Glu-C-1",        "GluC",        "LC_GluC"),

    # ----- Herceptin (PXD023419) -----
    _her_run("elastase",    "Peng2021_Herceptin_elastase",    "Elastase"),
    _her_run("gluc",        "Peng2021_Herceptin_gluC",        "GluC"),
    _her_run("lysc",        "Peng2021_Herceptin_lysC",        "LysC"),
    _her_run("lysn",        "Peng2021_Herceptin_lysN",        "LysN"),
    _her_run("thermolysin", "Peng2021_Herceptin_thermolysin", "Thermolysin"),

    # ----- anti-FLAG-M2 (PXD023419, mouse) -----
    _flag_run("aspn",        "Peng2021_anti-FLAG-M2_aspN",        "AspN"),
    _flag_run("chymo",       "Peng2021_anti-FLAG-M2_chymo",       "Chymotrypsin"),
    _flag_run("elastase",    "Peng2021_anti-FLAG-M2_elastase",    "Elastase"),
    _flag_run("gluc",        "Peng2021_anti-FLAG-M2_gluC",        "GluC"),
    _flag_run("lysc",        "Peng2021_anti-FLAG-M2_lysC",        "LysC"),
    _flag_run("lysn",        "Peng2021_anti-FLAG-M2_lysN",        "LysN"),
    _flag_run("thermolysin", "Peng2021_anti-FLAG-M2_thermolysin", "Thermolysin"),

    # ----- WIgG1 Heavy (MSV000079801, mouse — only 2 non-tryp proteases on disk) -----
    _wigg_run("H", "aspn",  "WIgG1-Heavy-AspN",         "AspN",         "MouseHC_AspN"),
    _wigg_run("H", "chymo", "WIgG1-Heavy-Chymotrypsin", "Chymotrypsin", "MouseHC_Chymotrypsin"),

    # ----- WIgG1 Light (MSV000079801, mouse — same set) -----
    _wigg_run("L", "aspn",  "WIgG1-Light-AspN",         "AspN",         "MouseLC_AspN"),
    _wigg_run("L", "chymo", "WIgG1-Light-Chymotrypsin", "Chymotrypsin", "MouseLC_Chymotrypsin"),
]

# Protease keys grouped for "one figure per protease" plotting.  Display order:
PROTEASE_ORDER = [
    "aspn", "chymo", "gluc", "lysc", "lysn",
    "proteinasek", "elastase", "thermolysin",
]
PROTEASE_LABEL = {
    "aspn": "AspN", "chymo": "Chymotrypsin", "gluc": "GluC", "lysc": "LysC",
    "lysn": "LysN", "proteinasek": "Proteinase K",
    "elastase": "Elastase", "thermolysin": "Thermolysin",
}


def runs_for_protease(protease: str):
    return [r for r in RUNS if r.protease == protease]


def main():
    import os
    print(f"{len(RUNS)} non-tryptic runs registered:\n")
    for i, r in enumerate(RUNS):
        exists = "OK " if os.path.exists(r.src_mgf) else "MISSING"
        print(f"  [{i:2d}] {r.mab_id:14s} {r.protease:12s} -> {exists}  {r.src_mgf}")


if __name__ == "__main__":
    main()
