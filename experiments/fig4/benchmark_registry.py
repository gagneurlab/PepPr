"""Registry of (mAb, non-tryptic-protease) benchmark runs.

Each Run describes one protease's FragPipe + Casanovo end-to-end pipeline:
  - source MGF (centroided) holding that protease's MS2 spectra
  - per-mAb target+decoy FASTA
  - MSFragger enzyme params for the protease

The registry drives the antibody pipeline in experiments/fig4/benchmark_prep.py
(prepare-nontryp / annotate) and experiments/fig4/run_mab.sh (fragpipe / casanovo
/ submit).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from experiments.paths import (
    MABS_BENCHMARK_DIR,
    MABS_REFERENCES_DIR,
    MABS_RESULTS_DIR,
    WORK_DIR,
)

_FIGSHARE_ROOT = os.path.join(MABS_BENCHMARK_DIR, "figshare_21394143")
# Centroided MGFs for the MSV000079801 mAbs (IgG1_Human, WIgG1).
FIGSHARE_BES = os.path.join(
    _FIGSHARE_ROOT, "01-RawData", "MSV000079801_CompleteAssembly", "Centroided"
)
# RAW-derived MGFs for the PXD023419 mAbs (Herceptin, anti-FLAG-M2).
PXD023419_RAW = os.path.join(
    _FIGSHARE_ROOT, "01-RawData", "PXD023419_anti-FLAG-M2_antibody", "RAW"
)
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
    mab_id: str                  # IgG1_Human_H | IgG1_Human_L | Herceptin | ...
    protease: str                # key into ENZYMES
    sample: str                  # basename used everywhere (no extension)
    src_mgf: str                 # original MGF (centroided / RAW-derived)
    fasta_td: str                # target+decoy fasta for MSFragger

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


def _igg_run(mab_id, fasta, protease, sample_stem):
    """One IgG1 H or L non-tryptic Run (MSV000079801 centroided MGF)."""
    return NonTrypRun(
        mab_id=mab_id,
        protease=protease,
        sample=sample_stem,
        src_mgf=f"{FIGSHARE_BES}/{sample_stem}.mgf",
        fasta_td=fasta,
    )


def _flag_run(protease, sample_stem):
    """anti-FLAG-M2 (PXD023419, mouse) non-tryptic Run.  Mouse mAb -> +PP uses
    the germline_mouse pepLM."""
    return NonTrypRun(
        mab_id="anti-FLAG-M2",
        protease=protease,
        sample=sample_stem,
        src_mgf=f"{PXD023419_RAW}/{sample_stem}.mgf",
        fasta_td=FLAG_FASTA,
    )


def _wigg_run(chain, protease, sample_stem):
    """WIgG1 (MSV000079801, mouse) non-tryptic Run.  HC/LC are separate mab_ids
    (WIgG1_H / WIgG1_L), mirroring the IgG1_Human_H/_L split."""
    return NonTrypRun(
        mab_id=f"WIgG1_{chain}",
        protease=protease,
        sample=sample_stem,
        src_mgf=f"{FIGSHARE_BES}/{sample_stem}.mgf",
        fasta_td=WIGG_FASTA,
    )


def _her_run(protease, sample_stem):
    """Herceptin (PXD023419) non-tryptic Run."""
    return NonTrypRun(
        mab_id="Herceptin",
        protease=protease,
        sample=sample_stem,
        src_mgf=f"{PXD023419_RAW}/{sample_stem}.mgf",
        fasta_td=HER_FASTA,
    )


# Per-mAb non-tryptic availability (matches what's actually on disk):
#   IgG1_H: AspN, Chymotrypsin, GluC, LysC, ProteinaseK
#   IgG1_L: AspN, Chymotrypsin, GluC, LysC, ProteinaseK
#   Herceptin: elastase, gluC, lysC, lysN, thermolysin
#   anti-FLAG-M2: aspN, chymo, elastase, gluC, lysC, lysN, thermolysin
#   WIgG1 H/L: AspN, Chymotrypsin
RUNS = [
    # ----- IgG1 Human Heavy -----
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "aspn",        "Heavy-Chain-Asp-N-1"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "chymo",       "Heavy-Chain-Chymotrypsin-1"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "gluc",        "Heavy-Chain-Glu-C-1"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "lysc",        "Heavy-Chain-Lys-C-1"),
    _igg_run("IgG1_Human_H", IGG_H_FASTA, "proteinasek", "Heavy-Chain-Proteinase-K-1"),

    # ----- IgG1 Human Light -----
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "aspn",        "Light-Chain-Asp-N-1"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "chymo",       "Light-Chain-Chymotrypsin-1"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "lysc",        "Light-Chain-Lys-C-1"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "proteinasek", "Light-Chain-Proteinase-K-1"),
    _igg_run("IgG1_Human_L", IGG_L_FASTA, "gluc",        "Light-Chain-Glu-C-1"),

    # ----- Herceptin (PXD023419) -----
    _her_run("elastase",    "Peng2021_Herceptin_elastase"),
    _her_run("gluc",        "Peng2021_Herceptin_gluC"),
    _her_run("lysc",        "Peng2021_Herceptin_lysC"),
    _her_run("lysn",        "Peng2021_Herceptin_lysN"),
    _her_run("thermolysin", "Peng2021_Herceptin_thermolysin"),

    # ----- anti-FLAG-M2 (PXD023419, mouse) -----
    _flag_run("aspn",        "Peng2021_anti-FLAG-M2_aspN"),
    _flag_run("chymo",       "Peng2021_anti-FLAG-M2_chymo"),
    _flag_run("elastase",    "Peng2021_anti-FLAG-M2_elastase"),
    _flag_run("gluc",        "Peng2021_anti-FLAG-M2_gluC"),
    _flag_run("lysc",        "Peng2021_anti-FLAG-M2_lysC"),
    _flag_run("lysn",        "Peng2021_anti-FLAG-M2_lysN"),
    _flag_run("thermolysin", "Peng2021_anti-FLAG-M2_thermolysin"),

    # ----- WIgG1 Heavy / Light (MSV000079801, mouse — 2 non-tryp proteases on disk) -----
    _wigg_run("H", "aspn",  "WIgG1-Heavy-AspN"),
    _wigg_run("H", "chymo", "WIgG1-Heavy-Chymotrypsin"),
    _wigg_run("L", "aspn",  "WIgG1-Light-AspN"),
    _wigg_run("L", "chymo", "WIgG1-Light-Chymotrypsin"),
]


def main():
    print(f"{len(RUNS)} non-tryptic runs registered:\n")
    for i, r in enumerate(RUNS):
        exists = "OK " if os.path.exists(r.src_mgf) else "MISSING"
        print(f"  [{i:2d}] {r.mab_id:14s} {r.protease:12s} -> {exists}  {r.src_mgf}")


if __name__ == "__main__":
    main()
