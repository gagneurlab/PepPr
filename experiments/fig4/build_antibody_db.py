"""Build the clean germline antibody corpus (human + mouse, separate FASTAs)
from the IMGT/GENE-DB bulk AA reference.

Heavy chains are NOT recombined into full V·D·J·C proteins. Instead each HC
V allele contributes one V-REGION record (FR1 -> FR3-end Cys) and each
(J allele x C isotype) pair contributes one J+C record (FR4-start -> end of
the assembled constant region). Heavy-chain CDR3, D-segments, and junctional
N-additions never enter the corpus. Light chains ARE emitted as full V·J·C,
since the LC CDR3 is conserved and safe to train on.

We keep ALL functional alleles per gene (no *01-only filter): IMGT germline
alleles vary by 1-5 AAs across the V-region and several real evaluation mAbs
match a non-*01 allele meaningfully better (e.g. 36H6 HC: IGHV1-18*04 at 95.8%
vs *01 at 89.6%). The corpus stays <2k records per species.

Outputs:
  antibody/antibody_human.fasta
  antibody/antibody_mouse.fasta

Header format matches what the downstream pipeline (in-silico digest + PLM
training-data generator) expects, so it is reused unchanged:
  >IG{H,K,L}V{N}|{isotype}|{record_id}
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peptide_priors.const import FASTAS_DIR, MABS_REFERENCES_DIR

# https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-AA-WithGaps-F%2BORF%2BinframeP
ANTIBODY_DIR = MABS_REFERENCES_DIR
IMGT_PATH = os.path.join(ANTIBODY_DIR, "imgt",
                         "IMGTGENEDB-ReferenceSequences.fasta-AA-WithGaps-F+ORF+inframeP")

# IMGT functionality codes:
#   F      -> Functional (always included)
#   [F]    -> Functional with reservations (often strain-specific; included)
#   (F)    -> Functional with reservations (alt notation; included)
#   ORF    -> Open reading frame, no expression evidence (excluded by default)
#   P, [P] -> Pseudogene (always excluded)
INCLUDE_ORF   = False
FUNCTIONAL_CODES = {"F", "[F]", "(F)"}

OUT_DIR = FASTAS_DIR

# ---------------------------------------------------------------------------
# Species selectors
# ---------------------------------------------------------------------------

def species_tag(sp: str) -> str | None:
    if sp == "Homo sapiens":
        return "human"
    # Accept any mouse strain ("Mus musculus_BALB/c", etc.) but NOT distantly
    # related species ("Mus pahari", "Mus spretus", "Mus saxicola", etc.).
    if sp == "Mus musculus" or sp.startswith("Mus musculus_"):
        return "mouse"
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_imgt(path: str) -> dict:
    """Return {species: {gene_allele: {region: seq}}}."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    cur_hdr = None
    cur_seq_lines: list[str] = []

    def flush():
        if cur_hdr is None:
            return
        parts = cur_hdr.split("|")
        if len(parts) < 5:
            return
        accession, gene_allele, species, func, region = parts[:5]
        tag = species_tag(species)
        if tag is None:
            return
        if func not in FUNCTIONAL_CODES and not (INCLUDE_ORF and func == "ORF"):
            return
        seq = "".join(cur_seq_lines).replace(".", "").replace(" ", "")\
                                    .strip("*").strip("X")
        if not seq:
            return
        out[tag][gene_allele][region] = seq

    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                flush()
                cur_hdr = line[1:].rstrip()
                cur_seq_lines = []
            else:
                cur_seq_lines.append(line.strip())
        flush()
    return out


# ---------------------------------------------------------------------------
# Recombination
# ---------------------------------------------------------------------------

def collect_by_gene_class(records: dict, prefix: str, region: str) -> list[tuple[str, str]]:
    """Return [(gene_allele, seq), ...] for entries whose gene starts with
    `prefix` (e.g. 'IGHV') AND have the requested region. Sorted alphabetically
    by gene_allele for deterministic output."""
    out = []
    for ga, regions in records.items():
        if not ga.startswith(prefix):
            continue
        if region in regions:
            out.append((ga, regions[region]))
    return sorted(out)


def assemble_c_region(records: dict, isotype_gene_allele: str) -> str | None:
    """For an IGHG/A/M/D/E allele, splice the multi-exon constant region into
    a single AA string (CH1 + H + CH2 + CH3-CHS, with variant fallbacks)."""
    if isotype_gene_allele not in records:
        return None
    regions = records[isotype_gene_allele]
    # The exon order varies by isotype. Try a sensible set of layouts.
    layouts = [
        # IGHG: CH1 + H + CH2 + CH3-CHS  (most common)
        ("CH1", "H", "CH2", "CH3-CHS"),
        # IGHA: CH1 + H-CH2 (combined) + CH3-CHS
        ("CH1", "H-CH2", "CH3-CHS"),
        # IGHM: CH1 + CH2 + CH3 + CH4-CHS (no hinge)
        ("CH1", "CH2", "CH3", "CH4-CHS"),
        # IGHE: CH1 + CH2 + CH3 + CH4-CHS (no hinge)
        ("CH1", "CH2", "CH3", "CH4-CHS"),
        # IGHD: CH1 + H1 + H2 + CH2 + CH3 + CHS  (mouse: CH1 + H + CH3 + CHS)
        ("CH1", "H1", "H2", "CH2", "CH3", "CHS"),
        ("CH1", "H", "CH2", "CH3", "CHS"),
        ("CH1", "H", "CH3", "CHS"),
        # Mouse IGHG: sometimes CHS is absent; ensure single-exon fallback
        ("CH1", "H", "CH2", "CH3"),
    ]
    for parts in layouts:
        chunks = [regions.get(p) for p in parts]
        if all(c is not None for c in chunks):
            return "".join(chunks)
    return None


def recombine_heavy(records: dict, c_isotype_genes: list[str]) -> list[tuple[str, str]]:
    """Emit clean HC records: one V-REGION per V allele, and one J+C record
    per (J allele x C isotype). No D-segments, CDR3, or N-additions.
    Returns list of (header_id, sequence)."""
    v_list = collect_by_gene_class(records, "IGHV", "V-REGION")
    j_list = collect_by_gene_class(records, "IGHJ", "J-REGION")

    # Discover C-region alleles matching the requested isotype gene names.
    # IMGT format: IGHG1*01 -> CH1/H/CH2/CH3-CHS exons stored under that allele.
    c_alleles: list[tuple[str, str]] = []
    for ga in records:
        if "*" not in ga:
            continue
        gene = ga.split("*")[0]
        if gene not in c_isotype_genes:
            continue
        c_seq = assemble_c_region(records, ga)
        if c_seq:
            c_alleles.append((ga, c_seq))
    c_alleles.sort()

    # V-REGION (FR1 -> FR3-end Cys) and J+C (FR4-start -> end of constant) as
    # separate records. CDR3, D, and N-additions never enter the corpus.
    print(f"  HC clean build: V={len(v_list)} (V-REGION only)  "
          f"J·C={len(j_list)*len(c_alleles)} (J-REGION + assembled C)")
    out = [(f"{vg}|V-REGION", vs) for vg, vs in v_list]
    for (jg, js), (cg, cs) in product(j_list, c_alleles):
        out.append((f"{jg}|{cg}|J+C", js + cs))
    return out


def recombine_light(records: dict, locus: str) -> list[tuple[str, str]]:
    """Build all V-J-C kappa or lambda combinations."""
    vp = f"IG{locus}V"
    jp = f"IG{locus}J"
    cp = f"IG{locus}C"

    v_list = collect_by_gene_class(records, vp, "V-REGION")
    j_list = collect_by_gene_class(records, jp, "J-REGION")
    c_list = collect_by_gene_class(records, cp, "C-REGION")

    print(f"  {locus}C pool: V={len(v_list)}  J={len(j_list)}  C={len(c_list)}")

    out = []
    for (vg, vs), (jg, js), (cg, cs) in product(v_list, j_list, c_list):
        seq = vs + js + cs
        hid = f"{vg}|{jg}|{cg}"
        out.append((hid, seq))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# All IGHG isotypes per species + IgM as the "default Fc". Skip IgA/D/E for
# now (we can add them; cost is small). Defaults match decision in chat: all
# IGG subclasses + IgM, kappa + lambda LC.
ISOTYPES_HUMAN = ["IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHM"]
ISOTYPES_MOUSE = ["IGHG1", "IGHG2A", "IGHG2B", "IGHG2C", "IGHG3", "IGHM"]


def build_species_fasta(records: dict, out_path: str, species: str,
                        isotype_genes: list[str]) -> None:
    print(f"\n=== {species} ===")
    h_recombs = recombine_heavy(records[species], isotype_genes)
    k_recombs = recombine_light(records[species], "K")
    l_recombs = recombine_light(records[species], "L")
    total = len(h_recombs) + len(k_recombs) + len(l_recombs)
    print(f"  totals: H={len(h_recombs)}  K={len(k_recombs)}  L={len(l_recombs)}  "
          f"all={total}")

    with open(out_path, "w") as f:
        for i, (hid, seq) in enumerate(h_recombs):
            f.write(f">H|{hid}|{i}\n{seq}\n")
        for i, (hid, seq) in enumerate(k_recombs):
            f.write(f">K|{hid}|{i}\n{seq}\n")
        for i, (hid, seq) in enumerate(l_recombs):
            f.write(f">L|{hid}|{i}\n{seq}\n")
    print(f"  wrote {out_path} ({os.path.getsize(out_path)//1024} KB)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Parsing IMGT: {IMGT_PATH}")
    records = parse_imgt(IMGT_PATH)
    for sp in ("human", "mouse"):
        print(f"  {sp}: {sum(len(r) for r in records[sp].values())} gene-region entries "
              f"across {len(records[sp])} alleles")

    build_species_fasta(records,
                        os.path.join(OUT_DIR, "antibody_human.fasta"),
                        "human", ISOTYPES_HUMAN)
    build_species_fasta(records,
                        os.path.join(OUT_DIR, "antibody_mouse.fasta"),
                        "mouse", ISOTYPES_MOUSE)


if __name__ == "__main__":
    main()
