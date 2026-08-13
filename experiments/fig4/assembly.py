#!/usr/bin/env python3
"""Single-mAb de novo assembly comparison — Figure 4, Panel E.

Two arms per mAb:
  baseline  — Casanovo v5, no pepLM ('vanilla' for the XA-Novo mAbs, 'noplm'
              for the Beslic mAbs)
  +PP       — Casanovo + the species-matched germline pepLM. Located by the
              mzTab's inference-run token (`MAB_SPECS[mab]['pp_arm']`, i.e.
              germline_<sp>_sw for XA-Novo, antibody_<sp> for Beslic) but
              labeled uniformly `antibody_<species>` in the output, matching
              plot_figure_4._ASSEMBLY_MABS.

For each (mAb, arm):
  1. Gather the per-protease Casanovo mzTabs (references.find_mztab).
  2. ALPS-assemble (Casanovo mass-consistency filter only) for k in 7..11.
  3. Union the contigs across k, dedup on I/L-folded sequence.
  4. npysearch each chain of the mAb reference; report coverage / accuracy,
     the longest aligned contig, NG50, and CDR-restricted accuracy / coverage.

Writes the two TSVs plot_figure_4 Panel E reads
(const.FIGURE_4_ASSEMBLY_TSV_PATHS): the XA-Novo mAbs to the `solo` summary,
the Beslic mAbs to the `beslic` summary. The mAb set, per-mAb references, and
protease lists all come from the canonical registry in references.MAB_SPECS.

ALPS is third-party and not redistributed: obtain ALPS.jar and set
DNPS_ALPS_JAR to it (or place it at the repo root). npysearch supplies the
PowerNovo-style contig↔reference alignment metrics.

Run: python experiments/fig4/assembly.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass

import npysearch as npy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peptide_priors import const
from experiments.fig4.references import (
    MAB_SPECS, find_mztab, read_fasta_with_regions, il_fold,
)

# ALPS is third-party and not redistributed with this repo. Obtain ALPS.jar and
# either set DNPS_ALPS_JAR to its path or place it at <repo>/ALPS.jar.
ALPS_JAR = os.environ.get(
    "DNPS_ALPS_JAR", os.path.join(const.PROJECT_ROOT, "ALPS.jar"))

# Union of ALPS contigs across these k (matches the shipped `..._k7to11.tsv`).
K_MERS = [7, 8, 9, 10, 11]
TOP_CONTIGS = 20
OUT_DIR = os.path.join(const.WORK_DIR, "mabs", "alps")  # scratch for CSVs/contigs

# Strip ProForma mods from an mzTab peptide: bracket mods (N[Deamidated]),
# Casanovo-v3 paren mods (C(+57.02)), and bare mass shifts (M+15.99).
_MOD_BRACKET = re.compile(r"\[[^\]]*\]")
_MOD_PAREN = re.compile(r"\([^)]*\)")
_MASS_SHIFT = re.compile(r"[+-]?\d+\.\d+")


def strip_mods(seq: str) -> str:
    seq = _MOD_BRACKET.sub("", seq)
    seq = _MOD_PAREN.sub("", seq)
    return _MASS_SHIFT.sub("", seq)


@dataclass
class PSMRow:
    spectrum: str
    peptide: str
    aa_scores: str   # space-separated per-residue scores
    score: float


def parse_mztab_for_alps(path: str) -> list[PSMRow]:
    """Load ALPS-ready PSM rows from one Casanovo mzTab."""
    rows: list[PSMRow] = []
    header = None
    sample_id = os.path.basename(path).removesuffix(".mztab")
    with open(path) as handle:
        for line in handle:
            if line.startswith("PSH\t"):
                header = line.rstrip("\n").split("\t")
                continue
            if not line.startswith("PSM\t") or header is None:
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            try:
                score = float(row["search_engine_score[1]"])
            except (KeyError, ValueError):
                continue
            peptide = strip_mods(
                row.get("opt_ms_run[1]_proforma") or row.get("sequence", "")
            )
            if not peptide or not peptide.isalpha():
                continue
            aa_values = [v for v in row.get("opt_ms_run[1]_aa_scores", "").split(",") if v]
            if len(aa_values) != len(peptide):
                # Length mismatch: ALPS would reject the row; skip rather than
                # pad with zeros and poison the de Bruijn graph.
                continue
            match = re.search(r"index=(\d+)", row.get("spectra_ref", ""))
            index = match.group(1) if match else str(len(rows))
            rows.append(PSMRow(f"{sample_id}_idx{index}", peptide,
                               " ".join(aa_values), score))
    return rows


def write_alps_csv(psms: list[PSMRow], out_csv: str) -> int:
    """Write ALPS-format PSMs (Spectrum Name, Peptide, aaScore, Score, Area).

    Drops only the mass-inconsistent PSMs (Casanovo score < 0); no confidence
    cutoff is applied — ALPS' soft per-AA weighting handles quality, and a
    score cutoff has been ruled out as a driver of the assembly outcome.
    """
    n = 0
    with open(out_csv, "w") as f:
        f.write("Spectrum Name,Casanovo Peptide,Casanovo aaScore,Casanovo Score,Area\n")
        for p in psms:
            if p.score < 0:
                continue
            f.write(f"{p.spectrum},{p.peptide},{p.aa_scores},{p.score},1\n")
            n += 1
    return n


def run_alps(csv_path: str, k: int, c: int = TOP_CONTIGS) -> str:
    """Run the ALPS jar; return the generated FASTA path. Reuses a cached per-k
    FASTA when it is newer than the input CSV."""
    fasta = f"{csv_path}.k{k}.fasta"
    if os.path.exists(fasta) and os.path.getmtime(fasta) >= os.path.getmtime(csv_path):
        print(f"  [skip] cached k={k}: {os.path.basename(fasta)}")
        return fasta
    cmd = ["java", "-jar", ALPS_JAR, csv_path, str(k), str(c)]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=os.path.dirname(csv_path),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("ALPS STDOUT:\n" + proc.stdout)
        print("ALPS STDERR:\n" + proc.stderr)
        raise RuntimeError(f"ALPS failed (exit {proc.returncode})")
    if not os.path.exists(fasta):
        fasta = os.path.join(os.path.dirname(csv_path),
                             os.path.basename(csv_path) + f".k{k}.fasta")
    return fasta


def read_fasta(path: str) -> list[tuple[str, str]]:
    out, name, seq = [], None, []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(seq)))
                name, seq = line[1:], []
            else:
                seq.append(line.strip())
        if name is not None:
            out.append((name, "".join(seq)))
    return out


def union_contigs(per_k: dict[int, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Union contigs across k values; dedup identical (I/L-folded) sequences.
    The first occurrence wins for the display name (tagged with its k)."""
    seen: dict[str, tuple[str, str]] = {}
    for k in sorted(per_k):
        for name, seq in per_k[k]:
            key = il_fold(seq)
            if key in seen:
                continue
            seen[key] = (f"k{k}:{name}", seq)
    return list(seen.values())


def npysearch_metrics(contigs: list[tuple[str, str]], ref: str,
                      min_identity: float = 0.75, max_accepts: int = 5) -> dict:
    """PowerNovo-style metrics via npysearch (BLAST-like protein alignment).
    I→L folded on both sides for mass-spec equivalence (Beslic convention).
    minIdentity=0.75, maxAccepts=5 match PowerNovo's peptide_aggregator defaults."""
    empty = {"mapped": [], "n_mapped": 0, "coverage": 0.0,
             "covered_positions": 0, "accuracy": 0.0,
             "total_matches": 0, "total_aligned": 0}
    if not contigs or not ref:
        return empty
    ref_il = il_fold(ref)
    queries = {f"c{i}": il_fold(seq) for i, (_n, seq) in enumerate(contigs) if seq.strip()}
    if not queries:
        return empty
    res = npy.blast(query=queries, database={"ref": ref_il},
                    minIdentity=min_identity, maxAccepts=max_accepts, alphabet="protein")
    n_hits = len(res.get("QueryId", []))
    covered = [False] * len(ref)
    total_matches = total_aligned = 0
    by_contig: dict[int, dict] = {}
    for i in range(n_hits):
        qid = res["QueryId"][i]
        try:
            ci = int(qid[1:])   # 'c<idx>'
        except ValueError:
            continue
        t_start = int(res["TargetMatchStart"][i]) - 1   # npysearch 1-based incl.
        t_end = int(res["TargetMatchEnd"][i])           # → 0-based excl.
        for k in range(max(0, t_start), min(t_end, len(covered))):
            covered[k] = True
        total_matches += int(res["NumMatches"][i])
        total_aligned += int(res["NumColumns"][i])
        ident = float(res["Identity"][i])
        if ci not in by_contig or ident > by_contig[ci]["identity"]:   # best HSP per contig
            by_contig[ci] = {
                "matches": int(res["NumMatches"][i]),
                "aligned_len": int(res["NumColumns"][i]),
                "identity": ident,
                "ref_start": t_start,
                "ref_end": t_end,
                "q_start": int(res["QueryMatchStart"][i]) - 1,
                "q_end": int(res["QueryMatchEnd"][i]),
                "query_match": res["QueryMatchSeq"][i],
                "target_match": res["TargetMatchSeq"][i],
            }
    mapped = [(contigs[ci][0], contigs[ci][1], hit) for ci, hit in sorted(by_contig.items())]
    return {
        "mapped": mapped, "n_mapped": len(mapped),
        "coverage": sum(covered) / len(ref) if ref else 0.0,
        "covered_positions": sum(covered),
        "accuracy": (total_matches / total_aligned) if total_aligned else 0.0,
        "total_matches": total_matches, "total_aligned": total_aligned,
    }


def cdr_metrics(npy_result: dict, cdrs: list[tuple[int, int]] | None) -> dict:
    """Restrict npysearch metrics to CDR positions. `cdrs` is a list of 1-based
    inclusive (start, end) ranges. Walks each mapped contig's aligned columns
    (`query_match`/`target_match`, BLAST-style with `-` gaps), counting matches
    and aligned columns whose *reference* position falls in a CDR. Returns
    per-CDR + combined accuracy/coverage; zeros if `cdrs` is None or unmapped."""
    empty = {"matches": 0, "aligned": 0, "accuracy": 0.0,
             "covered_positions": 0, "total_positions": 0, "coverage": 0.0}
    per_cdr = [dict(empty) for _ in range(3)]
    combined = dict(empty)
    if not cdrs or not npy_result.get("mapped"):
        return {"per_cdr": per_cdr, "combined": combined}

    cdr_pos = [set(range(s - 1, e)) for s, e in cdrs]   # 1-based incl → 0-based half-open
    all_cdr = set().union(*cdr_pos)
    for i, s in enumerate(cdr_pos):
        per_cdr[i]["total_positions"] = len(s)
    combined["total_positions"] = len(all_cdr)

    covered_by_cdr = [set() for _ in range(3)]
    for _name, _seq, hit in npy_result["mapped"]:
        q, t = hit["query_match"], hit["target_match"]
        ref_pos = hit["ref_start"]
        for qch, tch in zip(q, t):
            if tch == "-":
                continue
            for i, s in enumerate(cdr_pos):
                if ref_pos in s:
                    per_cdr[i]["aligned"] += 1
                    if qch == tch:
                        per_cdr[i]["matches"] += 1
                        covered_by_cdr[i].add(ref_pos)
            if ref_pos in all_cdr:
                combined["aligned"] += 1
                if qch == tch:
                    combined["matches"] += 1
            ref_pos += 1

    for i in range(3):
        per_cdr[i]["covered_positions"] = len(covered_by_cdr[i])
        per_cdr[i]["accuracy"] = (per_cdr[i]["matches"] / per_cdr[i]["aligned"]
                                  if per_cdr[i]["aligned"] else 0.0)
        per_cdr[i]["coverage"] = (per_cdr[i]["covered_positions"] / per_cdr[i]["total_positions"]
                                  if per_cdr[i]["total_positions"] else 0.0)
    combined["covered_positions"] = sum(len(c) for c in covered_by_cdr)
    combined["accuracy"] = (combined["matches"] / combined["aligned"]
                            if combined["aligned"] else 0.0)
    combined["coverage"] = (combined["covered_positions"] / combined["total_positions"]
                            if combined["total_positions"] else 0.0)
    return {"per_cdr": per_cdr, "combined": combined}


def ng50(lengths: list[int], genome_size: int) -> int:
    """NG50: largest L such that contigs of length ≥ L cumulatively cover at
    least half the reference chain length. 0 if the assembly never reaches G/2."""
    if not lengths or genome_size <= 0:
        return 0
    cumul = 0
    for L in sorted(lengths, reverse=True):
        cumul += L
        if cumul >= genome_size / 2.0:
            return L
    return 0


# npysearch occasionally fuses HC + LC k-mer fragments into a chimeric contig
# when both chains contribute PSMs to the same union. Chimera HSPs land at
# 80–92% identity, real single-chain assemblies at ≥95%; the stricter CDR
# threshold keeps chimera HSPs spanning a CDR from dragging CDR accuracy down.
NPY_MIN_IDENTITY_COVERAGE = 0.95
NPY_MIN_IDENTITY_CDR = 0.95


def summarize(contigs: list[tuple[str, str]], chain_name: str, chain_seq: str,
              cdrs: list[tuple[int, int]] | None) -> dict:
    """npysearch a contig union against one reference chain; return a flat row of
    coverage/accuracy, longest aligned contig, NG50, and CDR-restricted metrics."""
    pn = npysearch_metrics(contigs, chain_seq,
                           min_identity=NPY_MIN_IDENTITY_COVERAGE, max_accepts=5)
    longest_len = longest_mapped = longest_matches = 0
    if pn["mapped"]:
        # Pick the contig with the most correctly-matched residues against THIS
        # chain (matches = aligned_len × identity), not raw length — otherwise a
        # cross-chain chimera passing 0.75 identity can beat the real chain.
        longest = max(pn["mapped"], key=lambda t: t[2]["matches"])
        longest_len = len(longest[1])
        longest_mapped = longest[2]["aligned_len"]
        longest_matches = longest[2]["matches"]

    pn_cdr = npysearch_metrics(contigs, chain_seq,
                               min_identity=NPY_MIN_IDENTITY_CDR, max_accepts=5)
    cdr = cdr_metrics(pn_cdr, cdrs)
    pc, cb = cdr["per_cdr"], cdr["combined"]
    return {
        "chain": chain_name, "ref_len": len(chain_seq),
        "n_contigs": len(contigs), "n_mapped": pn["n_mapped"],
        "coverage": pn["coverage"], "covered_positions": pn["covered_positions"],
        "accuracy": pn["accuracy"], "matches": pn["total_matches"],
        "aligned": pn["total_aligned"],
        "longest_contig": longest_len, "longest_aligned": longest_mapped,
        "longest_matches": longest_matches,
        "ng50": ng50([len(seq) for _n, seq in contigs], len(chain_seq)),
        "cdr_total": cb["total_positions"], "cdr_covered": cb["covered_positions"],
        "cdr_coverage": cb["coverage"], "cdr_matches": cb["matches"],
        "cdr_aligned": cb["aligned"], "cdr_accuracy": cb["accuracy"],
        "cdr1_acc": pc[0]["accuracy"], "cdr2_acc": pc[1]["accuracy"],
        "cdr3_acc": pc[2]["accuracy"],
        "cdr1_cov": pc[0]["coverage"], "cdr2_cov": pc[1]["coverage"],
        "cdr3_cov": pc[2]["coverage"],
    }


# ── Per-mAb arms (from the canonical registry) ───────────────────────────────
def _pp_label(spec: dict) -> str:
    """Output arm label for the +PP arm (matches plot_figure_4._ASSEMBLY_MABS):
    the pepLM model name, `antibody_<species>`, independent of the inference-run
    file token used to locate the mzTabs."""
    return "antibody_human" if "human" in spec["species"] else "antibody_mouse"


def arms_for(mab: str) -> list[tuple[str, str]]:
    """(file_token, output_label) for the baseline and +PP arms of one mAb.
    file_token is what find_mztab globs for; output_label is written to the TSV."""
    spec = MAB_SPECS[mab]
    baseline = spec.get("vanilla_arm", "vanilla")   # 'vanilla' (XA-Novo) / 'noplm' (Beslic)
    return [(baseline, baseline), (spec["pp_arm"], _pp_label(spec))]


def assemble_arm(mab: str, file_arm: str) -> list[tuple[str, str]]:
    """ALPS-assemble one (mab, arm) across the mAb's proteases; return the
    across-k union of contigs. Empty list if no mzTabs / PSMs are found."""
    psms = []
    for protease in MAB_SPECS[mab]["proteases"]:
        path = find_mztab(mab, protease, file_arm)
        if path is None:
            print(f"  [{mab}/{file_arm}/{protease}] MISSING mztab", file=sys.stderr)
            continue
        got = parse_mztab_for_alps(path)
        print(f"  [{mab}/{file_arm}/{protease}] {os.path.basename(path)}: {len(got)} PSMs")
        psms.extend(got)
    if not psms:
        return []

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{mab}_{file_arm}.csv")
    n = write_alps_csv(psms, csv_path)
    print(f"  -> {n} PSMs -> {os.path.basename(csv_path)}")
    if n == 0:
        return []
    per_k = {}
    for k in K_MERS:
        per_k[k] = read_fasta(run_alps(csv_path, k=k, c=TOP_CONTIGS))
        print(f"    k={k}: {len(per_k[k])} contigs")
    return union_contigs(per_k)


TSV_COLS = ["mab", "arm", "chain", "ref_len", "n_contigs", "n_mapped",
            "coverage", "covered_positions", "accuracy", "matches", "aligned",
            "longest_contig", "longest_aligned", "longest_matches", "ng50",
            "cdr_total", "cdr_covered", "cdr_coverage", "cdr_matches",
            "cdr_aligned", "cdr_accuracy", "cdr1_acc", "cdr2_acc", "cdr3_acc",
            "cdr1_cov", "cdr2_cov", "cdr3_cov"]


def _out_path(source: str) -> str:
    """Pick the const-defined TSV for a mAb source ('xa_novo' → solo summary,
    'beslic' → beslic summary)."""
    tag = "solo" if source == "xa_novo" else "beslic"
    for path in const.FIGURE_4_ASSEMBLY_TSV_PATHS:
        if tag in os.path.basename(path):
            return path
    raise KeyError(f"no FIGURE_4_ASSEMBLY_TSV_PATHS entry for {tag}")


def _write_tsv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(TSV_COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in TSV_COLS) + "\n")
    print(f"Saved {len(rows)} rows -> {path}")


def main(mabs: list[str] | None = None) -> None:
    by_source: dict[str, list[dict]] = {"xa_novo": [], "beslic": []}
    for mab in (mabs or list(MAB_SPECS)):
        spec = MAB_SPECS[mab]
        ref = read_fasta_with_regions(spec["ref"])
        print(f"\n========= {mab} ({spec['species']}) =========")
        for file_arm, label in arms_for(mab):
            print(f"\n--- {mab} / {label} (mzTab token '{file_arm}') ---")
            union = assemble_arm(mab, file_arm)
            if not union:
                print(f"  no contigs for {mab}/{label}; skipping")
                continue
            print(f"  union(k={','.join(map(str, K_MERS))}): {len(union)} unique contigs")
            for chain_name, info in ref.items():
                row = summarize(union, chain_name, info["seq"], info["cdrs"])
                row["mab"], row["arm"] = mab, label
                by_source[spec["source"]].append(row)
    for source, rows in by_source.items():
        if rows:
            _write_tsv(_out_path(source), rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mabs", nargs="*", default=None,
                    help="Subset of mAb IDs to assemble (default: all in MAB_SPECS).")
    ap.add_argument("--k", type=int, nargs="*", default=None,
                    help="k-mer size(s) to assemble/union (default: 7 8 9 10 11).")
    args = ap.parse_args()
    if args.k:
        K_MERS[:] = args.k
    if args.mabs:
        unknown = [m for m in args.mabs if m not in MAB_SPECS]
        if unknown:
            raise SystemExit(f"Unknown mAb(s): {unknown}. Choose from {list(MAB_SPECS)}")
    main(args.mabs)
