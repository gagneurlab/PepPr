
"""De novo assembly comparison for single monoclonal antibodies.

Arms:
  vanilla              — Casanovo v5.0 baseline, no pepLM
  germline_<sp>_sw     — Casanovo+PP, species-matched germline pepLM trained on
                         sliding-window digest (germline_mouse_sw for 2B4/36H6/85F7,
                         germline_human_sw for S2P6)

For each (mAb, arm) tuple we:
  1. Gather the 5 protease mztabs from the mAb's casanovo_results dir
  2. ALPS-assemble (only Casanovo's mass-consistency filter applied) for k in {7,8,9,10}
  3. Union the contigs across k, dedup on I/L-folded sequence
  4. npysearch against the mature HC + LC in <mab>_ref.fasta (I/L-folded)

Reports longest aligned contig and CDR recall per chain and arm.
Output dumped to TSV for downstream plotting.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import npysearch as npy

from dnps_hybrid import const


ALPS_JAR = os.path.join(const.PROJECT_ROOT, "tools", "ALPS", "ALPS.jar")
_MOD_BRACKET = re.compile(r"\[[^\]]*\]")
_MOD_PAREN = re.compile(r"\([^)]*\)")
_MASS_SHIFT = re.compile(r"[+-]?\d+\.\d+")


@dataclass
class PSMRow:
    spectrum: str
    peptide: str
    aa_scores: str
    score: float


def strip_mods(seq: str) -> str:
    seq = _MOD_BRACKET.sub("", seq)
    seq = _MOD_PAREN.sub("", seq)
    return _MASS_SHIFT.sub("", seq)


def parse_mztab_for_alps(path: str) -> list[PSMRow]:
    """Load ALPS-ready PSM rows from a Casanovo mzTab file."""
    rows = []
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
            aa_values = [
                value for value in row.get("opt_ms_run[1]_aa_scores", "").split(",")
                if value
            ]
            if len(aa_values) != len(peptide):
                continue
            match = re.search(r"index=(\d+)", row.get("spectra_ref", ""))
            index = match.group(1) if match else str(len(rows))
            rows.append(
                PSMRow(
                    f"{sample_id}_idx{index}",
                    peptide,
                    " ".join(aa_values),
                    score,
                )
            )
    return rows


def write_alps_csv(
    psms: list[PSMRow],
    out_csv: str,
    score_threshold: float = 0.1,
) -> int:
    """Write ALPS input, excluding predictions below the score threshold."""
    count = 0
    with open(out_csv, "w") as handle:
        handle.write(
            "Spectrum Name,Casanovo Peptide,Casanovo aaScore,"
            "Casanovo Score,Area\n"
        )
        for psm in psms:
            if psm.score < score_threshold:
                continue
            handle.write(
                f"{psm.spectrum},{psm.peptide},{psm.aa_scores},{psm.score},1\n"
            )
            count += 1
    return count


def run_alps(csv_path: str, k: int, c: int) -> str:
    """Run ALPS and return the generated FASTA path."""
    fasta = f"{csv_path}.k{k}.fasta"
    if os.path.exists(fasta) and os.path.getmtime(fasta) >= os.path.getmtime(csv_path):
        print(f"  [skip] cached k={k}: {os.path.basename(fasta)}")
        return fasta
    command = ["java", "-jar", ALPS_JAR, csv_path, str(k), str(c)]
    print(f"  $ {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=os.path.dirname(csv_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ALPS STDOUT:\n" + result.stdout)
        print("ALPS STDERR:\n" + result.stderr)
        raise RuntimeError(f"ALPS failed (exit {result.returncode})")
    return fasta


def read_fasta(path: str) -> list[tuple[str, str]]:
    records = []
    name = None
    sequence = []
    if not os.path.exists(path):
        return records
    with open(path) as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence)))
                name, sequence = line[1:], []
            else:
                sequence.append(line.strip())
    if name is not None:
        records.append((name, "".join(sequence)))
    return records


def _parse_cdrs(header_tail: str) -> list[tuple[int, int]] | None:
    try:
        positions = [
            int(value)
            for value in header_tail.replace(",", " ").split()
            if value
        ]
    except ValueError:
        return None
    if len(positions) < 6:
        return None
    return [
        (positions[0], positions[1]),
        (positions[2], positions[3]),
        (positions[4], positions[5]),
    ]


def read_fasta_with_cdrs(path: str) -> dict[str, dict]:
    """Load reference sequences and optional CDR coordinates from FASTA."""
    records = {}
    name = None
    tail = ""
    sequence = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    records[name] = {
                        "seq": "".join(sequence),
                        "cdrs": _parse_cdrs(tail),
                    }
                header = line[1:].split(None, 1)
                name = header[0]
                tail = header[1] if len(header) > 1 else ""
                sequence = []
            else:
                sequence.append(line.strip())
    if name is not None:
        records[name] = {
            "seq": "".join(sequence),
            "cdrs": _parse_cdrs(tail),
        }
    return records


def il_eq(sequence: str) -> str:
    return sequence.replace("I", "L")


def npysearch_metrics(
    contigs: list[tuple[str, str]],
    reference: str,
    min_identity: float = 0.75,
    max_accepts: int = 5,
) -> dict:
    """Calculate PowerNovo-style local-alignment assembly metrics."""
    empty = {"mapped": []}
    if not contigs or not reference:
        return empty
    queries = {
        f"c{i}": il_eq(sequence)
        for i, (_, sequence) in enumerate(contigs)
        if sequence.strip()
    }
    if not queries:
        return empty
    result = npy.blast(
        query=queries,
        database={"ref": il_eq(reference)},
        minIdentity=min_identity,
        maxAccepts=max_accepts,
        alphabet="protein",
    )
    by_contig = {}
    for i, query_id in enumerate(result.get("QueryId", [])):
        try:
            contig_index = int(query_id[1:])
        except ValueError:
            continue
        ref_start = int(result["TargetMatchStart"][i]) - 1
        ref_end = int(result["TargetMatchEnd"][i])
        matches = int(result["NumMatches"][i])
        aligned = int(result["NumColumns"][i])
        identity = float(result["Identity"][i])
        if (
            contig_index not in by_contig
            or identity > by_contig[contig_index]["identity"]
        ):
            by_contig[contig_index] = {
                "matches": matches,
                "aligned_len": aligned,
                "identity": identity,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "q_start": int(result["QueryMatchStart"][i]) - 1,
                "q_end": int(result["QueryMatchEnd"][i]),
                "query_match": result["QueryMatchSeq"][i],
                "target_match": result["TargetMatchSeq"][i],
            }
    mapped = [
        (contigs[index][0], contigs[index][1], hit)
        for index, hit in sorted(by_contig.items())
    ]
    return {"mapped": mapped}


def cdr_recall(
    npy_result: dict,
    cdrs: list[tuple[int, int]] | None,
) -> float:
    """Return the fraction of CDR positions covered by a correct alignment."""
    if not cdrs or not npy_result.get("mapped"):
        return 0.0

    cdr_positions = set().union(
        *(set(range(start - 1, end)) for start, end in cdrs)
    )
    correctly_covered = set()
    for _, _, hit in npy_result["mapped"]:
        ref_position = hit["ref_start"]
        for query_aa, target_aa in zip(hit["query_match"], hit["target_match"]):
            if target_aa == "-":
                continue
            if ref_position in cdr_positions and query_aa == target_aa:
                correctly_covered.add(ref_position)
            ref_position += 1
    return len(correctly_covered) / len(cdr_positions) if cdr_positions else 0.0


def union_contigs(
    per_k: dict[int, list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    """Union per-k contigs, deduplicating I/L-equivalent sequences."""
    seen = {}
    for k in sorted(per_k):
        for name, sequence in per_k[k]:
            key = il_eq(sequence)
            if key not in seen:
                seen[key] = (f"k{k}:{name}", sequence)
    return list(seen.values())

XA_NOVO_DIR = const.XA_NOVO_DIR
BESLIC_DIR = const.BESLIC_DIR
NONTRYP_DIR = const.NONTRYP_DIR
OUT_DIR     = f"{XA_NOVO_DIR}/alps_2arm_all"
XANOVO_PROTEASES = ["aspn", "chymo", "elastase", "pepsin", "trypsin"]
BESLIC_H_PROTEASES = ["aspn", "chymo", "gluc", "lysc", "proteinasek", "trypsin"]
BESLIC_L_PROTEASES = ["aspn", "chymo", "gluc", "lysc", "proteinasek", "trypsin"]
FLAG_PROTEASES     = ["aspn", "chymo", "elastase", "gluc", "lysc", "lysn", "thermolysin"]
HERCEPTIN_PROTEASES = ["elastase", "gluc", "lysc", "lysn", "thermolysin"]
WIGG_PROTEASES     = ["aspn", "chymo"]


def _xanovo_res_dirs(mab: str) -> dict[str, str]:
    return {p: f"{XA_NOVO_DIR}/PXD060500_{mab}/casanovo_results"
            for p in XANOVO_PROTEASES}


def _beslic_res_dirs(mab: str, proteases: list[str]) -> dict[str, str]:
    out = {}
    for p in proteases:
        if p == "trypsin":
            out[p] = f"{BESLIC_DIR}/{mab}/casanovo_results"
        else:
            out[p] = f"{NONTRYP_DIR}/{mab}/{p}/casanovo_results"
    return out


# Per-mAb dispatch: which germline pepLM corresponds, and where the ref FASTA
# + casanovo_results dir live.
# `res_dirs` is keyed by protease so that the Beslic IgG1 mAbs (which live in
# per-protease subdirs across beslic_mab/ and nontryp/) plug in cleanly.
MAB_CONFIG = {
    "2B4":    {"species": "mouse", "pp_arm": "antibody_mouse",
               "ref": f"{XA_NOVO_DIR}/2B4_ref.fasta",
               "proteases": XANOVO_PROTEASES,
               "res_dirs": _xanovo_res_dirs("2B4")},
    "36H6":   {"species": "mouse", "pp_arm": "antibody_mouse",
               "ref": f"{XA_NOVO_DIR}/36H6_ref.fasta",
               "proteases": XANOVO_PROTEASES,
               "res_dirs": _xanovo_res_dirs("36H6")},
    "85F7":   {"species": "mouse", "pp_arm": "antibody_mouse",
               "ref": f"{XA_NOVO_DIR}/85F7_ref.fasta",
               "proteases": XANOVO_PROTEASES,
               "res_dirs": _xanovo_res_dirs("85F7")},
    "S2P6":   {"species": "human", "pp_arm": "antibody_human",
               "ref": f"{XA_NOVO_DIR}/PXD060500_S2P6/S2P6_ref.fasta",
               "proteases": XANOVO_PROTEASES,
               "res_dirs": _xanovo_res_dirs("S2P6")},
    # Beslic IgG1 mAbs — heavy/light chains in separate "mAbs", non-trypsin
    # results live in nontryp/<chain>/<protease>/, trypsin in beslic_mab/<chain>/.
    "IgG1_Human_H": {"species": "human (HC)", "pp_arm": "antibody_human",
                     "ref": f"{XA_NOVO_DIR}/IgG1_Human_H_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": BESLIC_H_PROTEASES,
                     "res_dirs": _beslic_res_dirs("IgG1_Human_H", BESLIC_H_PROTEASES)},
    "IgG1_Human_L": {"species": "human (LC)", "pp_arm": "antibody_human",
                     "ref": f"{XA_NOVO_DIR}/IgG1_Human_L_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": BESLIC_L_PROTEASES,
                     "res_dirs": _beslic_res_dirs("IgG1_Human_L", BESLIC_L_PROTEASES)},
    "Herceptin":    {"species": "human", "pp_arm": "antibody_human",
                     "ref": f"{XA_NOVO_DIR}/Herceptin_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": HERCEPTIN_PROTEASES,
                     "res_dirs": _beslic_res_dirs("Herceptin", HERCEPTIN_PROTEASES)},
    "anti-FLAG-M2": {"species": "mouse", "pp_arm": "antibody_mouse",
                     "ref": f"{XA_NOVO_DIR}/anti-FLAG-M2_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": FLAG_PROTEASES,
                     "res_dirs": _beslic_res_dirs("anti-FLAG-M2", FLAG_PROTEASES)},
    "WIgG1_H":      {"species": "mouse (HC)", "pp_arm": "antibody_mouse",
                     "ref": f"{XA_NOVO_DIR}/WIgG1_mouse_H_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": WIGG_PROTEASES,
                     "res_dirs": _beslic_res_dirs("WIgG1_H", WIGG_PROTEASES)},
    "WIgG1_L":      {"species": "mouse (LC)", "pp_arm": "antibody_mouse",
                     "ref": f"{XA_NOVO_DIR}/WIgG1_mouse_L_ref.fasta",
                     "vanilla_arm": "noplm",
                     "proteases": WIGG_PROTEASES,
                     "res_dirs": _beslic_res_dirs("WIgG1_L", WIGG_PROTEASES)},
}
MABS = list(MAB_CONFIG.keys())
K_MERS      = [7, 8, 9, 10]
TOP_CONTIGS = 20
# Suffix appended to the output TSV filename (set from --k so a single-k run
# doesn't clobber the union summary).
TSV_SUFFIX  = ""

# Optional global protease whitelist (set from --proteases). When non-None,
# each mAb's protease set is intersected with it.
PROTEASE_FILTER: set[str] | None = None


def proteases_for(mab: str) -> list[str]:
    ps = MAB_CONFIG[mab]["proteases"]
    if PROTEASE_FILTER is not None:
        ps = [p for p in ps if p in PROTEASE_FILTER]
    return ps


ARM_FILTER: set[str] | None = None  # set from --arms (matches series names)


def arms_for(mab: str) -> list[str]:
    cfg = MAB_CONFIG[mab]
    vanilla = cfg.get("vanilla_arm", "vanilla")
    arms = [vanilla, cfg["pp_arm"]]
    if ARM_FILTER is None:
        return arms
    series_for = {vanilla: "vanilla", cfg["pp_arm"]: "pp"}
    return [a for a in arms if series_for[a] in ARM_FILTER]


def find_mztab(mab: str, protease: str, arm: str) -> str | None:
    """Find the latest mztab for one (mab, protease, arm). Returns None if missing.
    Tries the protease-tagged filename first, then a protease-less fallback
    (legacy Beslic trypsin runs are named `casanovo_<mab>_<arm>_*.mztab`)."""
    res_dir = MAB_CONFIG[mab]["res_dirs"][protease]
    # Anchor arm name with the mab repeat that immediately follows it in the
    # filename (casanovo_{mab}_{protease}_{arm}_{mab}_*) so that
    # germline_mouse_* does not accidentally match germline_mouse_sw_* files.
    # The Beslic IgG1_Human_* legacy vanilla mztabs are named without the
    # trailing {mab}_{protease} repeat (e.g. casanovo_IgG1_Human_H_aspn_noplm_<jid>).
    # Newer runners emit the repeat (casanovo_...noplm_IgG1_Human_H_aspn_<jid>).
    # Try both, latest-mtime wins.
    patterns = [f"casanovo_{mab}_{protease}_{arm}_{mab}_*.mztab",
                f"casanovo_{mab}_{protease}_{arm}_[0-9]*.mztab",
                f"casanovo_{mab}_{arm}_*.mztab"]
    matches: list[str] = []
    for patt in patterns:
        matches.extend(glob.glob(os.path.join(res_dir, patt)))
    if not matches:
        return None
    # Latest by mtime so a re-run picks the newer file.
    return max(matches, key=os.path.getmtime)


def load_ref(mab: str) -> dict[str, dict]:
    """Return {chain_id: {"seq": str, "cdrs": [(s,e), (s,e), (s,e)] or None}}
    from <mab>_ref.fasta. CDR ranges come from the header tail
    (e.g. '>HC 26,33,51,58,97,108,120' → [(26,33), (51,58), (97,108)])."""
    return read_fasta_with_cdrs(MAB_CONFIG[mab]["ref"])


def assemble_arm(mab: str, arm: str) -> dict[int, list[tuple[str, str]]]:
    """Per-k ALPS contigs for (mab, arm), unioned across the mAb's proteases.
    Returns {k: [(name, seq), ...]}."""
    # Collect PSMs across all proteases.
    psms = []
    for protease in proteases_for(mab):
        path = find_mztab(mab, protease, arm)
        if path is None:
            print(f"  [{mab}/{arm}/{protease}] MISSING mztab", file=sys.stderr)
            continue
        got = parse_mztab_for_alps(path)
        print(f"  [{mab}/{arm}/{protease}] {os.path.basename(path)}: {len(got)} PSMs")
        psms.extend(got)
    if not psms:
        return {}

    os.makedirs(OUT_DIR, exist_ok=True)
    # write_alps_csv drops only the mass-mismatched (score<0) PSMs; everything
    # else is kept and ALPS' soft per-AA weighting handles quality.
    csv_path = f"{OUT_DIR}/{mab}_{arm}.csv"
    n = write_alps_csv(psms, csv_path)
    print(f"  -> {n} PSMs -> {os.path.basename(csv_path)}")
    if n == 0:
        return {}

    per_k: dict[int, list[tuple[str, str]]] = {}
    for k in K_MERS:
        fasta = run_alps(csv_path, k=k, c=TOP_CONTIGS)
        ctgs = read_fasta(fasta)
        per_k[k] = ctgs
        print(f"    k={k}: {len(ctgs)} contigs")
    return per_k


NPY_MIN_IDENTITY_CONTIG = 0.95
NPY_MIN_IDENTITY_CDR = 0.95
# Require high-identity alignments for both contig length and CDR recall so
# cross-chain chimeras do not contribute to either figure metric.


def summarize(contigs: list[tuple[str, str]], chain_name: str, chain_seq: str,
              cdrs: list[tuple[int, int]] | None) -> dict:
    """Calculate the assembly metrics used by the figure."""
    pn = npysearch_metrics(contigs, chain_seq,
                           min_identity=NPY_MIN_IDENTITY_CONTIG, max_accepts=5)
    longest_mapped = 0
    if pn["mapped"]:
        # Select by correctly matched residues rather than raw contig length.
        longest = max(pn["mapped"], key=lambda t: t[2]["matches"])
        longest_mapped = longest[2]["aligned_len"]

    pn_cdr = npysearch_metrics(contigs, chain_seq,
                               min_identity=NPY_MIN_IDENTITY_CDR, max_accepts=5)
    return {
        "chain": chain_name,
        "ref_len": len(chain_seq),
        "longest_aligned": longest_mapped,
        "cdr_coverage": cdr_recall(pn_cdr, cdrs),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for mab in MABS:
        ref = load_ref(mab)
        print(f"\n========= {mab} ({MAB_CONFIG[mab]['species']}) =========")
        print(f"  ref chains: " + ", ".join(
            f"{n}={len(d['seq'])} aa"
            + (f" CDRs={d['cdrs']}" if d['cdrs'] else " (no CDR header)")
            for n, d in ref.items()))

        for arm in arms_for(mab):
            print(f"\n--- {mab} / {arm} ---")
            per_k = assemble_arm(mab, arm)
            if not per_k:
                print(f"  no contigs for {mab}/{arm}; skipping")
                continue
            union = union_contigs(per_k)
            print(f"  union(k={','.join(map(str, K_MERS))}): "
                  f"{len(union)} unique contigs")

            for chain_name, chain_info in ref.items():
                row = summarize(
                    union, chain_name, chain_info["seq"], chain_info["cdrs"]
                )
                row["mab"] = mab
                row["arm"] = arm
                rows.append(row)

    print("\n\n=========================== SUMMARY ===========================")
    # Group by chain, placing canonical chain names first.
    canonical = ["HC", "LC", "HC2", "LC2"]
    other = sorted({r["chain"] for r in rows} - set(canonical))
    for chain in canonical + other:
        rows_c = [r for r in rows if r["chain"] == chain]
        if not rows_c:
            continue
        print(f"\n[{chain}]")
        hdr = (f"{'mab':<16s} {'arm':<24s} {'ref':>4s} "
               f"{'longest':>7s} {'CDRrec%':>7s}")
        print(hdr)
        for r in rows_c:
            print(f"{r['mab']:<16s} {r['arm']:<24s} "
                  f"{r['ref_len']:>4d} {r['longest_aligned']:>7d} "
                  f"{r['cdr_coverage']*100:>7.2f}")

    # TSV dump for downstream plotting.
    tsv = f"{OUT_DIR}/2arm_summary{TSV_SUFFIX}.tsv"
    with open(tsv, "w") as f:
        cols = [
            "mab", "arm", "chain", "ref_len", "longest_aligned", "cdr_coverage"
        ]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nSaved -> {tsv}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mabs", nargs="*", default=None,
                     help="Subset of mAb IDs to run (default: all)")
    _ap.add_argument("--k", type=int, nargs="*", default=None,
                     help="k-mer size(s) to assemble/union (default: 7 8 9 10)")
    _ap.add_argument("--proteases", nargs="*", default=None,
                     help="Restrict assembly to these proteases (intersected "
                          "with each mAb's available set; default: all).")
    _ap.add_argument("--arms", nargs="*", default=None,
                     choices=["vanilla", "pp"],
                     help="Restrict arms by series name (default: both).")
    _ap.add_argument("--tsv-suffix", default="",
                     help="Suffix appended to 2arm_summary.tsv (use to avoid "
                          "clobbering the main summary when running a subset).")
    _ap.add_argument("--cdr-identity", type=float, default=None,
                     help="Override the CDR-recall minimum identity "
                          "(default: 0.95).")
    _args = _ap.parse_args()
    if _args.cdr_identity is not None:
        NPY_MIN_IDENTITY_CDR = _args.cdr_identity
    if _args.proteases:
        PROTEASE_FILTER = set(_args.proteases)
    if _args.arms:
        ARM_FILTER = set(_args.arms)
    if _args.tsv_suffix:
        TSV_SUFFIX = _args.tsv_suffix
    if _args.mabs:
        for _m in _args.mabs:
            if _m not in MAB_CONFIG:
                raise SystemExit(f"Unknown mAb: {_m}. Choose from {list(MAB_CONFIG)}")
        MABS[:] = _args.mabs
    if _args.k:
        K_MERS[:] = _args.k
    main()
