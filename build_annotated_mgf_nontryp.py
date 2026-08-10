"""Build SEQ-annotated MGFs for each non-tryptic (mAb, protease) run.

Reads:
  - <run.workdir>/<exp>_<bioreplicate>/psm.tsv (FragPipe 1% FDR PSMs)
  - <run.prepared_mgf>                          (pepXML-style TITLE + SCANS=)
Writes:
  - <run.annotated_mgf>                         (adds SEQ=<ProForma> per matched scan)

Usage:
  python build_annotated_mgf_nontryp.py                # all entries
  python build_annotated_mgf_nontryp.py --idx 3 5      # only those indices
"""
from __future__ import annotations

import argparse
import os
import re

from nontryp_registry import RUNS


def parse_psm_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_spec = header.index("Spectrum")
        i_pep = header.index("Peptide")
        i_z = header.index("Charge")
        i_mods = header.index("Assigned Modifications")
        for line in f:
            row = line.rstrip("\n").split("\t")
            spec = row[i_spec]
            parts = spec.rsplit(".", 3)
            if len(parts) < 4:
                continue
            _raw_base, scan_str, _scan_hi, _z = parts
            try:
                scan = int(scan_str)
                z = int(row[i_z])
            except ValueError:
                continue
            yield scan, z, row[i_pep], row[i_mods]


def mods_to_proforma(plain_peptide: str, mods_str: str) -> str:
    out = list(plain_peptide)
    if not mods_str:
        return "".join(out)
    pat = re.compile(r"(\d+)([A-Za-z])\(([+-]?[\d.]+)\)")
    for m in pat.finditer(mods_str):
        pos = int(m.group(1))
        mass = float(m.group(3))
        if not 1 <= pos <= len(out):
            continue
        sign = "+" if mass >= 0 else "-"
        out[pos - 1] = f"{out[pos - 1]}[{sign}{abs(mass):.3f}]"
    return "".join(out)


def find_psm_tsv(workdir):
    for entry in sorted(os.listdir(workdir)):
        full = os.path.join(workdir, entry)
        if os.path.isdir(full):
            p = os.path.join(full, "psm.tsv")
            if os.path.exists(p):
                return p
    return None


def annotate_mgf(src_mgf, scan_to_psm, out_mgf):
    n_in = n_written = 0
    with open(src_mgf) as fi, open(out_mgf, "w") as fo:
        block = []
        scan = None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block = [line]
                scan = None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and scan in scan_to_psm:
                    seq, _charge = scan_to_psm[scan]
                    out_lines = []
                    inserted = False
                    for bl in block:
                        out_lines.append(bl)
                        if not inserted and bl.startswith("SCANS="):
                            out_lines.append(f"SEQ={seq}\n")
                            inserted = True
                    if not inserted:
                        out_lines = []
                        for bl in block:
                            out_lines.append(bl)
                            if not inserted and bl.startswith("CHARGE="):
                                out_lines.append(f"SEQ={seq}\n")
                                inserted = True
                    fo.writelines(out_lines)
                    fo.write("END IONS\n\n")
                    n_written += 1
                block = []
                continue
            block.append(line)
            if line.startswith("SCANS="):
                try:
                    scan = int(line.strip().split("=", 1)[1])
                except (ValueError, IndexError):
                    scan = None
    return n_in, n_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, nargs="*")
    args = p.parse_args()
    idx_set = set(args.idx) if args.idx else set(range(len(RUNS)))

    for i, run in enumerate(RUNS):
        if i not in idx_set:
            continue
        print(f"\n[{i:2d}] {run.mab_id:14s} {run.protease:12s}")
        psm_tsv = find_psm_tsv(run.workdir)
        if psm_tsv is None:
            print(f"     SKIP: no psm.tsv under {run.workdir}")
            continue
        scan_to_psm = {}
        n_rows = 0
        for scan, z, pep, mods in parse_psm_tsv(psm_tsv):
            n_rows += 1
            if scan in scan_to_psm:
                continue
            scan_to_psm[scan] = (mods_to_proforma(pep, mods), z)
        print(f"     PSMs 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(run.prepared_mgf):
            print(f"     SKIP: missing prepared MGF {run.prepared_mgf}")
            continue
        n_in, n_out = annotate_mgf(run.prepared_mgf, scan_to_psm, run.annotated_mgf)
        print(f"     MS2 in:  {n_in}  annotated: {n_out}")
        print(f"     -> {run.annotated_mgf}")


if __name__ == "__main__":
    main()
