"""Build SEQ-annotated MGFs for mAb benchmarks from FragPipe psm.tsv + spectra.

Two subcommands share the same PSM-parsing / ProForma / MGF-annotation core:

  beslic   -- the beslic_mab tryptic chains (IgG1_Human H/L). Sources may be
              mzML, mzXML, or MGF; the source flavour is auto-dispatched.
  nontryp  -- every (mAb, protease) entry in the non-tryptic registry. Sources
              are the pepXML-style prepared MGFs written by prepare_nontryp.py.

Outputs a copy of the spectra with `SEQ=<ProForma>` inserted for each scan that
carries a 1%-FDR PSM.

Usage:
  python -m antibody.build_annotated_mgf beslic                 # all beslic mAbs
  python -m antibody.build_annotated_mgf beslic IgG1_Human_H    # one mAb
  python -m antibody.build_annotated_mgf nontryp                # all registry runs
  python -m antibody.build_annotated_mgf nontryp --idx 3 5      # only those indices
"""
from __future__ import annotations

import argparse
import os
import re

from dnps_hybrid.const import DATA_PATH
from antibody.nontryp_registry import RUNS

# beslic_mab tryptic chains: (mab_id, sample_basename, ext). After the MSV
# samples are converted they ship as centroided MGF with
# TITLE=<sample>.<scan>.<scan>.<charge> + SCANS=<scan>.
BESLIC_MABS = [
    ("IgG1_Human_H", "Heavy-Chain-Trypsin-1", "mgf"),
    ("IgG1_Human_L", "Light-Chain-Trypsin-1", "mgf"),
]


# ── Shared PSM → ProForma helpers ─────────────────────────────────────────
def parse_psm_tsv(path):
    """Yield (raw_basename, scan, charge, peptide, modifications) from a psm.tsv."""
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
            raw_base, scan_str, _scan_hi, _z = parts
            try:
                scan = int(scan_str)
                z = int(row[i_z])
            except ValueError:
                continue
            yield raw_base, scan, z, row[i_pep], row[i_mods]


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


def find_psm_tsv(workdir, required=True):
    """psm.tsv lives under fragpipe_workdir/<experiment>_<bioreplicate>/psm.tsv."""
    for entry in sorted(os.listdir(workdir)):
        full = os.path.join(workdir, entry)
        if os.path.isdir(full):
            p = os.path.join(full, "psm.tsv")
            if os.path.exists(p):
                return p
    if required:
        raise FileNotFoundError(f"no psm.tsv under {workdir}")
    return None


def scan_to_psm_map(psm_tsv):
    """Collapse a psm.tsv to {scan: (ProForma, charge)} keeping the first PSM per scan."""
    scan_to_psm = {}
    n_rows = 0
    for _raw_base, scan, z, pep, mods in parse_psm_tsv(psm_tsv):
        n_rows += 1
        if scan in scan_to_psm:
            continue
        scan_to_psm[scan] = (mods_to_proforma(pep, mods), z)
    return scan_to_psm, n_rows


def _insert_seq(block, seq):
    """Insert a `SEQ=<seq>` line after SCANS= (or CHARGE= as fallback) in an MGF block."""
    out_lines, inserted = [], False
    for bl in block:
        out_lines.append(bl)
        if not inserted and bl.startswith("SCANS="):
            out_lines.append(f"SEQ={seq}\n")
            inserted = True
    if not inserted:
        out_lines, inserted = [], False
        for bl in block:
            out_lines.append(bl)
            if not inserted and bl.startswith("CHARGE="):
                out_lines.append(f"SEQ={seq}\n")
                inserted = True
    return out_lines


def annotate_mgf_from_mgf(mgf_in, scan_to_psm, out_mgf):
    """Source MGF already carries SCANS=<scan>; emit a copy with SEQ= for matches."""
    n_in = n_written = 0
    with open(mgf_in) as fi, open(out_mgf, "w") as fo:
        block, scan = [], None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block, scan = [line], None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and scan in scan_to_psm:
                    seq, _charge = scan_to_psm[scan]
                    fo.writelines(_insert_seq(block, seq))
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


def annotate_mzml(mzml_path, scan_to_psm, out_mgf):
    from pyteomics import mzml

    n_ms2 = n_written = 0
    with open(out_mgf, "w") as out, mzml.read(mzml_path) as reader:
        for spec in reader:
            if spec.get("ms level") != 2:
                continue
            n_ms2 += 1
            scan_id = spec.get("id", "")
            m = re.search(r"\bscan=(\d+)\b", scan_id)
            if not m:
                continue
            scan = int(m.group(1))
            psm = scan_to_psm.get(scan)
            if psm is None:
                continue
            seq, charge = psm
            try:
                pl = spec["precursorList"]["precursor"][0]
                sil = pl["selectedIonList"]["selectedIon"][0]
                pepmass = sil.get("selected ion m/z")
            except (KeyError, IndexError):
                continue
            try:
                rt_min = spec["scanList"]["scan"][0].get("scan start time")
                rt_sec = float(rt_min) * 60.0 if rt_min is not None else None
            except (KeyError, IndexError):
                rt_sec = None
            out.write("BEGIN IONS\n")
            out.write(f"TITLE={scan_id}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt_sec is not None:
                out.write(f"RTINSECONDS={rt_sec}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(spec["m/z array"], spec["intensity array"]):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


def annotate_mzxml(mzxml_path, scan_to_psm, out_mgf):
    from pyteomics import mzxml

    n_ms2 = n_written = 0
    with open(out_mgf, "w") as out, mzxml.read(mzxml_path) as reader:
        for spec in reader:
            if spec.get("msLevel") != 2:
                continue
            n_ms2 += 1
            scan = int(spec.get("num"))
            psm = scan_to_psm.get(scan)
            if psm is None:
                continue
            seq, charge = psm
            try:
                prec = spec["precursorMz"][0]
                pepmass = prec.get("precursorMz")
            except (KeyError, IndexError):
                continue
            rt = spec.get("retentionTime")  # pyteomics returns seconds (float)
            out.write("BEGIN IONS\n")
            out.write(f"TITLE=scan={scan}\n")
            out.write(f"PEPMASS={pepmass}\n")
            if rt is not None:
                out.write(f"RTINSECONDS={float(rt)}\n")
            out.write(f"CHARGE={charge}+\n")
            out.write(f"SCANS={scan}\n")
            out.write(f"SEQ={seq}\n")
            for mz, it in zip(spec["m/z array"], spec["intensity array"]):
                out.write(f"{mz} {it}\n")
            out.write("END IONS\n\n")
            n_written += 1
    return n_ms2, n_written


_ANNOTATE_BY_EXT = {
    "mzml": annotate_mzml,
    "mzxml": annotate_mzxml,
    "mgf": annotate_mgf_from_mgf,
}


# ── Subcommands ───────────────────────────────────────────────────────────
def cmd_beslic(args):
    beslic = os.path.join(DATA_PATH, "beslic_mab")
    only = set(args.mabs) or None
    for mab_id, basename, ext in BESLIC_MABS:
        if only is not None and mab_id not in only:
            continue
        d = f"{beslic}/{mab_id}"
        spec_path = f"{d}/mzml/{basename}.{ext}"
        out_mgf = f"{d}/{basename}_annotated.mgf"
        print(f"\n=== {mab_id} ===")
        try:
            psm_tsv = find_psm_tsv(f"{d}/fragpipe_workdir")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        scan_to_psm, n_rows = scan_to_psm_map(psm_tsv)
        print(f"  PSMs at 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(spec_path):
            print(f"  SKIP: spectra file missing {spec_path}")
            continue
        annotate = _ANNOTATE_BY_EXT.get(ext.lower())
        if annotate is None:
            print(f"  SKIP: unknown extension {ext}")
            continue
        n_ms2, n_written = annotate(spec_path, scan_to_psm, out_mgf)
        print(f"  MS2 in spectra: {n_ms2}; MGF blocks written: {n_written}")
        print(f"  -> {out_mgf}")


def cmd_nontryp(args):
    idx_set = set(args.idx) if args.idx else set(range(len(RUNS)))
    for i, run in enumerate(RUNS):
        if i not in idx_set:
            continue
        print(f"\n[{i:2d}] {run.mab_id:14s} {run.protease:12s}")
        psm_tsv = find_psm_tsv(run.workdir, required=False)
        if psm_tsv is None:
            print(f"     SKIP: no psm.tsv under {run.workdir}")
            continue
        scan_to_psm, n_rows = scan_to_psm_map(psm_tsv)
        print(f"     PSMs 1% FDR: {n_rows}; unique scans: {len(scan_to_psm)}")
        if not os.path.exists(run.prepared_mgf):
            print(f"     SKIP: missing prepared MGF {run.prepared_mgf}")
            continue
        n_in, n_out = annotate_mgf_from_mgf(run.prepared_mgf, scan_to_psm, run.annotated_mgf)
        print(f"     MS2 in:  {n_in}  annotated: {n_out}")
        print(f"     -> {run.annotated_mgf}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("beslic", help="beslic_mab tryptic chains")
    pb.add_argument("mabs", nargs="*", help="mab ids to build (default: all)")
    pb.set_defaults(func=cmd_beslic)

    pn = sub.add_parser("nontryp", help="non-tryptic (mAb, protease) registry runs")
    pn.add_argument("--idx", type=int, nargs="*", help="indices into RUNS (default: all)")
    pn.set_defaults(func=cmd_nontryp)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
