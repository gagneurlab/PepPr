"""Prep per-(mAb, protease) FragPipe workdir for the non-tryptic registry.

For each NonTrypRun:
  1. Copy/rewrite the source MGF into nontryp/<mab>/<protease>/<sample>.mgf
     with FragPipe-friendly TITLE (=<sample>.<scan>.<scan>.<charge>) and SCANS=<scan>.
     Handles two source flavours:
       - Beslic Centroided MGF: TITLE has 'Scan: N', no SCANS=.
       - PXD023419 RAW MGF: TITLE is raw filename, SCANS= already present.
  2. Build fragpipe_workdir/<sample>/  (FragPipe expects a per-experiment subdir).
  3. Write fragpipe.workflow with the right enzyme params + FASTA path.
  4. Write fragpipe-files.fp-manifest pointing at the prepared MGF.

Usage:
  python prepare_nontryp.py            # prep all entries
  python prepare_nontryp.py --idx 3 5  # prep indices 3 and 5 only
  python prepare_nontryp.py --check    # report what's prepared without writing
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

from nontryp_registry import RUNS, NonTrypRun

TRYP_WORKFLOW_TEMPLATE = (
    "/s/project/denovo-prosit/SamKhan/dnps_hybrid/beslic_mab/IgG1_Human_H/"
    "fragpipe_workdir/fragpipe.workflow"
)
TITLE_SCAN_RE = re.compile(r"Scan:\s*(\d+)")


def _fixup_mgf(src: str, dst: str, sample: str) -> tuple[int, int]:
    """Rewrite TITLE to pepXML-style `<sample>.<scan>.<scan>.<charge>` and ensure SCANS=<scan>.

    Source variants:
      A) Beslic Centroided: TITLE='Run: <name>, Index: I, Scan: N'  -> parse Scan from title.
      B) Herceptin RAW:    TITLE='<path>.raw' + SCANS=<N>           -> use existing SCANS.

    Always writes a fresh file (no hardlink to source).
    """
    n_in = n_out = 0
    with open(src) as fi, open(dst, "w") as fo:
        block = []
        scan = None
        charge = None
        for line in fi:
            if line.startswith("BEGIN IONS"):
                block = [line]
                scan = None
                charge = None
                continue
            if line.startswith("END IONS"):
                n_in += 1
                if scan is not None and charge is not None:
                    new_block = []
                    title_done = scans_done = False
                    for bl in block:
                        if bl.startswith("TITLE="):
                            if not title_done:
                                new_block.append(
                                    f"TITLE={sample}.{scan}.{scan}.{charge}\n"
                                )
                                title_done = True
                            continue
                        if bl.startswith("SCANS="):
                            # we'll re-emit a canonical SCANS line below
                            continue
                        new_block.append(bl)
                    # Insert SCANS after CHARGE.
                    out_lines = []
                    for bl in new_block:
                        out_lines.append(bl)
                        if not scans_done and bl.startswith("CHARGE="):
                            out_lines.append(f"SCANS={scan}\n")
                            scans_done = True
                    if not title_done:
                        # Prepend TITLE if the source had none
                        out_lines.insert(1, f"TITLE={sample}.{scan}.{scan}.{charge}\n")
                    if not scans_done:
                        # Last-resort: append SCANS before END IONS
                        out_lines.append(f"SCANS={scan}\n")
                    fo.writelines(out_lines)
                    fo.write("END IONS\n")
                    n_out += 1
                block = []
                continue
            block.append(line)
            if line.startswith("TITLE="):
                m = TITLE_SCAN_RE.search(line)
                if m:
                    scan = int(m.group(1))
            elif line.startswith("SCANS=") and scan is None:
                try:
                    scan = int(line.strip().split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CHARGE="):
                m = re.match(r"^CHARGE=(\d+)", line)
                if m:
                    charge = int(m.group(1))
    return n_in, n_out


def _patch_workflow(template_text: str, run: NonTrypRun) -> str:
    """Swap database/enzyme params in the tryptic template for this run."""
    out = []
    enz = run.enzyme
    replacements = {
        "database.db-path":                    run.fasta_td,
        "msfragger.search_enzyme_name_1":      enz.name,
        "msfragger.search_enzyme_cut_1":       enz.cut if enz.cut != "-" else "",
        "msfragger.search_enzyme_nocut_1":     enz.nocut,
        "msfragger.search_enzyme_sense_1":     enz.sense,
        "msfragger.allowed_missed_cleavage_1": str(enz.missed_cleavage),
        # Semi-specific: one peptide terminus must match enzyme rule, the other
        # can be anywhere. Captures ragged mAb termini (signal peptide remnants,
        # pyroglutamate, C-terminal K clipping) and looseness of elastase /
        # thermolysin / lysN that fully-specific (=2) would miss.
        "msfragger.num_enzyme_termini":        "1",
        # Single-mAb databases have only 2-4 proteins (target + decoy of H/L
        # chains), so Philosopher's protein-FDR filter can't converge even at
        # --prot 1.00 under semi-specific (too many decoy PSMs survive). PSM-
        # level FDR (1%) is what we actually need for the ground truth, so
        # disable protein-level inference entirely.
        "protein-prophet.run-protein-prophet": "false",
    }
    if enz.name == "nonspecific":
        # MSFragger: nonspecific search -> num_enzyme_termini=0, missed_cleavage=0
        replacements["msfragger.num_enzyme_termini"] = "0"
        replacements["msfragger.allowed_missed_cleavage_1"] = "0"
    for line in template_text.splitlines(keepends=True):
        key = line.split("=", 1)[0] if "=" in line else line
        if key in replacements:
            out.append(f"{key}={replacements[key]}\n")
        else:
            out.append(line)
    return "".join(out)


def _write_manifest(path: str, prepared_mgf: str, exp_name: str) -> None:
    # Format: <spectra-path>\t<experiment>\t<bioreplicate>\t<data-type>
    with open(path, "w") as fo:
        fo.write(f"{prepared_mgf}\t{exp_name}\t1\tDDA\n")


def _prepare_one(run: NonTrypRun, force: bool = False) -> dict:
    out_dir = os.path.dirname(run.prepared_mgf)
    workdir = run.workdir
    res_dir = run.res_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # 1. Fixup MGF.
    if force or not os.path.exists(run.prepared_mgf):
        n_in, n_out = _fixup_mgf(run.src_mgf, run.prepared_mgf, run.sample)
        status_mgf = f"rewritten {n_in}->{n_out}"
    else:
        status_mgf = "exists"

    # 2. Workflow.
    workflow_path = os.path.join(workdir, "fragpipe.workflow")
    with open(TRYP_WORKFLOW_TEMPLATE) as fi:
        tpl = fi.read()
    patched = _patch_workflow(tpl, run)
    with open(workflow_path, "w") as fo:
        fo.write(patched)

    # 3. Manifest. Experiment name = sample (lowercased, alnum-safe).
    exp = re.sub(r"[^A-Za-z0-9_]+", "_", run.sample).strip("_").lower()
    manifest_path = os.path.join(workdir, "fragpipe-files.fp-manifest")
    _write_manifest(manifest_path, run.prepared_mgf, exp)

    return {
        "mab": run.mab_id, "protease": run.protease,
        "mgf": status_mgf,
        "workflow": workflow_path,
        "manifest": manifest_path,
        "exp_name": exp,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, nargs="*",
                   help="Indices into RUNS (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Re-rewrite prepared MGFs even if they exist")
    p.add_argument("--check", action="store_true",
                   help="Report planned actions, do not write")
    args = p.parse_args()
    idx_set = set(args.idx) if args.idx else set(range(len(RUNS)))

    for i, run in enumerate(RUNS):
        if i not in idx_set:
            continue
        print(f"[{i:2d}] {run.mab_id:14s} {run.protease:12s}", flush=True)
        if args.check:
            print(f"     src    : {run.src_mgf} ({'OK' if os.path.exists(run.src_mgf) else 'MISSING'})")
            print(f"     prepd  : {run.prepared_mgf}")
            print(f"     workdir: {run.workdir}")
            continue
        result = _prepare_one(run, force=args.force)
        print(f"     mgf      : {result['mgf']}")
        print(f"     workflow : {result['workflow']}")
        print(f"     manifest : {result['manifest']}")
        print(f"     exp_name : {result['exp_name']}")


if __name__ == "__main__":
    main()
