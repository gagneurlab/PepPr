#!/usr/bin/env python3
"""Plot a single MS/MS spectrum from an MGF file by scan number.

Usage:
    python scripts/plot_spectrum.py MGF SCAN [--seq PROFORMA] [--out OUTPUT.png]

Examples:
    # Plot scan 2636 from the 36H6 pepsin MGF, annotated with the +PP prediction
    python scripts/plot_spectrum.py \\
        xa_novo/PXD060500_36H6/mgf/36H6-pepsin-HCD-20240524.mgf \\
        2636 \\
        --seq LKRADAAPTVS \\
        --out 36H6_pepsin_scan2636.png

    # If the MGF has its own SEQ= annotation, omit --seq to use it
    python scripts/plot_spectrum.py path/to.mgf 1234

The output filename defaults to '<mgf-basename>_scan<N>.png' next to the
working dir, so calls don't clobber each other.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
from pyteomics import mgf
import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from dnps_hybrid.const import COLOR_LIGHT_GRAY, COLOR_RED


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mgf", help="Path to MGF file")
    p.add_argument("scan", type=int, help="Scan number (SCANS= in MGF)")
    p.add_argument("--seq", default=None,
                   help="ProForma sequence to annotate (overrides SEQ= in MGF)")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <mgf-basename>_scan<N>.png)")
    p.add_argument("--title", default=None,
                   help="Optional title text (default: blank)")
    args = p.parse_args()

    with mgf.read(args.mgf) as reader:
        for spec in reader:
            if int(spec["params"].get("scans", -1)) == args.scan:
                break
        else:
            sys.exit(f"No spectrum with SCANS={args.scan} in {args.mgf}")

    params = spec["params"]
    spectrum = sus.MsmsSpectrum(
        identifier=f"scan={args.scan}",
        precursor_mz=float(params["pepmass"][0]),
        precursor_charge=int(params["charge"][0]),
        mz=spec["m/z array"],
        intensity=spec["intensity array"],
    )

    sup.colors["y"] = COLOR_RED
    sup.colors["?"] = COLOR_LIGHT_GRAY

    seq = args.seq if args.seq is not None else params.get("seq", "")
    if seq:
        spectrum.annotate_proforma(
            seq, fragment_tol_mass=20, fragment_tol_mode="ppm", ion_types="yb",
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    sup.spectrum(spectrum, ax=ax, grid=False)
    for line in ax.get_lines():
        line.set_linewidth(3)
    for text in ax.texts:
        text.set_fontsize(14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.yaxis.set_ticks([0, 1])
    ax.yaxis.set_ticklabels(["0%", "100%"])
    ax.set_xlabel(ax.get_xlabel(), fontsize=15, alpha=1.0)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15, alpha=1)
    ax.tick_params(labelsize=11, width=2)
    ax.set_title(args.title or "")
    plt.tight_layout()

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.mgf))[0]
        out = f"{base}_scan{args.scan}.png"
    else:
        out = args.out
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
