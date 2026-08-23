"""Plot a single MS/MS spectrum from an MGF file by scan number.

Importable utility (no CLI):

    from experiments.utils.plot_spectrum import plot_spectrum
    plot_spectrum("path/to.mgf", 1234, seq="PEPTIDE", out="scan1234.png")

If ``seq`` is None the MGF's own ``SEQ=`` annotation is used. ``out`` defaults to
``<mgf-basename>_scan<N>.png``. Returns the output path.
"""
import os
import sys

import matplotlib.pyplot as plt
from pyteomics import mgf
import spectrum_utils.spectrum as sus
import spectrum_utils.plot as sup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from peppr.const import COLOR_LIGHT_GRAY, COLOR_RED


def plot_spectrum(mgf_path, scan, seq=None, out=None, title=None):
    """Plot MS/MS scan ``scan`` from ``mgf_path``, annotate with ``seq`` (or the
    MGF's ``SEQ=``), save to ``out``, and return the output path.

    Raises ValueError if no spectrum with ``SCANS=<scan>`` is found.
    """
    with mgf.read(mgf_path) as reader:
        for spec in reader:
            if int(spec["params"].get("scans", -1)) == scan:
                break
        else:
            raise ValueError(f"No spectrum with SCANS={scan} in {mgf_path}")

    params = spec["params"]
    spectrum = sus.MsmsSpectrum(
        identifier=f"scan={scan}",
        precursor_mz=float(params["pepmass"][0]),
        precursor_charge=int(params["charge"][0]),
        mz=spec["m/z array"],
        intensity=spec["intensity array"],
    )

    sup.colors["y"] = COLOR_RED
    sup.colors["?"] = COLOR_LIGHT_GRAY

    seq = seq if seq is not None else params.get("seq", "")
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
    ax.set_title(title or "")
    plt.tight_layout()

    if out is None:
        base = os.path.splitext(os.path.basename(mgf_path))[0]
        out = f"{base}_scan{scan}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
