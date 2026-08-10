#!/usr/bin/env python3
"""Sequence-diff + annotated spectrum panel for a Casanovo "wrong precursor
mass" prediction that Casanovo+PepPr recovers.

Picked from the mouse 9-species venn (`pp-only / wrong-mass` cell of
plot_scatter_precision.py):

  spectra_ref : ms_run[4]:index=1624
  MGF         : 20160323_CoAN_Deliti2_3379.mgf, SCANS=20843
  precursor   : 679.868 m/z, +2  →  [M+H]+ = 1358.73  →  M = 1357.72 Da
  truth       : VSSQTFPLAPSPK         (mass 1357.72 Da)
  Casanovo    : GSQQTFPLAPSPK  s=0.35 (mass 1356.69 Da → 1.03 Da / 752 ppm off precursor)
  Casanovo+PepPr : VSSQTFPLAPSPK  s=0.84 (correct)

Coupled error: V→G drops 42 Da at position 1, S→Q adds 41 Da at position 3,
net -1 Da. Each individual substitution gives a non-deamidation residue swap
(unlike Q↔E / N↔D), so the failure cannot be confused with a deamidation PTM.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnps_hybrid.metrics import (
    plot_sequence_diff, plot_spectrum_on_ax,
    annotate_missing_ion_gap, annotate_precursor_readout,
)
from dnps_hybrid.const import nine_species_benchmark_dir

EXAMPLE = dict(
    mgf=os.path.join(nine_species_benchmark_dir("mouse"),
                     "20160323_CoAN_Deliti2_3379.mgf"),
    scan=20843,
    true_seq="VSSQTFPLAPSPK",
    v_pred  ="GSQQTFPLAPSPK", v_score=0.35,
    pp_pred ="VSSQTFPLAPSPK", pp_score=0.84,
    precursor_mz=679.868469238281, precursor_charge=2,
)


def main(out_path):
    plt.style.use("ggplot")
    plt.rcParams.update({
        "axes.facecolor": "white", "axes.edgecolor": "black",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "font.family": "sans-serif",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
    })

    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(2, 1, height_ratios=[1, 2.2], hspace=0.30, figure=fig)
    ax_diff = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])

    # Sequence diff with PP / Casanovo / truth lines
    plot_sequence_diff(
        ax_diff,
        true_seq=EXAMPLE["true_seq"],
        preds=[("Casanovo",      EXAMPLE["v_pred"],  EXAMPLE["v_score"]),
               ("Casanovo + PepPr", EXAMPLE["pp_pred"], EXAMPLE["pp_score"])],
        boundary_pos=0,
        title=(f"{os.path.basename(EXAMPLE['mgf'])}, scan {EXAMPLE['scan']}  "
               f"— Casanovo prediction is 1.03 Da (752 ppm) lighter than precursor"),
    )
    # Remove the dashed boundary line the helper adds when boundary_pos=0
    for ln in list(ax_diff.lines):
        ln.remove()
    for txt in list(ax_diff.texts):
        if txt.get_text() == "V/C boundary":
            txt.remove()

    plot_spectrum_on_ax(ax_spec, mgf_path=EXAMPLE["mgf"],
                        scan=EXAMPLE["scan"], seq=EXAMPLE["true_seq"])

    # Diagnostic ions for Casanovo's two errors. b1 distinguishes V (100.08 m/z)
    # from G (58.03 m/z) at position 1; y11 vs y10 pins residue 3 (S vs Q).
    annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'b',
                              missing_idx=1, observed_idx=2)
    # y11 and y12 are both missing — drawing both as dashed lines and chaining
    # the arrows y10→y11→y12→precursor so each arrow corresponds to a single
    # residue (S at pos 3, S at pos 2, V at pos 1). Compact labels to avoid
    # collision since y11 (1172.6) and y12 (1259.7) are only 87 m/z apart.
    annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'y',
                              missing_idx=11, observed_idx=10,
                              missing_label="y₁₁")
    annotate_missing_ion_gap(ax_spec, EXAMPLE["true_seq"], 'y',
                              missing_idx=12, observed_idx=11,
                              missing_label="y₁₂")

    # Precursor readout — the actual mass-mismatch tell. Casanovo's prediction
    # would imply a [M+H]+ ~1 Da lower than the observed precursor.
    annotate_precursor_readout(ax_spec, EXAMPLE["true_seq"],
                               precursor_mz=EXAMPLE["precursor_mz"],
                               precursor_charge=EXAMPLE["precursor_charge"],
                               from_y_idx=12)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wrong_mass_example.png"
    main(out)
