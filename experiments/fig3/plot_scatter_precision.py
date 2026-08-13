#!/usr/bin/env python3
"""Precision-coverage curves and scatter for same- and cross-species PLM.

Only the PLM differs:
  * same-species:  per-species PLM, asymbnln fusion head → reads 9s_<sp>_asymbnln.mztab
  * cross-species: human_iso PLM (transfer), hybrid fusion head

Caching is automatic and incremental: existing per-(condition, species)
curves are kept; only missing entries are recomputed. Delete the cache files
to force a full refresh.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})
from sklearn.metrics import auc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from experiments.utils.evaluation import (
    _normalize_to_massivekb, evaluate, MASSIVEKB_MASSES, load_mztab_with_mgf,
)
from peptide_priors.const import (
    COLOR_CASANOVO as COLOR_DNPS,
    COLOR_PP,
    result_run_path,
)

CACHE_CSV = "scatter_precision_cache.csv"
CURVES_CACHE = "pc_curves_cache.pkl"

SPECIES_DIRS = {
    "human":       "casanovo",
    "mouse":       "mus_musculus",
    "yeast":       "saccharomyces_cerevisiae",
    "bacillus":    "bacillus_subtilis",
    "honeybee":    "apis_mellifera",
    "tomato":      "solanum_lycopersicum",
    "cowpea":      "vigna_mungo",
    "archaeon":    "methanosarcina_mazei",
    "endoloripes": "candidatus_endoloripes",
}

# Display labels for plots (titles / scatter point labels). Lookup keys
# stay as above because the mztab filenames use them verbatim.
SPECIES_LABELS = {
    "cowpea":      "ricebean",
    "archaeon":    "mmazei",
    "endoloripes": "clambacteria",
}


def species_label(sp: str) -> str:
    return SPECIES_LABELS.get(sp, sp)

# (mztab suffix for the "+PP" arm, panel title)
CONDITIONS = {
    "same":  ("asymbnln",        "Same-species PLM"),
    "cross": ("plmhuman_iso_hybrid", "Cross-species PLM (human_iso)"),
}

YEAST_RESULTS = Path(result_run_path("saccharomyces_cerevisiae"))
YEAST_PP_PATHS = sorted(
    str(path) for path in YEAST_RESULTS.glob("9s_yeast_hybrid_*.mztab")
)


def load_pc_curve(path):
    paths = [path] if isinstance(path, (str, os.PathLike)) else path
    df = pd.concat(
        [load_mztab_with_mgf(input_path) for input_path in paths],
        ignore_index=True,
    )
    df["pred"] = df["pred"].apply(_normalize_to_massivekb)
    df["true_seq"] = df["true_seq"].apply(_normalize_to_massivekb)
    df["score"] = df["score"].astype(float)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    aa_matches_batch = evaluate.aa_match_batch(
        df["true_seq"].tolist(), df["pred"].tolist(), MASSIVEKB_MASSES,
    )
    peptide_matches = np.asarray([m[1] for m in aa_matches_batch[0]])
    precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
    coverage = np.arange(1, len(peptide_matches) + 1) / len(peptide_matches)
    return precision, coverage, float(precision[-1]), len(df)


# ── Load / refresh per-species PC curves ─────────────────────────────────────
curves = {}
if os.path.exists(CURVES_CACHE):
    with open(CURVES_CACHE, "rb") as _f:
        curves = pickle.load(_f)
    # Legacy "humanhead" key → "same"; legacy "hybrid" inner field → "pp".
    # Keep current-schema entries ("same"/"cross") as-is.
    def _mig(v):
        return {
            "dnps": v["dnps"],
            "pp": v.get("pp", v.get("hybrid")),
            "n": v["n"],
            "source_files": v.get("source_files"),
        }
    curves = {
        (("same" if k == "humanhead" else k), sp): _mig(v)
        for (k, sp), v in curves.items()
        if k in ("humanhead", "same", "cross")
    }
    print(f"Loaded {len(curves)} cached PC curves from {CURVES_CACHE}", file=sys.stderr)

# Human cross is degenerate (human PLM on human data = same-species), drop it.
curves.pop(("cross", "human"), None)

rows = []
for species, dirname in SPECIES_DIRS.items():
    base = result_run_path(dirname)
    dnps_path = os.path.join(base, f"9s_{species}_dnps.mztab")
    if not os.path.exists(dnps_path):
        continue
    d_curve = None  # lazy: only load if any condition needs (re)computation
    for cond, (suffix, _label) in CONDITIONS.items():
        if cond == "cross" and species == "human":
            continue
        if cond == "same" and species == "yeast":
            pp_path = YEAST_PP_PATHS
            source_files = tuple(pp_path)
        elif cond == "same" and species == "human":
            pp_path = os.path.join(base, "9s_human_plmhuman_iso_hybrid.mztab")
            source_files = None
        else:
            pp_path = os.path.join(base, f"9s_{species}_{suffix}.mztab")
            source_files = None
        if isinstance(pp_path, str) and not os.path.exists(pp_path):
            stem = Path(pp_path).stem
            candidates = sorted(Path(base).glob(f"{stem}_*.mztab"))
            if cond == "cross" and species in {"yeast", "bacillus"}:
                pp_path = [str(path) for path in candidates]
            elif candidates:
                pp_path = str(max(candidates, key=lambda path: path.stat().st_mtime))
        if source_files is None and (cond == "cross" or species == "human"):
            source_files = tuple(pp_path) if isinstance(pp_path, list) else (pp_path,)
        if not pp_path or (
            isinstance(pp_path, str) and not os.path.exists(pp_path)
        ):
            if (cond, species) in curves:
                cv = curves[(cond, species)]
                d_final = float(cv["dnps"][1][-1])
                pp_final = float(cv["pp"][1][-1])
                rows.append({
                    "species": species, "condition": cond,
                    "casanovo": d_final, "casanovo_pp": pp_final,
                    "rel_change": (pp_final - d_final) / d_final,
                    "n_psms": cv["n"],
                })
            continue
        cache_is_stale = (
            source_files is not None
            and curves.get((cond, species), {}).get("source_files") != source_files
        )
        if (cond, species) not in curves or cache_is_stale:
            if d_curve is None:
                print(f"Loading {species} dnps...", flush=True, file=sys.stderr)
                d_curve = load_pc_curve(dnps_path)
            d_prec, d_cov, _, n = d_curve
            print(f"Loading {species} {cond}...", flush=True, file=sys.stderr)
            pp_prec, pp_cov, _, _ = load_pc_curve(pp_path)
            curves[(cond, species)] = {
                "dnps": (d_cov, d_prec, auc(d_cov, d_prec)),
                "pp":   (pp_cov, pp_prec, auc(pp_cov, pp_prec)),
                "n": n,
                "source_files": source_files,
            }
        cv = curves[(cond, species)]
        d_final = float(cv["dnps"][1][-1])
        pp_final = float(cv["pp"][1][-1])
        rows.append({
            "species": species, "condition": cond,
            "casanovo": d_final, "casanovo_pp": pp_final,
            "rel_change": (pp_final - d_final) / d_final,
            "n_psms": cv["n"],
        })

with open(CURVES_CACHE, "wb") as _f:
    pickle.dump(curves, _f)
df = pd.DataFrame(rows)
df.to_csv(CACHE_CSV, index=False)
print(f"Cache: {len(curves)} curves → {CURVES_CACHE}, {len(df)} rows → {CACHE_CSV}",
      file=sys.stderr)
print("\n" + df.to_string(index=False))


# ── PC-curve grid ───────────────────────────────────────────────────────────
def _plot_panel(ax, cov_d, prec_d, auc_d, cov_p, prec_p, auc_p, title, n):
    ax.plot(cov_d, prec_d, color=COLOR_DNPS, lw=2.0,
            label=f"Casanovo (AUC={auc_d:.3f})")
    ax.plot(cov_p, prec_p, color=COLOR_PP, lw=2.0,
            label=f"Casanovo+PepPr (AUC={auc_p:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"{title} (n={n:,})", fontsize=13)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.grid(False)
    ax.tick_params(labelsize=11)


def _grid_figure(cond_key, ncols, out_path, suptitle):
    species_list = [sp for sp in SPECIES_DIRS if (cond_key, sp) in curves]
    if not species_list:
        return
    nrows = (len(species_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, species in enumerate(species_list):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        cv = curves[(cond_key, species)]
        _plot_panel(ax, *cv["dnps"], *cv["pp"], species_label(species), cv["n"])
        if r == nrows - 1:
            ax.set_xlabel("Coverage", fontsize=12)
        if c == 0:
            ax.set_ylabel("Peptide precision", fontsize=12)
    for idx in range(len(species_list), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    fig.suptitle(suptitle, fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


_grid_figure("same",  ncols=3, out_path="pc_curves_same_species.png",
             suptitle="Precision–Coverage: Same-species PLM")
_grid_figure("cross", ncols=3, out_path="pc_curves_cross_species.png",
             suptitle="Precision–Coverage: Cross-species PLM, human_iso")


# ── Scatter ─────────────────────────────────────────────────────────────────
from adjustText import adjust_text


def _scatter_panel(sc, out_path, title, legend_loc="lower right",
                   legend_bbox=None, legend_ncol=1):
    fig, ax = plt.subplots(figsize=(5, 5))
    for _, row in sc.iterrows():
        ax.plot([row["casanovo"], row["casanovo"]],
                [row["casanovo"], row["casanovo_pp"]],
                color=COLOR_PP, lw=1.5, alpha=0.5, zorder=1)
    median_delta = np.median(sc["casanovo_pp"] - sc["casanovo"])
    ax.scatter(sc["casanovo"], sc["casanovo_pp"], s=50, zorder=3,
               marker="o", color=COLOR_PP, edgecolors="white", linewidths=0.4,
               label=f"Casanovo+PepPr (median Δ={median_delta*100:+.1f} pp)")
    glow = [patheffects.withStroke(linewidth=3.0, foreground="white")]
    texts = [
        ax.text(row["casanovo"], row["casanovo_pp"],
                f"{species_label(row['species'])}\n({(row['casanovo_pp']-row['casanovo'])*100:+.1f} pp)",
                fontsize=10, color="black", path_effects=glow, zorder=5,
                ha="center", va="center", linespacing=1.05)
        for _, row in sc.iterrows()
    ]
    adjust_text(texts, ax=ax, seed=42,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                alpha=0.6, zorder=4),
                force_points=(10, 10), force_text=(2, 2), min_arrow_len=5,
                expand=(2, 2))
    all_vals = pd.concat([sc["casanovo"], sc["casanovo_pp"]])
    lo = all_vals.min() - 0.03
    hi = all_vals.max() + 0.03
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLOR_DNPS, lw=1.5, alpha=1.0,
            zorder=0, label="Casanovo")
    ax.set_xlim(lo, 0.8); ax.set_ylim(lo, hi)
    legend_kwargs = dict(fontsize=11, loc=legend_loc, framealpha=0.8,
                         ncol=legend_ncol)
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    ax.legend(**legend_kwargs)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Casanovo peptide precision", fontsize=13)
    ax.set_ylabel("Casanovo+PepPr peptide precision", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


same_sc  = df[df["condition"] == "same"]
cross_sc = df[df["condition"] == "cross"]
if len(same_sc):
    _scatter_panel(same_sc, "scatter_casanovo_vs_pp_same.png",
                   "Same-species PLM (human fusion head)")
if len(cross_sc):
    _scatter_panel(cross_sc, "scatter_casanovo_vs_pp_cross.png",
                   "Human prior, Nine-species data excl. human",
                   legend_loc="upper center",
                   legend_bbox=(0.5, -0.12), legend_ncol=2)
