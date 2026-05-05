#!/usr/bin/env python3
"""Plot archived 3-ROI FC as a compact conditional-GCA-style bar plot.

Uses the same old FC source and Fisher-z values as the within-condition
one-sample t-tests:
  glm_unified/connectivity_analysis/condition_specific_connectivity.csv

Figure layout follows fig_gca_condition_30s_bars.png: ROI pairs on the y-axis,
CA/JX/OI as offset coloured horizontal bars within each pair. No legend,
significance stars, or note text are drawn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RELEASE = Path(__file__).resolve().parents[1]
FIG_DIR = RELEASE / "figures/source_panels"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT = RELEASE / "data/connectivity/condition_specific_connectivity_legacy_subject_values.csv"
STATS_IN = RELEASE / "data/connectivity/fc_3roi_within_condition_fisherz_tests.csv"

OUT_BAR_PNG = FIG_DIR / "fig_fc_3roi_condition_bars_gca_style.png"
OUT_BAR_PDF = FIG_DIR / "fig_fc_3roi_condition_bars_gca_style.pdf"
OUT_VALUES = FIG_DIR / "fig_fc_3roi_condition_bars_gca_style_values.csv"
OUT_POINTS = FIG_DIR / "fig_fc_3roi_condition_bars_gca_style_subject_values.csv"

# Keep the previously shared filenames updated so manuscript/report links do not break.
OUT_DOTS_COMPAT_PNG = FIG_DIR / "fig_fc_3roi_condition_dots_gca_style.png"
OUT_DOTS_COMPAT_PDF = FIG_DIR / "fig_fc_3roi_condition_dots_gca_style.pdf"
OUT_ACTIVITY_COMPAT_PNG = FIG_DIR / "fig_3roi_activity_scatter_by_condition.png"
OUT_ACTIVITY_COMPAT_PDF = FIG_DIR / "fig_3roi_activity_scatter_by_condition.pdf"

COND_ORDER = ["CA", "JX", "OI"]
COLORS = {
    "CA": "#6B4C9A",  # muted purple
    "JX": "#E69F00",  # soft orange
    "OI": "#4C78A8",  # muted blue
}

PAIR_LABELS = {
    tuple(sorted(("PCC", "vmPFC"))): "PCC-vmPFC",
    tuple(sorted(("PCC", "L_Angular"))): "PCC-L_Angular",
    tuple(sorted(("L_Angular", "vmPFC"))): "L_Angular-vmPFC",
}
PAIR_ORDER = ["PCC-L_Angular", "L_Angular-vmPFC", "PCC-vmPFC"]
PAIR_DISPLAY = {
    "PCC-L_Angular": "PCC–L.Angular",
    "L_Angular-vmPFC": "L.Angular–vmPFC",
    "PCC-vmPFC": "PCC–vmPFC",
}


def canonical_pair(roi1: str, roi2: str) -> str | None:
    return PAIR_LABELS.get(tuple(sorted((roi1, roi2))))


def main() -> None:
    df = pd.read_csv(INPUT)
    df["pair"] = [canonical_pair(a, b) for a, b in zip(df["roi1"], df["roi2"])]
    df = df.loc[df["pair"].isin(PAIR_ORDER) & df["condition"].isin(COND_ORDER)].copy()
    df["fisher_z"] = np.arctanh(df["r"].clip(-0.999999, 0.999999))
    df["pair"] = pd.Categorical(df["pair"], categories=PAIR_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)
    df = df.sort_values(["pair", "condition", "subject"]).reset_index(drop=True)
    df.to_csv(OUT_POINTS, index=False)

    stats_df = pd.read_csv(STATS_IN)
    stats_df = stats_df.loc[stats_df["pair"].isin(PAIR_ORDER) & stats_df["condition"].isin(COND_ORDER)].copy()
    stats_df["pair"] = pd.Categorical(stats_df["pair"], categories=PAIR_ORDER, ordered=True)
    stats_df["condition"] = pd.Categorical(stats_df["condition"], categories=COND_ORDER, ordered=True)
    stats_df = stats_df.sort_values(["pair", "condition"]).reset_index(drop=True)
    stats_df.to_csv(OUT_VALUES, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 19,
        "savefig.dpi": 300,
    })

    # Match the conditional GCA bar plot canvas height.
    fig, ax = plt.subplots(figsize=(10.2, 7.1), dpi=240)
    base_y = np.arange(len(PAIR_ORDER))[::-1]
    offsets = {"CA": 0.22, "JX": 0.0, "OI": -0.22}
    bar_h = 0.18

    for cond in COND_ORDER:
        yy = base_y + offsets[cond]
        sub_stats = stats_df.loc[stats_df["condition"].astype(str).eq(cond)].set_index("pair").loc[PAIR_ORDER]
        means = sub_stats["mean_fisher_z"].to_numpy(dtype=float)
        sems = sub_stats["sem_fisher_z"].to_numpy(dtype=float)
        ax.barh(
            yy,
            means,
            xerr=sems,
            height=bar_h,
            color=COLORS[cond],
            alpha=0.94,
            error_kw={"elinewidth": 1.8, "capsize": 4, "capthick": 1.8, "ecolor": "#2B2B2B"},
            zorder=3,
        )

    ax.axvline(0, color="#404040", lw=1.0)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(base_y, [PAIR_DISPLAY[p] for p in PAIR_ORDER])
    ax.set_xlabel("Functional connectivity (Fisher z)", fontsize=21)
    ax.set_xlim(-0.04, 0.72)
    ax.tick_params(axis="both", labelsize=19)

    fig.tight_layout()
    for out in [OUT_BAR_PNG, OUT_BAR_PDF, OUT_DOTS_COMPAT_PNG, OUT_DOTS_COMPAT_PDF, OUT_ACTIVITY_COMPAT_PNG, OUT_ACTIVITY_COMPAT_PDF]:
        fig.savefig(out, bbox_inches="tight")

    print(f"Saved {OUT_BAR_PNG}")
    print(f"Saved {OUT_BAR_PDF}")
    print(f"Saved compatibility copies {OUT_DOTS_COMPAT_PNG} and {OUT_ACTIVITY_COMPAT_PNG}")
    print(f"Saved {OUT_VALUES}")
    print(f"Saved {OUT_POINTS}")
    print(stats_df[["pair", "condition", "N", "mean_fisher_z", "sem_fisher_z", "p", "q_3pairs_x_3conditions"]].to_string(index=False))


if __name__ == "__main__":
    main()
