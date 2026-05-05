#!/usr/bin/env python3
"""Plot reduced 3-ROI PCC search-onset RT-modulation PPI effects.

Creates a compact bar graph for PCC→L_Angular and PCC→vmPFC beta estimates
across CA, JX, and OI from the reduced 3-ROI behavioral gPPI.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
OUT_DIR = RELEASE / "data/ppi"
FIG_DIR = RELEASE / "figures/source_panels"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GROUP_CSV = OUT_DIR / "ppi_group_rt_modulation_betas_fdr.csv"
PLOT_CSV = FIG_DIR / "ppi_rt_modulation_3roi_bar_values.csv"
PNG = FIG_DIR / "ppi_rt_modulation_3roi_bars.png"
PDF = FIG_DIR / "ppi_rt_modulation_3roi_bars.pdf"

TARGET_ORDER = ["L_Angular", "vmPFC"]
TARGET_LABELS = {"L_Angular": "PCC → L Angular", "vmPFC": "PCC → vmPFC"}
CONDITION_ORDER = ["CA", "JX", "OI"]
COLORS = {
    "CA": "#5B3C88",   # muted deep purple
    "JX": "#D88945",   # soft orange
    "OI": "#6F7F8F",   # muted blue-grey
}


def main() -> None:
    df = pd.read_csv(GROUP_CSV)
    df = df[df["target"].isin(TARGET_ORDER) & df["condition"].isin(CONDITION_ORDER)].copy()
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
    df = df.sort_values(["target", "condition"])
    df.to_csv(PLOT_CSV, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharex=True, sharey=True)
    y = np.arange(len(CONDITION_ORDER))

    xlim = (-0.05, 0.025)
    xticks = [-0.05, -0.03, -0.01, 0.01, 0.025]
    xticklabels = [f"{x:.2f}" for x in xticks]

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].set_index("condition").loc[CONDITION_ORDER]
        means = sub["mean_beta"].to_numpy()
        sems = sub["sem_beta"].to_numpy()
        colors = [COLORS[c] for c in CONDITION_ORDER]

        ax.barh(y, means, xerr=sems, height=0.60, color=colors, edgecolor="white", linewidth=1.2,
                error_kw={"elinewidth": 1.3, "capsize": 4, "capthick": 1.3, "ecolor": "#333333"})
        ax.axvline(0, color="#333333", linewidth=1.1)
        ax.set_title(TARGET_LABELS[target], pad=8, weight="bold")
        ax.set_yticks(y, CONDITION_ORDER)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="x", color="#D0D0D0", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0].set_ylabel("Condition")
    fig.supxlabel("PPI beta (PCC × RT modulation)", y=0.055, fontsize=14)
    fig.suptitle("PCC-seed RT-modulation PPI effects", y=0.98, fontsize=18, weight="bold")
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.94), w_pad=2.0)

    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(f"Saved PNG: {PNG}")
    print(f"Saved PDF: {PDF}")
    print(f"Saved plotted values: {PLOT_CSV}")


if __name__ == "__main__":
    main()
