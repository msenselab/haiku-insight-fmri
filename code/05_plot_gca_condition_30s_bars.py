#!/usr/bin/env python3
"""Plot section 4.3 full 30-s condition-window GCA DOI values as bars."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

RELEASE = Path(__file__).resolve().parents[1]
VALUES = RELEASE / "data/granger/gca_figure_values.csv"
FIG_DIR = RELEASE / "figures/source_panels"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "fig_gca_condition_30s_bars.png"
OUT_PDF = FIG_DIR / "fig_gca_condition_30s_bars.pdf"
OUT_STATS = FIG_DIR / "fig_gca_condition_30s_bars_values.csv"

PAIR_ORDER = ["L_Angular_PCC", "L_Angular_vmPFC", "PCC_vmPFC"]
PAIR_LABELS = {
    "L_Angular_PCC": "L.Angular → PCC",
    "L_Angular_vmPFC": "L.Angular → vmPFC",
    "PCC_vmPFC": "PCC → vmPFC",
}
COND_ORDER = ["CA", "JX", "OI"]
COLORS = {
    "CA": "#6B4C9A",  # muted purple
    "JX": "#E69F00",  # soft orange
    "OI": "#4C78A8",  # muted blue
}


def sig_marker(q: float) -> str:
    """Markers are based on the plotted q values from the 9 within-condition tests."""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    if q < 0.10:
        return "†"
    return ""


def main() -> None:
    df = pd.read_csv(VALUES)
    cond = df.loc[df["analysis"].eq("condition_30s_window")].copy()
    cond = cond.loc[cond["pair"].isin(PAIR_ORDER) & cond["condition"].isin(COND_ORDER)].copy()
    cond["pair"] = pd.Categorical(cond["pair"], categories=PAIR_ORDER, ordered=True)
    cond["condition"] = pd.Categorical(cond["condition"], categories=COND_ORDER, ordered=True)
    cond = cond.sort_values(["pair", "condition"]).reset_index(drop=True)
    cond["pair_label"] = cond["pair"].map(PAIR_LABELS)
    cond.to_csv(OUT_STATS, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 17,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(10.2, 7.1), dpi=240)
    base_y = np.arange(len(PAIR_ORDER))[::-1]
    offsets = {"CA": 0.25, "JX": 0.0, "OI": -0.25}
    bar_h = 0.21

    for cond_name in COND_ORDER:
        sub = cond.loc[cond["condition"].eq(cond_name)].set_index("pair").loc[PAIR_ORDER]
        yy = base_y + offsets[cond_name]
        vals = sub["beta_loggc_doi"].to_numpy(dtype=float)
        sem = sub["sem_loggc_doi"].to_numpy(dtype=float)
        ax.barh(
            yy,
            vals,
            xerr=sem,
            height=bar_h,
            color=COLORS[cond_name],
            alpha=0.94,
            label=cond_name,
            error_kw={"elinewidth": 1.8, "capsize": 4, "capthick": 1.8, "ecolor": "#2B2B2B"},
        )
        for y_i, v, e, q in zip(yy, vals, sem, sub["q_within_family"].to_numpy(dtype=float)):
            marker = sig_marker(float(q))
            if not marker:
                continue
            ax.text(
                v + e + 0.004,
                y_i,
                marker,
                va="center",
                ha="left",
                fontsize=26,
                fontweight="bold",
                color="#111111",
                path_effects=[path_effects.withStroke(linewidth=3.2, foreground="white")],
                zorder=6,
            )

    ax.axvline(0, color="#404040", lw=1.0)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(base_y, [PAIR_LABELS[p] for p in PAIR_ORDER])
    ax.set_xlabel("β / mean DOI (logGC forward − reverse)", fontsize=22)
    ax.set_xlim(-0.004, 0.065)
    ax.tick_params(axis="both", labelsize=20)
    leg = ax.legend(title="Condition", frameon=True, loc="upper right", title_fontsize=17)
    leg.get_frame().set_edgecolor("#D0D0D0")
    leg.get_frame().set_alpha(0.94)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_STATS}")
    print(cond[["direction_label", "condition", "n", "beta_loggc_doi", "sem_loggc_doi", "p_within_unc", "q_within_family"]].to_string(index=False))


if __name__ == "__main__":
    main()
