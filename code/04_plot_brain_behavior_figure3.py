#!/usr/bin/env python3
"""Plot ROI-beta brain-behavior associations for RT and mean insights.

Inputs:
- glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/
  roi_betas_individual_rt_lmm_wide_subject_condition.csv
- glm_unified/behavioral_mean_insights_per_trial.csv

Outputs:
- integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures/
  fig1_brain_behavior_roi_beta_rt.png/.pdf
  behavior_roi_beta_insight.png/.pdf
  behavior_roi_beta_insight_correlations_fdr.csv

The RT figure intentionally overwrites the existing report figure with matched
CA/JX/OI colors. The insight figure is a new companion plot using the same ROI
beta table and subject-level mean insights per trial.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

RELEASE = Path(__file__).resolve().parents[1]
ROI_WIDE = RELEASE / "data" / "brain_behavior" / "roi_betas_rt_wide_subject_condition.csv"
INSIGHT = RELEASE / "data" / "behavioral" / "behavioral_mean_insights_per_trial.csv"
FIG_DIR = RELEASE / "figures" / "source_panels"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RT_PNG = FIG_DIR / "fig1_brain_behavior_roi_beta_rt.png"
RT_PDF = FIG_DIR / "fig1_brain_behavior_roi_beta_rt.pdf"
INSIGHT_PNG = FIG_DIR / "behavior_roi_beta_insight.png"
INSIGHT_PDF = FIG_DIR / "behavior_roi_beta_insight.pdf"
INSIGHT_STATS = FIG_DIR / "behavior_roi_beta_insight_correlations_fdr.csv"
RT_STATS = FIG_DIR / "fig1_brain_behavior_roi_beta_rt_correlations_fdr.csv"

ROIS = ["ATL", "L_Angular", "PCC", "vmPFC"]
ROI_TITLES = {"ATL": "ATL", "L_Angular": "L Angular", "PCC": "PCC", "vmPFC": "vmPFC"}
CONDITIONS = ["CA", "JX", "OI"]
# Matched to the recent connectivity figure: CA purple, JX golden-brown, OI slate blue/grey.
COLORS = {"CA": "#5E3C99", "JX": "#E69F00", "OI": "#4E79A7"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 18,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})


def bh_fdr(p_values: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, returned in original order."""
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def load_rt_long() -> pd.DataFrame:
    df = pd.read_csv(ROI_WIDE)
    cols = ["subject", "haiku_type", "RT"] + [f"{roi}_beta_z" for roi in ROIS]
    return df[cols].copy()


def load_insight_long(rt_like: pd.DataFrame) -> pd.DataFrame:
    insight = pd.read_csv(INSIGHT)
    long = insight.melt(
        id_vars="subject",
        value_vars=[f"mean_insights_{cond}" for cond in CONDITIONS],
        var_name="metric",
        value_name="mean_insights_per_trial",
    )
    long["haiku_type"] = long["metric"].str.replace("mean_insights_", "", regex=False)
    long = long.drop(columns="metric")
    merged = rt_like.drop(columns="RT").merge(long, on=["subject", "haiku_type"], how="inner")
    return merged


def compute_correlations(df: pd.DataFrame, y_col: str, out_path: Path) -> pd.DataFrame:
    rows = []
    for roi in ROIS:
        x_col = f"{roi}_beta_z"
        for cond in CONDITIONS:
            sub = df.loc[df["haiku_type"] == cond, ["subject", x_col, y_col]].dropna()
            if len(sub) >= 3:
                r, p = stats.pearsonr(sub[x_col], sub[y_col])
            else:
                r, p = np.nan, np.nan
            rows.append({
                "roi": roi,
                "haiku_type": cond,
                "n": int(len(sub)),
                "r": r,
                "p": p,
            })
    stat_df = pd.DataFrame(rows)
    stat_df["q_roi_x_condition_corr_12"] = bh_fdr(stat_df["p"].fillna(1).to_numpy())
    stat_df.to_csv(out_path, index=False)
    return stat_df


def p_text(p: float) -> str:
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def make_four_panel_scatter(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    stats_df: pd.DataFrame,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.7), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, roi in zip(axes, ROIS):
        x_col = f"{roi}_beta_z"
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
        for cond in CONDITIONS:
            sub = df.loc[df["haiku_type"] == cond, ["subject", x_col, y_col]].dropna()
            color = COLORS[cond]
            ax.scatter(
                sub[x_col], sub[y_col],
                s=62, color=color, edgecolor="white", linewidth=0.9,
                alpha=0.92, zorder=3,
            )
            if len(sub) >= 3 and sub[x_col].nunique() > 1:
                slope, intercept, *_ = stats.linregress(sub[x_col], sub[y_col])
                xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 80)
                ax.plot(xs, intercept + slope * xs, color=color, lw=2.4, alpha=0.85, zorder=2)

        ax.set_title(ROI_TITLES[roi], fontweight="bold", pad=10)
        ax.set_xlabel("Standardized ROI beta (z)")
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
        ax.set_xlim(-2.6, 2.6)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

        # Compact annotation: only show FDR-surviving correlations; trends/NS stay out of the plot.
        sig_rows = stats_df[(stats_df["roi"] == roi) & (stats_df["q_roi_x_condition_corr_12"] < 0.05)]
        if not sig_rows.empty:
            text_lines = []
            for _, row in sig_rows.iterrows():
                text_lines.append(f"{row['haiku_type']}: r = {row['r']:.2f}, q = {row['q_roi_x_condition_corr_12']:.3f}".replace("0.", ".").replace("-0.", "-."))
            ax.text(
                0.03, 0.97, "\n".join(text_lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
            )

    axes[0].set_ylabel(y_label)
    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c],
                   markeredgecolor="white", markeredgewidth=0.8, markersize=9, label=c)
            for c in CONDITIONS
        ]
        axes[0].legend(handles=legend_handles, title="Condition", frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=22, fontweight="bold", y=1.03)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rt_df = load_rt_long()
    rt_stats = compute_correlations(rt_df, "RT", RT_STATS)
    make_four_panel_scatter(
        rt_df,
        y_col="RT",
        y_label="RT to first insight (s)",
        title="Original condition ROI betas predicting RT",
        out_png=RT_PNG,
        out_pdf=RT_PDF,
        stats_df=rt_stats,
        y_limits=None,
        show_legend=True,
    )

    insight_df = load_insight_long(rt_df)
    insight_stats = compute_correlations(insight_df, "mean_insights_per_trial", INSIGHT_STATS)
    make_four_panel_scatter(
        insight_df,
        y_col="mean_insights_per_trial",
        y_label="Mean insights per trial",
        title="Original condition ROI betas predicting mean insights",
        out_png=INSIGHT_PNG,
        out_pdf=INSIGHT_PDF,
        stats_df=insight_stats,
        y_limits=None,
        show_legend=False,
    )

    print(f"Saved {RT_PNG}")
    print(f"Saved {RT_PDF}")
    print(f"Saved {RT_STATS}")
    print(f"Saved {INSIGHT_PNG}")
    print(f"Saved {INSIGHT_PDF}")
    print(f"Saved {INSIGHT_STATS}")
    print("\nInsight correlations:")
    print(insight_stats.to_string(index=False))


if __name__ == "__main__":
    main()
