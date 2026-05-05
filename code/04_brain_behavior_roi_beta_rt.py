#!/usr/bin/env python3
"""Prepare original condition-level ROI-beta × RT data and run LMMs.

Uses the existing manuscript-era / previous ROI beta table:
    glm_unified/roi_betas_individual.csv
and merges it with condition-level behavioral RT:
    glm_unified/behavioral_rt.csv

Outputs both a wide subject × condition table and a long subject × condition × ROI
table suitable for LMM analysis.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

RELEASE = Path(__file__).resolve().parents[1]
ROI_WIDE = RELEASE / "data" / "roi" / "roi_betas_individual.csv"
RT_WIDE = RELEASE / "data" / "behavioral" / "behavioral_rt.csv"
OUT_DIR = RELEASE / "data" / "brain_behavior" / "roi_beta_rt_lmm_recomputed"
FIG_DIR = RELEASE / "figures" / "source_panels"
REPORT = RELEASE / "reports" / "brain_behavior_roi_beta_rt_lmm_recomputed.md"

COND_ORDER = ["CA", "JX", "OI"]
# Current reporting ROI set: keep ATL correlation results, remove dmPFC.
ROI_ORDER = ["ATL", "L_Angular", "PCC", "vmPFC"]


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    q[valid] = tmp
    return q.tolist()


def p_fmt(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < .001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


def load_and_reshape() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ROI_WIDE.exists():
        raise FileNotFoundError(f"Missing ROI beta file: {ROI_WIDE}")
    if not RT_WIDE.exists():
        raise FileNotFoundError(f"Missing RT file: {RT_WIDE}")

    roi = pd.read_csv(ROI_WIDE)
    rt = pd.read_csv(RT_WIDE)

    roi_subjects = set(roi["subject"])
    rt_subjects = set(rt["subject"])
    if roi_subjects != rt_subjects:
        missing_rt = sorted(roi_subjects - rt_subjects)
        missing_roi = sorted(rt_subjects - roi_subjects)
        raise ValueError(f"Subject mismatch. Missing RT={missing_rt}; missing ROI={missing_roi}")

    wide_rows = []
    long_rows = []
    for _, rr in roi.iterrows():
        sub = rr["subject"]
        rt_row = rt.loc[rt["subject"] == sub].iloc[0]
        for cond in COND_ORDER:
            wide = {"subject": sub, "haiku_type": cond, "RT": float(rt_row[f"RT_{cond}"])}
            for roi_name in ROI_ORDER:
                col = f"{roi_name}_beta_{cond}"
                if col not in roi.columns:
                    raise ValueError(f"Missing ROI column: {col}")
                beta = float(rr[col])
                wide[f"{roi_name}_beta"] = beta
                long_rows.append({
                    "subject": sub,
                    "haiku_type": cond,
                    "roi": roi_name,
                    "RT": wide["RT"],
                    "roi_beta": beta,
                })
            wide_rows.append(wide)

    wide_long = pd.DataFrame(wide_rows).sort_values(["subject", "haiku_type"]).reset_index(drop=True)
    long = pd.DataFrame(long_rows).sort_values(["roi", "subject", "haiku_type"]).reset_index(drop=True)
    wide_long["haiku_type"] = pd.Categorical(wide_long["haiku_type"], COND_ORDER, ordered=True)
    long["haiku_type"] = pd.Categorical(long["haiku_type"], COND_ORDER, ordered=True)
    long["roi"] = pd.Categorical(long["roi"], ROI_ORDER, ordered=True)

    # Standardize beta within each ROI across all subject × condition cells.
    long["roi_beta_z"] = long.groupby("roi", observed=True)["roi_beta"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    # Add z-scored versions to wide table too.
    for roi_name in ROI_ORDER:
        vals = wide_long[f"{roi_name}_beta"]
        wide_long[f"{roi_name}_beta_z"] = (vals - vals.mean()) / vals.std(ddof=0)

    source_manifest = pd.DataFrame([
        {"source": "ROI beta source", "path": str(ROI_WIDE), "n_subjects": roi["subject"].nunique(), "n_rows": len(roi)},
        {"source": "Behavioral RT source", "path": str(RT_WIDE), "n_subjects": rt["subject"].nunique(), "n_rows": len(rt)},
        {"source": "Wide LMM-ready output", "path": str(OUT_DIR / "roi_betas_individual_rt_lmm_wide_subject_condition.csv"), "n_subjects": wide_long["subject"].nunique(), "n_rows": len(wide_long)},
        {"source": "Long LMM-ready output", "path": str(OUT_DIR / "roi_betas_individual_rt_lmm_long.csv"), "n_subjects": long["subject"].nunique(), "n_rows": len(long)},
    ])
    return wide_long, long, source_manifest


def fit_mixedlm(formula: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=data, groups=data["subject"])
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = model.fit(reml=False, method=method, maxiter=2000, disp=False)
                if np.isfinite(res.llf):
                    return res
            except Exception:
                continue
    raise RuntimeError(f"Could not fit model: {formula}")


def lrt(full, reduced, df: int) -> tuple[float, float]:
    chi2 = max(0.0, 2 * (full.llf - reduced.llf))
    return chi2, float(stats.chi2.sf(chi2, df))


def slope_for_condition(res, condition: str) -> tuple[float, float, float, float]:
    params = res.params
    cov = res.cov_params()
    terms = {"roi_beta_z": 1.0}
    if condition != "CA":
        t1 = f"roi_beta_z:C(haiku_type, Treatment(reference='CA'))[T.{condition}]"
        t2 = f"C(haiku_type, Treatment(reference='CA'))[T.{condition}]:roi_beta_z"
        term = t1 if t1 in params.index else t2
        terms[term] = 1.0
    names = list(params.index)
    weights = np.zeros(len(names), dtype=float)
    for name, weight in terms.items():
        if name in names:
            weights[names.index(name)] = weight
    slope = float(sum(params.get(name, 0.0) * weight for name, weight in terms.items()))
    se = float(np.sqrt(weights @ cov.loc[names, names].values @ weights))
    z = float(slope / se) if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return slope, se, z, p


def run_lmms(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lrt_rows = []
    slope_rows = []
    fixed_rows = []
    corr_rows = []

    for roi_name in ROI_ORDER:
        d = long[long["roi"] == roi_name].copy()
        d["haiku_type"] = pd.Categorical(d["haiku_type"], COND_ORDER, ordered=True)
        full = fit_mixedlm("RT ~ roi_beta_z * C(haiku_type, Treatment(reference='CA'))", d)
        additive = fit_mixedlm("RT ~ roi_beta_z + C(haiku_type, Treatment(reference='CA'))", d)
        cond_only = fit_mixedlm("RT ~ C(haiku_type, Treatment(reference='CA'))", d)

        chi2_omni, p_omni = lrt(full, cond_only, 3)
        chi2_int, p_int = lrt(full, additive, 2)
        lrt_rows.extend([
            {"roi": roi_name, "test": "roi_beta_plus_interaction_vs_haiku_type_only", "df": 3, "chi2": chi2_omni, "p": p_omni, "n_cells": len(d), "n_subjects": d["subject"].nunique()},
            {"roi": roi_name, "test": "roi_beta_by_haiku_type_interaction", "df": 2, "chi2": chi2_int, "p": p_int, "n_cells": len(d), "n_subjects": d["subject"].nunique()},
        ])

        for name, val in full.params.items():
            fixed_rows.append({
                "roi": roi_name,
                "term": name,
                "estimate": float(val),
                "se": float(full.bse.get(name, np.nan)),
                "z": float(full.tvalues.get(name, np.nan)),
                "p": float(full.pvalues.get(name, np.nan)),
            })

        for cond in COND_ORDER:
            dd = d[d["haiku_type"] == cond]
            slope, se, z, p = slope_for_condition(full, cond)
            r, rp = stats.pearsonr(dd["roi_beta"], dd["RT"])
            slope_rows.append({
                "roi": roi_name,
                "haiku_type": cond,
                "slope_seconds_per_1sd_roi_beta": slope,
                "se": se,
                "z": z,
                "p": p,
                "n_subjects": dd["subject"].nunique(),
            })
            corr_rows.append({
                "roi": roi_name,
                "haiku_type": cond,
                "n": dd["subject"].nunique(),
                "r": float(r),
                "p": float(rp),
            })

    n_rois = len(ROI_ORDER)
    n_roi_conditions = len(ROI_ORDER) * len(COND_ORDER)

    lrt_df = pd.DataFrame(lrt_rows)
    lrt_df[f"q_within_test_family_{n_rois}"] = np.nan
    for test, idx in lrt_df.groupby("test").groups.items():
        lrt_df.loc[idx, f"q_within_test_family_{n_rois}"] = bh_fdr(lrt_df.loc[idx, "p"].tolist())

    slope_df = pd.DataFrame(slope_rows)
    slope_df[f"q_roi_x_condition_slopes_{n_roi_conditions}"] = bh_fdr(slope_df["p"].tolist())

    corr_df = pd.DataFrame(corr_rows)
    corr_df[f"q_roi_x_condition_corr_{n_roi_conditions}"] = bh_fdr(corr_df["p"].tolist())
    fixed_df = pd.DataFrame(fixed_rows)
    return lrt_df, slope_df, fixed_df, corr_df


def make_plot(long: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"CA": "#5B5F97", "JX": "#D7907B", "OI": "#7AA974"}
    fig, axes = plt.subplots(1, len(ROI_ORDER), figsize=(13.6, 3.7), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for i, roi_name in enumerate(ROI_ORDER):
        ax = axes[i]
        d_roi = long[long["roi"] == roi_name]
        for cond in COND_ORDER:
            d = d_roi[d_roi["haiku_type"] == cond]
            ax.scatter(d["roi_beta"], d["RT"], s=34, alpha=0.75, color=colors[cond], edgecolor="white", linewidth=0.4, label=cond if i == 0 else None)
            if len(d) >= 2:
                slope, intercept, _, _, _ = stats.linregress(d["roi_beta"], d["RT"])
                xs = np.linspace(d["roi_beta"].min(), d["roi_beta"].max(), 50)
                ax.plot(xs, intercept + slope * xs, color=colors[cond], lw=1.8)
        ax.set_title(roi_name.replace("_", " "), fontsize=12, weight="bold")
        ax.axhline(d_roi["RT"].mean(), color="#e0e0e0", lw=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Mean RT to first insight (s)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Original condition ROI beta", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        title="Condition",
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=10,
        title_fontsize=10,
    )
    fig.suptitle("Original condition ROI betas predicting RT", y=1.02, fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "roi_betas_individual_rt_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "roi_betas_individual_rt_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(wide: pd.DataFrame, long: pd.DataFrame, lrt_df: pd.DataFrame, slope_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_subjects = wide["subject"].nunique()
    n_cells = len(long)
    n_subject_conditions = len(wide)
    n_rois = len(ROI_ORDER)
    n_roi_conditions = len(ROI_ORDER) * len(COND_ORDER)
    lrt_q_col = f"q_within_test_family_{n_rois}"
    slope_q_col = f"q_roi_x_condition_slopes_{n_roi_conditions}"
    corr_q_col = f"q_roi_x_condition_corr_{n_roi_conditions}"

    omni = lrt_df[lrt_df["test"] == "roi_beta_plus_interaction_vs_haiku_type_only"]
    interaction = lrt_df[lrt_df["test"] == "roi_beta_by_haiku_type_interaction"]
    sig_omni = omni[omni[lrt_q_col] < .05]
    sig_int = interaction[interaction[lrt_q_col] < .05]
    sig_slopes = slope_df[slope_df[slope_q_col] < .05]
    sig_corr = corr_df[corr_df[corr_q_col] < .05]

    lines = []
    lines.append("# Original ROI beta values per condition predicting RT")
    lines.append("")
    lines.append("## Data construction")
    lines.append("")
    lines.append(
        f"This analysis uses the previous condition-level ROI beta table `{ROI_WIDE}` and merges it with `{RT_WIDE}`. "
        f"The new LMM-ready data contain {n_subjects} participants, {n_subject_conditions} subject × condition rows in wide format, "
        f"and {n_cells} subject × condition × ROI rows in long format."
    )
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"- Wide original-data file: `{OUT_DIR / 'roi_betas_individual_rt_lmm_wide_subject_condition.csv'}`")
    lines.append(f"- Long LMM file: `{OUT_DIR / 'roi_betas_individual_rt_lmm_long.csv'}`")
    lines.append(f"- Source manifest: `{OUT_DIR / 'source_manifest.csv'}`")
    lines.append("")
    lines.append("## LMM method")
    lines.append("")
    lines.append(
        "For each ROI separately, mean RT was modeled as `RT ~ roi_beta_z * haiku_type + (1 | subject)`. "
        "`roi_beta_z` is the original condition-level ROI beta z-scored within ROI across subject × condition cells. "
        f"Omnibus ROI-behavior tests and beta × condition interaction tests were FDR-corrected across the {n_rois} ROIs. "
        f"Condition-specific slopes and simple Pearson correlations were FDR-corrected across the {n_roi_conditions} ROI × condition tests."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Omnibus ROI-behavior LRT")
    lines.append("")
    if len(sig_omni):
        txt = "; ".join(f"{r.roi} (χ²({int(r.df)}) = {r.chi2:.2f}, p {p_fmt(r.p)}, q = {getattr(r, lrt_q_col):.3f})" for r in sig_omni.itertuples())
        lines.append(f"FDR-surviving omnibus ROI-behavior effects: {txt}.")
    else:
        lines.append(f"No omnibus ROI-behavior LRT survived FDR correction across the {n_rois} ROIs.")
    lines.append("")
    for r in omni.itertuples():
        lines.append(f"- {r.roi}: χ²({int(r.df)}) = {r.chi2:.2f}, p {p_fmt(r.p)}, q = {getattr(r, lrt_q_col):.3f}.")
    lines.append("")
    lines.append("### ROI beta × condition interaction LRT")
    lines.append("")
    if len(sig_int):
        txt = "; ".join(f"{r.roi} (χ²({int(r.df)}) = {r.chi2:.2f}, p {p_fmt(r.p)}, q = {getattr(r, lrt_q_col):.3f})" for r in sig_int.itertuples())
        lines.append(f"FDR-surviving interactions: {txt}.")
    else:
        lines.append(f"No ROI beta × condition interaction survived FDR correction across the {n_rois} ROIs.")
    lines.append("")
    for r in interaction.itertuples():
        lines.append(f"- {r.roi}: χ²({int(r.df)}) = {r.chi2:.2f}, p {p_fmt(r.p)}, q = {getattr(r, lrt_q_col):.3f}.")
    lines.append("")
    lines.append("### Condition-specific slopes")
    lines.append("")
    if len(sig_slopes):
        txt = "; ".join(f"{r.roi} {r.haiku_type}: b = {r.slope_seconds_per_1sd_roi_beta:.2f} s/SD, p {p_fmt(r.p)}, q = {getattr(r, slope_q_col):.3f}" for r in sig_slopes.itertuples())
        lines.append(f"FDR-surviving LMM simple slopes: {txt}.")
    else:
        lines.append(f"No LMM simple slope survived FDR correction across {n_roi_conditions} ROI × condition tests.")
    lines.append("")
    for roi in ROI_ORDER:
        lines.append(f"**{roi}.**")
        for r in slope_df[slope_df["roi"] == roi].itertuples():
            lines.append(f" {r.haiku_type}: b = {r.slope_seconds_per_1sd_roi_beta:.2f} s/SD, SE = {r.se:.2f}, z = {r.z:.2f}, p {p_fmt(r.p)}, q = {getattr(r, slope_q_col):.3f};")
        lines.append("")
    lines.append("### Simple condition-wise correlations")
    lines.append("")
    if len(sig_corr):
        txt = "; ".join(f"{r.roi} {r.haiku_type}: r = {r.r:.2f}, p {p_fmt(r.p)}, q = {getattr(r, corr_q_col):.3f}" for r in sig_corr.itertuples())
        lines.append(f"FDR-surviving correlations: {txt}.")
    else:
        lines.append(f"No simple ROI beta–RT correlation survived FDR correction across {n_roi_conditions} ROI × condition tests.")
    lines.append("")
    for roi in ROI_ORDER:
        vals = []
        for r in corr_df[corr_df["roi"] == roi].itertuples():
            vals.append(f"{r.haiku_type}: r = {r.r:.2f}, p {p_fmt(r.p)}, q = {getattr(r, corr_q_col):.3f}")
        lines.append(f"- {roi}: " + "; ".join(vals) + ".")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Using the previous condition-level ROI beta values gives a stronger condition-level signal than the newly derived subject × condition beta-series means. "
        "The strongest results are the original left angular gyrus and ATL CA beta–RT associations. However, the beta × condition interaction tests do not survive FDR, "
        "so the safest wording is that the condition-wise correlations show semantic-region CA brain–behavior associations, rather than claiming a reliable cross-condition difference in slopes."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Script: `{Path(__file__).name}`")
    lines.append(f"- Report: `{REPORT}`")
    lines.append(f"- Figure: `{FIG_DIR / 'roi_betas_individual_rt_scatter.png'}`")
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wide, long, manifest = load_and_reshape()
    lrt_df, slope_df, fixed_df, corr_df = run_lmms(long)

    wide.to_csv(OUT_DIR / "roi_betas_individual_rt_lmm_wide_subject_condition.csv", index=False)
    long.to_csv(OUT_DIR / "roi_betas_individual_rt_lmm_long.csv", index=False)
    manifest.to_csv(OUT_DIR / "source_manifest.csv", index=False)
    lrt_df.to_csv(OUT_DIR / "roi_betas_individual_rt_lmm_lrt_fdr.csv", index=False)
    slope_df.to_csv(OUT_DIR / "roi_betas_individual_rt_lmm_condition_slopes_fdr.csv", index=False)
    fixed_df.to_csv(OUT_DIR / "roi_betas_individual_rt_lmm_fixed_effects.csv", index=False)
    corr_df.to_csv(OUT_DIR / "roi_betas_individual_rt_correlations_fdr.csv", index=False)
    make_plot(long, corr_df)
    write_report(wide, long, lrt_df, slope_df, corr_df)

    code_dir = OUT_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    print(f"Prepared original ROI-beta RT LMM data and report: {OUT_DIR}")


if __name__ == "__main__":
    main()
