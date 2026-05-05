#!/usr/bin/env python3
"""PCC-seed behavioral gPPI: search-onset condition-specific RT modulation, reduced 3-ROI set.

This implements Siyi's requested behavioral-modulation PPI:
- seed is PCC only;
- targets are vmPFC and L_Angular only (reduced PCC/vmPFC/L_Angular ROI set);
- behavioral modulators are RT/search duration values within each condition;
- regressors are `PCC x RT-within-CA`, `PCC x RT-within-JX`, `PCC x RT-within-OI`.

Important: RT is used as a mean-centered/z-scored trial-wise amplitude modulator,
not as the event duration. Events are anchored to search onset with a fixed 1-s
stick/window before HRF convolution, so this tests whether RT modulates coupling,
rather than whether a longer modeled event window changes coupling.

Outputs:
  /dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from nilearn.maskers import NiftiSpheresMasker
from scipy import stats
from statsmodels.stats.multitest import multipletests

RELEASE = Path(__file__).resolve().parents[1]
BASE_DIR = RELEASE
# Raw/private inputs are not shipped with this public release. Set these env vars
# when rerunning from a private derivative checkout.
FMRIPREP_DIR = Path(os.environ.get("FMRIPREP_DIR", str(BASE_DIR / "derivatives" / "fmriprep")))
EVENTS_DIR = Path(os.environ.get("HAIKU_EVENTS_DIR", str(BASE_DIR / "glm_insight_search" / "events")))
OUTPUT_DIR = BASE_DIR / "data" / "ppi" / "ppi_recomputed_raw_derivatives_required"
SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = OUTPUT_DIR / "code"
TS_DIR = OUTPUT_DIR / "roi_timeseries_preprocessed_bold"
TS_FALLBACK_DIR = BASE_DIR / "data" / "ppi" / "roi_timeseries_preprocessed_bold_private_optional"

EXCLUDED_SUBJECTS = {"sub-005", "sub-015", "sub-017", "sub-022"}
TR = 1.0
ROI_RADIUS_MM = 6
EVENT_DURATION_SECONDS = 1.0
CONDITIONS = ["CA", "JX", "OI"]
CONDITION_TO_EVENT = {cond: f"search_{cond}" for cond in CONDITIONS}
PRIMARY_SEED = "PCC"
TARGETS = ["L_Angular", "vmPFC"]

ROIS = {
    "PCC": (0, -52, 26),
    "vmPFC": (0, 46, -10),
    "L_Angular": (-48, -64, 30),
}

MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
ACOMPCOR_COLS = [f"a_comp_cor_{i:02d}" for i in range(6)]
OTHER_CONFOUND_PREFIXES = ("motion_outlier", "non_steady_state_outlier")
OPTIONAL_CONFOUND_COLS = ["framewise_displacement"]


@dataclass(frozen=True)
class SubjectFiles:
    subject: str
    bold: Path
    json_sidecar: Path
    confounds: Path
    events: Path


def zscore_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = {col: zscore_1d(df[col].to_numpy(dtype=float)) for col in df.columns}
    return pd.DataFrame(data, index=df.index)


def convolve_hrf(neural: np.ndarray, tr: float = TR) -> np.ndarray:
    neural = np.asarray(neural, dtype=float)
    hrf = np.asarray(glover_hrf(t_r=tr, oversampling=1, time_length=32), dtype=float)
    if hrf.sum() != 0:
        hrf = hrf / hrf.sum()
    return np.convolve(neural, hrf, mode="full")[: len(neural)]


def wiener_deconvolve(signal: np.ndarray, tr: float = TR, regularization: float = 0.1) -> np.ndarray:
    y = zscore_1d(np.asarray(signal, dtype=float))
    n = len(y)
    h = np.asarray(glover_hrf(t_r=tr, oversampling=1, time_length=32), dtype=float)
    if h.sum() != 0:
        h = h / h.sum()
    h_pad = np.zeros(n, dtype=float)
    h_pad[: min(len(h), n)] = h[: min(len(h), n)]
    H = np.fft.rfft(h_pad)
    Y = np.fft.rfft(y)
    lam = regularization * np.nanmax(np.abs(H) ** 2)
    neural = np.fft.irfft(Y * np.conj(H) / (np.abs(H) ** 2 + lam), n=n)
    return zscore_1d(np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0))


def build_condition_and_rt_vectors(
    events: pd.DataFrame,
    n_scans: int,
    tr: float = TR,
    event_duration: float = EVENT_DURATION_SECONDS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Build condition sticks and within-condition RT parametric modulators.

    RT comes from the event TSV `duration` column, but is used only as a
    within-condition z-scored amplitude. The modeled event duration is fixed.
    """
    condition_vecs = {cond: np.zeros(n_scans, dtype=float) for cond in CONDITIONS}
    rt_vecs = {cond: np.zeros(n_scans, dtype=float) for cond in CONDITIONS}
    qc_rows: list[dict[str, object]] = []

    for cond in CONDITIONS:
        label = CONDITION_TO_EVENT[cond]
        cond_events = events[events["trial_type"].astype(str) == label].sort_values("onset").copy()
        cond_events["duration"] = pd.to_numeric(cond_events["duration"], errors="coerce")
        cond_events["onset"] = pd.to_numeric(cond_events["onset"], errors="coerce")
        valid_rt = cond_events["duration"].to_numpy(dtype=float)
        rt_z = zscore_1d(valid_rt)
        for trial_i, ((_, row), rt_amp) in enumerate(zip(cond_events.iterrows(), rt_z)):
            onset = float(row["onset"])
            rt_raw = float(row["duration"])
            start = int(math.floor(onset / tr))
            n_dur = int(math.ceil(event_duration / tr))
            end = start + n_dur
            clipped_start = max(start, 0)
            clipped_end = min(end, n_scans)
            included = clipped_end > clipped_start and np.isfinite(rt_amp)
            if included:
                condition_vecs[cond][clipped_start:clipped_end] = 1.0
                rt_vecs[cond][clipped_start:clipped_end] = float(rt_amp)
            qc_rows.append(
                {
                    "condition": cond,
                    "event_label": label,
                    "trial_index": trial_i,
                    "onset": onset,
                    "rt_search_duration": rt_raw,
                    "rt_z_within_condition": float(rt_amp),
                    "used_event_duration": event_duration,
                    "start_idx": start,
                    "end_idx": end,
                    "clipped_start_idx": clipped_start,
                    "clipped_end_idx": clipped_end,
                    "n_samples": max(clipped_end - clipped_start, 0),
                    "included": included,
                    "drop_reason": "" if included else "window_outside_run_or_invalid_rt",
                }
            )
    return condition_vecs, rt_vecs, pd.DataFrame(qc_rows)


def get_subject_files() -> list[SubjectFiles]:
    files: list[SubjectFiles] = []
    for ev in sorted(EVENTS_DIR.glob("sub-*_search_events.tsv")):
        subject = ev.name.split("_search_events.tsv")[0]
        if subject in EXCLUDED_SUBJECTS:
            continue
        func_dir = FMRIPREP_DIR / subject / "func"
        bold = func_dir / f"{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        js = func_dir / f"{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.json"
        conf = func_dir / f"{subject}_task-haiku_desc-confounds_timeseries.tsv"
        if bold.exists() and js.exists() and conf.exists() and ev.exists():
            files.append(SubjectFiles(subject, bold, js, conf, ev))
    return files


def verify_tr(sf: SubjectFiles) -> float:
    with sf.json_sidecar.open("r", encoding="utf-8") as f:
        tr = float(json.load(f).get("RepetitionTime"))
    if abs(tr - TR) > 1e-6:
        raise ValueError(f"{sf.subject}: expected TR={TR}, found {tr} in {sf.json_sidecar}")
    return tr


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")
    events["trial_type"] = events["trial_type"].astype(str)
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    return events


def load_confounds(path: Path, n_scans: int) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    cols: list[str] = []
    for col in MOTION_COLS + ACOMPCOR_COLS + OPTIONAL_CONFOUND_COLS:
        if col in raw.columns:
            cols.append(col)
    cols.extend([c for c in raw.columns if any(c.startswith(prefix) for prefix in OTHER_CONFOUND_PREFIXES)])
    cf = raw[cols].copy() if cols else pd.DataFrame(index=np.arange(len(raw)))
    cf = cf.apply(pd.to_numeric, errors="coerce").bfill().fillna(0.0)
    if len(cf) != n_scans:
        cf = cf.iloc[:n_scans].reindex(range(n_scans)).bfill().fillna(0.0)
    return zscore_columns(cf)


def extract_roi_timeseries(sf: SubjectFiles) -> pd.DataFrame:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    out = TS_DIR / f"{sf.subject}_roi_timeseries_preprocessed_bold.tsv"
    if out.exists():
        return pd.read_csv(out, sep="\t")
    fallback = TS_FALLBACK_DIR / out.name
    if fallback.exists():
        df = pd.read_csv(fallback, sep="\t")
        df.to_csv(out, sep="\t", index=False)
        return df
    masker = NiftiSpheresMasker(
        seeds=list(ROIS.values()),
        radius=ROI_RADIUS_MM,
        standardize=False,
        detrend=False,
        t_r=TR,
    )
    ts = masker.fit_transform(str(sf.bold))
    df = pd.DataFrame(ts, columns=list(ROIS.keys()))
    df.to_csv(out, sep="\t", index=False)
    return df


def build_design(
    seed_signal: np.ndarray,
    condition_vecs: dict[str, np.ndarray],
    rt_vecs: dict[str, np.ndarray],
    confounds: pd.DataFrame,
    tr: float = TR,
) -> pd.DataFrame:
    n = len(seed_signal)
    seed_bold = zscore_1d(seed_signal)
    seed_neural = wiener_deconvolve(seed_bold, tr=tr)
    cols: dict[str, np.ndarray] = {}
    for cond in CONDITIONS:
        cols[f"task_{cond}"] = zscore_1d(convolve_hrf(condition_vecs[cond], tr=tr))
    for cond in CONDITIONS:
        cols[f"rt_{cond}"] = zscore_1d(convolve_hrf(rt_vecs[cond], tr=tr))
    cols["seed"] = seed_bold
    for cond in CONDITIONS:
        cols[f"ppi_task_{cond}"] = zscore_1d(convolve_hrf(seed_neural * condition_vecs[cond], tr=tr))
    for cond in CONDITIONS:
        cols[f"ppi_rt_{cond}"] = zscore_1d(convolve_hrf(seed_neural * rt_vecs[cond], tr=tr))
    design = pd.DataFrame(cols)
    if not confounds.empty:
        design = pd.concat([design, confounds.reset_index(drop=True)], axis=1)
    design["linear_trend"] = zscore_1d(np.linspace(-1, 1, n))
    design["intercept"] = 1.0
    return design.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_ols(y: np.ndarray, X: pd.DataFrame) -> tuple[pd.Series, np.ndarray, float]:
    y = zscore_1d(y)
    Xmat = X.to_numpy(dtype=float)
    pinv = np.linalg.pinv(Xmat)
    beta = pinv @ y
    resid = y - Xmat @ beta
    df_resid = max(Xmat.shape[0] - np.linalg.matrix_rank(Xmat), 1)
    sigma2 = float((resid @ resid) / df_resid)
    cov_beta = sigma2 * np.linalg.pinv(Xmat.T @ Xmat)
    return pd.Series(beta, index=X.columns), cov_beta, sigma2


def design_qc(design: pd.DataFrame) -> dict[str, float]:
    cols = [c for c in design.columns if c != "intercept"]
    X = design[cols].to_numpy(dtype=float)
    if X.shape[1] <= 1:
        return {"max_abs_design_corr": np.nan, "max_abs_rt_ppi_corr": np.nan, "condition_number": np.nan}
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu_indices_from(corr, k=1)
    rt_ppi_names = [f"ppi_rt_{cond}" for cond in CONDITIONS]
    rt_ppi_idx = [cols.index(c) for c in rt_ppi_names if c in cols]
    rt_ppi_corr = corr[np.ix_(rt_ppi_idx, rt_ppi_idx)] if len(rt_ppi_idx) > 1 else np.array([[0.0]])
    rt_ppi_upper = np.triu_indices_from(rt_ppi_corr, k=1)
    return {
        "max_abs_design_corr": float(np.max(np.abs(corr[upper]))) if len(upper[0]) else np.nan,
        "max_abs_rt_ppi_corr": float(np.max(np.abs(rt_ppi_corr[rt_ppi_upper]))) if len(rt_ppi_upper[0]) else np.nan,
        "condition_number": float(np.linalg.cond(design.to_numpy(dtype=float))),
    }


def fdr_column(df: pd.DataFrame, p_col: str, q_col: str, mask: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    valid = out[p_col].notna() if mask is None else (out[p_col].notna() & mask)
    if valid.any():
        out.loc[valid, q_col] = multipletests(out.loc[valid, p_col].to_numpy(), method="fdr_bh")[1]
    return out


def rt_contrast_weights(design_columns: list[str], contrast: str) -> np.ndarray:
    weights = {
        "CA_gt_JX": {"ppi_rt_CA": 1.0, "ppi_rt_JX": -1.0},
        "CA_gt_OI": {"ppi_rt_CA": 1.0, "ppi_rt_OI": -1.0},
        "JX_gt_OI": {"ppi_rt_JX": 1.0, "ppi_rt_OI": -1.0},
        "CA_gt_nonCA": {"ppi_rt_CA": 1.0, "ppi_rt_JX": -0.5, "ppi_rt_OI": -0.5},
    }[contrast]
    w = np.zeros(len(design_columns), dtype=float)
    for name, value in weights.items():
        w[design_columns.index(name)] = value
    return w


def run_subject(sf: SubjectFiles) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    tr = verify_tr(sf)
    roi_ts = extract_roi_timeseries(sf)
    n_scans = len(roi_ts)
    events = load_events(sf.events)
    condition_vecs, rt_vecs, event_qc = build_condition_and_rt_vectors(events, n_scans=n_scans, tr=tr)
    confounds = load_confounds(sf.confounds, n_scans=n_scans)
    design = build_design(roi_ts[PRIMARY_SEED].to_numpy(dtype=float), condition_vecs, rt_vecs, confounds, tr=tr)
    qc = design_qc(design)
    event_counts = {f"n_{cond}_events": int((events["trial_type"] == CONDITION_TO_EVENT[cond]).sum()) for cond in CONDITIONS}
    sample_counts = {f"n_{cond}_samples": int(condition_vecs[cond].sum()) for cond in CONDITIONS}
    rt_summary = {}
    for cond in CONDITIONS:
        cond_rt = event_qc.loc[event_qc["condition"] == cond, "rt_search_duration"].astype(float)
        rt_summary[f"mean_rt_{cond}"] = float(cond_rt.mean())
        rt_summary[f"sd_rt_{cond}"] = float(cond_rt.std(ddof=1))
    design_qc_row = {"subject": sf.subject, "seed": PRIMARY_SEED, **qc, **event_counts, **sample_counts, **rt_summary}

    beta_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for target in TARGETS:
        beta, cov_beta, sigma2 = fit_ols(roi_ts[target].to_numpy(dtype=float), design)
        base = {
            "subject": sf.subject,
            "seed": PRIMARY_SEED,
            "target": target,
            "seed_target": f"{PRIMARY_SEED}_to_{target}",
            "sigma2": sigma2,
            "rank": int(np.linalg.matrix_rank(design.to_numpy(dtype=float))),
            "n_scans": n_scans,
        }
        for cond in CONDITIONS:
            name = f"ppi_rt_{cond}"
            beta_rows.append({**base, "condition": cond, "rt_ppi_beta": float(beta[name])})
        for contrast in ["CA_gt_nonCA", "CA_gt_JX", "CA_gt_OI", "JX_gt_OI"]:
            w = rt_contrast_weights(list(design.columns), contrast)
            value = float(w @ beta.to_numpy(dtype=float))
            se = float(np.sqrt(max(w @ cov_beta @ w, 0.0)))
            contrast_rows.append({**base, "contrast": contrast, "contrast_beta": value, "first_level_se": se, "first_level_t": value / se if se > 0 else np.nan})

    qc_df = pd.concat(
        [event_qc.assign(subject=sf.subject), pd.DataFrame([design_qc_row])],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    return beta_rows, contrast_rows, qc_df


def summarize_group(beta_df: pd.DataFrame, contrast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_rows: list[dict[str, object]] = []
    for (target, condition), g in beta_df.groupby(["target", "condition"]):
        vals = g["rt_ppi_beta"].to_numpy(dtype=float)
        n = int(np.isfinite(vals).sum())
        t, p = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
        beta_rows.append(
            {
                "seed": PRIMARY_SEED,
                "target": target,
                "seed_target": f"{PRIMARY_SEED}_to_{target}",
                "condition": condition,
                "n_subjects": n,
                "mean_beta": float(np.nanmean(vals)),
                "sem_beta": float(stats.sem(vals, nan_policy="omit")),
                "t_vs_zero": float(t),
                "p_vs_zero": float(p),
                "cohens_dz": float(t / np.sqrt(n)) if n and np.isfinite(t) else np.nan,
            }
        )
    beta_summary = pd.DataFrame(beta_rows)
    beta_summary = fdr_column(beta_summary, "p_vs_zero", "q_rt_modulation_6")
    beta_summary = fdr_column(beta_summary, "p_vs_zero", "q_vmPFC_3", beta_summary["target"] == "vmPFC")

    contrast_rows: list[dict[str, object]] = []
    for (target, contrast), g in contrast_df.groupby(["target", "contrast"]):
        vals = g["contrast_beta"].to_numpy(dtype=float)
        n = int(np.isfinite(vals).sum())
        t, p = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
        contrast_rows.append(
            {
                "seed": PRIMARY_SEED,
                "target": target,
                "seed_target": f"{PRIMARY_SEED}_to_{target}",
                "contrast": contrast,
                "n_subjects": n,
                "mean_contrast_beta": float(np.nanmean(vals)),
                "sem_contrast_beta": float(stats.sem(vals, nan_policy="omit")),
                "t_vs_zero": float(t),
                "p_vs_zero": float(p),
                "cohens_dz": float(t / np.sqrt(n)) if n and np.isfinite(t) else np.nan,
            }
        )
    contrast_summary = pd.DataFrame(contrast_rows)
    contrast_summary = fdr_column(contrast_summary, "p_vs_zero", "q_condition_difference_8")
    contrast_summary = fdr_column(contrast_summary, "p_vs_zero", "q_vmPFC_condition_difference_4", contrast_summary["target"] == "vmPFC")
    return beta_summary, contrast_summary


def write_report(subjects: list[str], beta_summary: pd.DataFrame, contrast_summary: pd.DataFrame, qc: pd.DataFrame) -> Path:
    report = OUTPUT_DIR / "PPI_PCC_RT_WITHIN_CONDITION_SEARCH_ONSET_3ROI_REPORT.md"
    n_subjects = len(subjects)
    max_corr = qc["max_abs_design_corr"].dropna().max() if "max_abs_design_corr" in qc else np.nan
    max_rt_ppi_corr = qc["max_abs_rt_ppi_corr"].dropna().max() if "max_abs_rt_ppi_corr" in qc else np.nan
    max_cond = qc["condition_number"].dropna().max() if "condition_number" in qc else np.nan
    sig_rt = beta_summary[beta_summary["q_rt_modulation_6"] < 0.05].copy()
    sig_diff = contrast_summary[contrast_summary["q_condition_difference_8"] < 0.05].copy()

    lines: list[str] = []
    lines.append("# PCC behavioral gPPI: search-onset condition-specific RT modulation, reduced 3-ROI set\n")
    lines.append("## Analysis status\n")
    lines.append(f"- Completed PCC-seed RT-modulation PPI for N={n_subjects} subjects: {', '.join(subjects)}.")
    lines.append(f"- Output folder: `{OUTPUT_DIR}`")
    lines.append(f"- Script: `{SCRIPT_PATH}`")
    lines.append(f"- Code copy: `{CODE_DIR / SCRIPT_PATH.name}`")
    lines.append("\n## Method summary\n")
    lines.append("- Seed: PCC `(0,-52,26)`, 6-mm sphere.")
    lines.append("- Reduced ROI set: PCC seed `(0,-52,26)`, targets vmPFC `(0,46,-10)` and L_Angular `(-48,-64,30)`.")
    lines.append("- Events: `search_CA`, `search_JX`, `search_OI`; report labels are CA/JX/OI.")
    lines.append("- Behavioral modulators: RT/search duration from the event-file `duration` column, z-scored within subject and condition.")
    lines.append("- Timing: fixed 1-s search-onset event for both condition sticks and RT modulators; RT is an amplitude modulator, not the modeled duration.")
    lines.append("- GLM included condition task regressors, RT parametric regressors, PCC seed signal, baseline condition PPI terms `PCC×condition`, RT-modulation terms `PCC×RT-within-condition`, nuisance confounds, linear trend, and intercept.")
    lines.append("- Group inference used one-sample t-tests on subject-level `PCC×RT-within-condition` beta estimates.")
    lines.append("\n## QC snapshot\n")
    lines.append(f"- Max absolute design correlation across subjects: {max_corr:.3f}")
    lines.append(f"- Max absolute correlation among RT-PPI regressors: {max_rt_ppi_corr:.3f}")
    lines.append(f"- Max design condition number: {max_cond:.1f}")
    lines.append("\n## Main tests: PCC × RT-within-condition\n")
    lines.append("| target | condition | N | mean beta | t | p | q over 6 | dz |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in beta_summary.sort_values(["target", "condition"]).iterrows():
        lines.append(
            f"| {r['target']} | {r['condition']} | {int(r['n_subjects'])} | {r['mean_beta']:.4f} | {r['t_vs_zero']:.2f} | {r['p_vs_zero']:.4g} | {r['q_rt_modulation_6']:.4g} | {r['cohens_dz']:.3f} |"
        )
    lines.append("\n## Condition differences in RT modulation\n")
    lines.append("| target | contrast | N | mean beta | t | p | q over 8 | dz |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in contrast_summary.sort_values(["target", "contrast"]).iterrows():
        lines.append(
            f"| {r['target']} | {r['contrast']} | {int(r['n_subjects'])} | {r['mean_contrast_beta']:.4f} | {r['t_vs_zero']:.2f} | {r['p_vs_zero']:.4g} | {r['q_condition_difference_8']:.4g} | {r['cohens_dz']:.3f} |"
        )
    lines.append("\n## FDR survivors\n")
    if sig_rt.empty:
        lines.append("- No PCC×RT-within-condition effect survived BH-FDR over the 6 target×condition tests.")
    else:
        for _, r in sig_rt.sort_values("q_rt_modulation_6").iterrows():
            lines.append(f"- {r['seed_target']} {r['condition']}: mean={r['mean_beta']:.4f}, p={r['p_vs_zero']:.4g}, q={r['q_rt_modulation_6']:.4g}")
    if sig_diff.empty:
        lines.append("- No between-condition RT-modulation contrast survived BH-FDR over the 8 target×contrast tests.")
    else:
        for _, r in sig_diff.sort_values("q_condition_difference_8").iterrows():
            lines.append(f"- {r['seed_target']} {r['contrast']}: mean={r['mean_contrast_beta']:.4f}, p={r['p_vs_zero']:.4g}, q={r['q_condition_difference_8']:.4g}")
    lines.append("\n## Output files\n")
    lines.append("- `ppi_subject_level_rt_modulation_betas.csv` — subject-level PCC×RT-within-condition beta estimates.")
    lines.append("- `ppi_subject_level_rt_modulation_contrasts.csv` — subject-level between-condition RT-modulation contrasts.")
    lines.append("- `ppi_group_rt_modulation_betas_fdr.csv` — group tests for RT-modulation slopes.")
    lines.append("- `ppi_group_rt_modulation_contrasts_fdr.csv` — group tests for condition differences in RT modulation.")
    lines.append("- `ppi_subject_qc.csv` — event/modulator and design-QC information.")
    lines.append("\n## Interpretation note\n")
    lines.append("These effects are behavioral gPPI slopes: they test whether trial-to-trial RT modulates PCC-seed coupling with each target ROI. They are not causal direction estimates.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    subjects_files = get_subject_files()
    if not subjects_files:
        raise RuntimeError("No eligible subjects found for RT-modulation PPI analysis.")

    all_betas: list[dict[str, object]] = []
    all_contrasts: list[dict[str, object]] = []
    qc_frames: list[pd.DataFrame] = []
    included_subjects: list[str] = []
    for sf in subjects_files:
        print(f"Processing {sf.subject}...")
        beta_rows, contrast_rows, qc_df = run_subject(sf)
        all_betas.extend(beta_rows)
        all_contrasts.extend(contrast_rows)
        qc_frames.append(qc_df)
        included_subjects.append(sf.subject)

    beta_df = pd.DataFrame(all_betas)
    contrast_df = pd.DataFrame(all_contrasts)
    qc_df = pd.concat(qc_frames, ignore_index=True, sort=False)
    beta_summary, contrast_summary = summarize_group(beta_df, contrast_df)

    beta_df.to_csv(OUTPUT_DIR / "ppi_subject_level_rt_modulation_betas.csv", index=False)
    contrast_df.to_csv(OUTPUT_DIR / "ppi_subject_level_rt_modulation_contrasts.csv", index=False)
    qc_df.to_csv(OUTPUT_DIR / "ppi_subject_qc.csv", index=False)
    beta_summary.to_csv(OUTPUT_DIR / "ppi_group_rt_modulation_betas_fdr.csv", index=False)
    contrast_summary.to_csv(OUTPUT_DIR / "ppi_group_rt_modulation_contrasts_fdr.csv", index=False)

    report = write_report(included_subjects, beta_summary, contrast_summary, qc_df)
    shutil.copy2(SCRIPT_PATH, CODE_DIR / SCRIPT_PATH.name)

    print(f"\nSaved RT-modulation PPI outputs to: {OUTPUT_DIR}")
    print(f"Report: {report}")
    print("PCC x RT-within-condition effects:")
    for _, r in beta_summary.sort_values(["target", "condition"]).iterrows():
        print(
            f"  PCC->{r['target']} {r['condition']}: mean={r['mean_beta']:.4f}, "
            f"t={r['t_vs_zero']:.2f}, p={r['p_vs_zero']:.4g}, q6={r['q_rt_modulation_6']:.4g}"
        )


if __name__ == "__main__":
    main()
