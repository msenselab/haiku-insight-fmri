#!/usr/bin/env python3
"""Full-run fMRI-GCA robustness pass using raw non-negative F statistics.

This intentionally keeps the revised pipeline's inputs/transformations unchanged
(subject exclusions, ROIs, nuisance regression, deconvolution, censoring, BIC lag
selection, planned ROI pairs), but replaces insight-locked condition segments with
one continuous full-run segment per subject and summarizes raw directional F
statistics with one-sample t-tests against zero.

Important: the group t-test on non-negative F values is included because it was
requested as a robustness/descriptive pass. It should not be used as primary
path-existence evidence in the manuscript; DOI/logGC directionality remains the
more defensible inferential target.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_revised_gca import (  # noqa: E402
    _fit_ols,
    _var_bic,
    bh_fdr,
    build_confounds,
    deconvolve_dataframe,
    extract_roi_timeseries,
    find_subject_files,
    get_subjects,
    load_config,
    make_lagged_design,
    residualize_and_standardize,
)


def directional_f(
    segments: np.ndarray,
    source_idx: int,
    target_idx: int,
    lag: int,
    censor_segments: np.ndarray | None,
) -> dict[str, Any]:
    """Return directional F, p, logGC and usable rows for X -> Y."""
    y, y_lags, x_lags, _ = make_lagged_design(segments, source_idx, target_idx, lag, censor_segments)
    if len(y) == 0:
        return {"F": np.nan, "p": np.nan, "loggc": np.nan, "usable_rows": 0, "df1": lag, "df2": np.nan}
    rss_r, _, _ = _fit_ols(y, y_lags)
    rss_u, _, df_u = _fit_ols(y, np.column_stack([y_lags, x_lags]))
    if not np.isfinite(rss_r) or not np.isfinite(rss_u) or rss_u <= 0 or df_u <= 0:
        return {"F": np.nan, "p": np.nan, "loggc": np.nan, "usable_rows": len(y), "df1": lag, "df2": df_u}
    F = max(((rss_r - rss_u) / lag) / (rss_u / df_u), 0.0)
    p = float(stats.f.sf(F, lag, df_u))
    loggc = float(np.log(max(rss_r, 1e-300) / max(rss_u, 1e-300)))
    return {"F": float(F), "p": p, "loggc": loggc, "usable_rows": int(len(y)), "df1": int(lag), "df2": int(df_u)}


def prepare_subject_fullrun(subject: str, cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame | None, np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    """Extract, denoise, deconvolve and concatenate full-run ROI series."""
    tr = float(cfg["bold"].get("tr_seconds", 1.0))
    runs, find_warnings = find_subject_files(subject, cfg)
    inclusion = {"subject": subject, "included_input": True, "warnings": ";".join(find_warnings)}
    stability_rows: list[dict[str, Any]] = []
    if not runs:
        inclusion["included_input"] = False
        return None, None, inclusion, stability_rows

    denoised_parts: list[pd.DataFrame] = []
    deconv_parts: list[pd.DataFrame] = []
    censor_parts: list[np.ndarray] = []

    for ri, run in enumerate(runs):
        with run.bold_json.open("r", encoding="utf-8") as f:
            bold_meta = json.load(f)
        json_tr = float(bold_meta.get("RepetitionTime", tr))
        if abs(json_tr - tr) > 1e-6:
            raise ValueError(f"{subject} TR mismatch for {run.bold.name}: json={json_tr}, config={tr}")
        raw = extract_roi_timeseries(
            run,
            cfg["rois"],
            radius=float(cfg["bold"].get("roi_radius_mm", 6)),
            tr=tr,
        )
        confounds_df = pd.read_csv(run.confounds, sep="\t")
        if len(confounds_df) != len(raw):
            raise ValueError(f"{subject} confounds length {len(confounds_df)} != BOLD length {len(raw)} for {run.bold.name}")
        X, censor, _ = build_confounds(
            confounds_df,
            tr=tr,
            fd_threshold=float(cfg.get("censoring", {}).get("fd_threshold_mm", 0.5)),
            motion_columns=cfg.get("confounds", {}).get("motion_24"),
            tissue_columns=cfg.get("confounds", {}).get("tissue", ["white_matter", "csf"]),
            acompcor_columns=cfg.get("confounds", {}).get("acompcor_columns"),
        )
        den = pd.DataFrame(residualize_and_standardize(raw.to_numpy(dtype=float), X), columns=raw.columns)
        dec, qc = deconvolve_dataframe(den, tr=tr)
        qc.insert(0, "run", ri + 1)
        qc.insert(0, "subject", subject)
        qc["bold_file"] = str(run.bold)
        stability_rows.extend(qc.to_dict("records"))
        denoised_parts.append(den)
        deconv_parts.append(dec)
        censor_parts.append(censor)

    denoised = pd.concat(denoised_parts, ignore_index=True)
    deconv = pd.concat(deconv_parts, ignore_index=True)
    censor_all = np.concatenate(censor_parts)

    (output_dir / "roi_timeseries_denoised").mkdir(parents=True, exist_ok=True)
    (output_dir / "roi_timeseries_deconvolved").mkdir(parents=True, exist_ok=True)
    denoised.to_csv(output_dir / "roi_timeseries_denoised" / f"{subject}_roi_denoised.tsv", sep="\t", index=False)
    deconv.to_csv(output_dir / "roi_timeseries_deconvolved" / f"{subject}_roi_deconvolved.tsv", sep="\t", index=False)

    inclusion["n_timepoints"] = int(len(deconv))
    inclusion["n_censored_volumes"] = int(censor_all.sum())
    return deconv, censor_all, inclusion, stability_rows


def run_subject(subject: str, cfg: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    deconv, censor_all, inclusion, stability_rows = prepare_subject_fullrun(subject, cfg, output_dir)
    if deconv is None or censor_all is None:
        return [], [], inclusion, stability_rows

    roi_names = list(deconv.columns)
    segments = deconv.to_numpy(dtype=float)[None, :, :]
    censor_segments = censor_all[None, :]

    subject_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    min_rows = int(cfg.get("gca", {}).get("minimum_usable_rows_after_boundary_and_censoring", 50))

    for pair in cfg.get("planned_pairs", []):
        pair_name = pair["name"]
        src, tgt = pair["forward"]
        src_i, tgt_i = roi_names.index(src), roi_names.index(tgt)
        bic_rows = []
        for lag in cfg.get("gca", {}).get("lag_candidates_tr", [1, 2, 3, 4]):
            bic, n = _var_bic(segments, src_i, tgt_i, int(lag), censor_segments)
            bic_rows.append({"lag": int(lag), "bic": bic, "usable_rows": int(n)})
            lag_rows.append({"subject": subject, "pair": pair_name, "lag": int(lag), "bic": bic, "usable_rows": int(n)})
        finite = [r for r in bic_rows if np.isfinite(r["bic"])]
        if not finite:
            selected_lag = np.nan
            forward = reverse = {"F": np.nan, "p": np.nan, "loggc": np.nan, "usable_rows": 0, "df1": np.nan, "df2": np.nan}
            included = False
            skip_reason = "no_estimable_lag"
        else:
            selected_lag = int(min(finite, key=lambda r: r["bic"])["lag"])
            forward = directional_f(segments, src_i, tgt_i, selected_lag, censor_segments)
            reverse = directional_f(segments, tgt_i, src_i, selected_lag, censor_segments)
            included = bool(
                min(forward["usable_rows"], reverse["usable_rows"]) >= min_rows
                and np.isfinite(forward["F"])
                and np.isfinite(reverse["F"])
            )
            skip_reason = "" if included else "insufficient_rows_or_fit_failed"

        base = {
            "subject": subject,
            "pair": pair_name,
            "forward_source": src,
            "forward_target": tgt,
            "selected_lag": selected_lag,
            "included": included,
            "skip_reason": skip_reason,
            "primary": bool(pair.get("primary", False)),
            "n_timepoints": int(len(deconv)),
        }
        subject_rows.append({
            **base,
            "direction": "forward",
            "source": src,
            "target": tgt,
            "F": forward["F"],
            "p_directional_f": forward["p"],
            "loggc": forward["loggc"],
            "usable_rows": forward["usable_rows"],
            "df1": forward["df1"],
            "df2": forward["df2"],
            "doi_loggc_forward_minus_reverse": forward["loggc"] - reverse["loggc"] if np.isfinite(forward["loggc"]) and np.isfinite(reverse["loggc"]) else np.nan,
            "F_forward_minus_reverse": forward["F"] - reverse["F"] if np.isfinite(forward["F"]) and np.isfinite(reverse["F"]) else np.nan,
        })
        subject_rows.append({
            **base,
            "direction": "reverse",
            "source": tgt,
            "target": src,
            "F": reverse["F"],
            "p_directional_f": reverse["p"],
            "loggc": reverse["loggc"],
            "usable_rows": reverse["usable_rows"],
            "df1": reverse["df1"],
            "df2": reverse["df2"],
            "doi_loggc_forward_minus_reverse": forward["loggc"] - reverse["loggc"] if np.isfinite(forward["loggc"]) and np.isfinite(reverse["loggc"]) else np.nan,
            "F_forward_minus_reverse": forward["F"] - reverse["F"] if np.isfinite(forward["F"]) and np.isfinite(reverse["F"]) else np.nan,
        })
    return subject_rows, lag_rows, inclusion, stability_rows


def group_f_tests(subject_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (pair, direction, source, target), g in subject_df[subject_df["included"] == True].groupby(["pair", "direction", "source", "target"], sort=False):
        vals = g["F"].dropna().to_numpy(dtype=float)
        if len(vals) >= 3 and not np.allclose(vals, vals[0]):
            t_stat, p_unc = stats.ttest_1samp(vals, 0.0)
        elif len(vals) >= 3:
            t_stat, p_unc = np.inf, 0.0 if vals[0] > 0 else (np.nan, np.nan)
        else:
            t_stat, p_unc = np.nan, np.nan
        rows.append({
            "pair": pair,
            "direction": direction,
            "source": source,
            "target": target,
            "path": f"{source}→{target}",
            "n": int(len(vals)),
            "mean_F": float(np.mean(vals)) if len(vals) else np.nan,
            "sd_F": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "median_F": float(np.median(vals)) if len(vals) else np.nan,
            "min_F": float(np.min(vals)) if len(vals) else np.nan,
            "max_F": float(np.max(vals)) if len(vals) else np.nan,
            "pct_subject_directional_p_lt_05": float(np.mean(g["p_directional_f"].to_numpy(dtype=float) < 0.05) * 100) if len(g) else np.nan,
            "t_vs_zero": float(t_stat) if np.isscalar(t_stat) else np.nan,
            "p_unc": float(p_unc) if np.isscalar(p_unc) else np.nan,
        })
    group = pd.DataFrame(rows)
    if not group.empty:
        reject, q = bh_fdr(group["p_unc"].to_numpy(dtype=float))
        group["p_fdr_10_direction_tests"] = q
        group["reject_fdr_10_direction_tests"] = reject

    doi_rows: list[dict[str, Any]] = []
    fwd = subject_df[(subject_df["included"] == True) & (subject_df["direction"] == "forward")].copy()
    for pair, g in fwd.groupby("pair", sort=False):
        doi = g["doi_loggc_forward_minus_reverse"].dropna().to_numpy(dtype=float)
        fdoi = g["F_forward_minus_reverse"].dropna().to_numpy(dtype=float)
        t_doi, p_doi = stats.ttest_1samp(doi, 0.0) if len(doi) >= 3 else (np.nan, np.nan)
        t_fdoi, p_fdoi = stats.ttest_1samp(fdoi, 0.0) if len(fdoi) >= 3 else (np.nan, np.nan)
        doi_rows.append({
            "pair": pair,
            "n": int(len(doi)),
            "mean_loggc_doi": float(np.mean(doi)) if len(doi) else np.nan,
            "median_loggc_doi": float(np.median(doi)) if len(doi) else np.nan,
            "t_loggc_doi_vs_zero": float(t_doi),
            "p_loggc_doi_unc": float(p_doi),
            "mean_F_forward_minus_reverse": float(np.mean(fdoi)) if len(fdoi) else np.nan,
            "median_F_forward_minus_reverse": float(np.median(fdoi)) if len(fdoi) else np.nan,
            "t_F_forward_minus_reverse": float(t_fdoi),
            "p_F_forward_minus_reverse_unc": float(p_fdoi),
        })
    doi_group = pd.DataFrame(doi_rows)
    if not doi_group.empty:
        reject, q = bh_fdr(doi_group["p_loggc_doi_unc"].to_numpy(dtype=float))
        doi_group["p_loggc_doi_fdr_5_pairs"] = q
        doi_group["reject_loggc_doi_fdr_5_pairs"] = reject
    return group, doi_group


def write_report(output_dir: Path, subjects: list[str], group: pd.DataFrame, doi_group: pd.DataFrame) -> None:
    lines = [
        "# Full-run GCA robustness: raw non-negative F-statistic tests",
        "",
        f"Subjects requested/processed: {len(subjects)}",
        "Input/transforms kept from revised GCA config: subject exclusions, 6-mm ROIs, TR=1 s, fMRIPrep confound regression, ROI-level Wiener deconvolution, FD/spike censoring for VAR rows, BIC lag selection over 1–4 TR, and the five planned ROI pairs.",
        "Changed only: no condition/insight segmentation; each subject contributes one continuous full-run series. Group robustness tests use one-sample t-tests of raw directional F-statistics against zero.",
        "",
        "Caveat: because F-statistics are non-negative by construction, these t-tests are descriptive/legacy-compatibility checks, not valid primary evidence that a directed path exists.",
        "",
        "## Directional F-statistic group tests",
    ]
    if group.empty:
        lines.append("No group rows produced.")
    else:
        cols = ["path", "n", "mean_F", "sd_F", "pct_subject_directional_p_lt_05", "t_vs_zero", "p_unc", "p_fdr_10_direction_tests", "reject_fdr_10_direction_tests"]
        lines.append(group[cols].to_markdown(index=False))
    lines.extend(["", "## Directional contrast summaries (forward minus reverse)"])
    if doi_group.empty:
        lines.append("No DOI rows produced.")
    else:
        cols = ["pair", "n", "mean_loggc_doi", "median_loggc_doi", "p_loggc_doi_unc", "p_loggc_doi_fdr_5_pairs", "mean_F_forward_minus_reverse", "p_F_forward_minus_reverse_unc"]
        lines.append(doi_group[cols].to_markdown(index=False))
    (output_dir / "GCA_FULLRUN_FSTAT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    release = Path(__file__).resolve().parents[1]
    default_config = release / "data" / "granger" / "fullrun_gca_config_used.json"
    default_output = release / "data" / "granger" / "fullrun_recomputed"
    p = argparse.ArgumentParser(description="Run full-run raw-F GCA robustness pass. Requires raw/private fMRIPrep derivatives not included in this public tier.")
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--output-dir", default=str(default_output))
    p.add_argument("--subjects", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    for subdir in ["subject_level", "group_level", "qc", "roi_timeseries_denoised", "roi_timeseries_deconvolved"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    cfg_used = dict(cfg)
    cfg_used["runtime"] = {"analysis_variant": "fullrun_raw_nonnegative_F", "output_dir": str(output_dir)}
    (output_dir / "config_used.json").write_text(json.dumps(cfg_used, indent=2), encoding="utf-8")

    subjects = get_subjects(cfg, args.subjects)
    print(f"Running full-run F-stat GCA for {len(subjects)} subject(s): {', '.join(subjects)}")

    all_subject_rows: list[dict[str, Any]] = []
    all_lag_rows: list[dict[str, Any]] = []
    all_inclusions: list[dict[str, Any]] = []
    all_stability_rows: list[dict[str, Any]] = []

    for subject in subjects:
        print(f"[{subject}] processing")
        try:
            sub_rows, lag_rows, inclusion, stability_rows = run_subject(subject, cfg, output_dir)
            all_subject_rows.extend(sub_rows)
            all_lag_rows.extend(lag_rows)
            all_inclusions.append(inclusion)
            all_stability_rows.extend(stability_rows)
        except (OSError, ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"{subject} failed: {exc}")
            all_inclusions.append({"subject": subject, "included_input": False, "warnings": f"processing_failed:{exc}"})

    subject_df = pd.DataFrame(all_subject_rows)
    lag_df = pd.DataFrame(all_lag_rows)
    inclusion_df = pd.DataFrame(all_inclusions)
    stability_df = pd.DataFrame(all_stability_rows)

    subject_df.to_csv(output_dir / "subject_level" / "gca_fullrun_subject_directional_f.csv", index=False)
    lag_df.to_csv(output_dir / "qc" / "fullrun_lag_selection.csv", index=False)
    inclusion_df.to_csv(output_dir / "qc" / "subject_inclusion_fullrun_fstat.csv", index=False)
    stability_df.to_csv(output_dir / "qc" / "roi_deconvolution_stability.csv", index=False)

    group, doi_group = group_f_tests(subject_df) if not subject_df.empty else (pd.DataFrame(), pd.DataFrame())
    group.to_csv(output_dir / "group_level" / "gca_fullrun_group_f_ttests.csv", index=False)
    doi_group.to_csv(output_dir / "group_level" / "gca_fullrun_directional_contrasts.csv", index=False)
    write_report(output_dir, subjects, group, doi_group)

    print(f"Done. Outputs written to {output_dir}")
    if not group.empty:
        print(group[["path", "n", "mean_F", "t_vs_zero", "p_unc", "p_fdr_10_direction_tests"]].to_string(index=False))
    if not doi_group.empty:
        print("\nForward-minus-reverse summaries:")
        print(doi_group[["pair", "n", "mean_loggc_doi", "p_loggc_doi_unc", "mean_F_forward_minus_reverse", "p_F_forward_minus_reverse_unc"]].to_string(index=False))


if __name__ == "__main__":
    main()
