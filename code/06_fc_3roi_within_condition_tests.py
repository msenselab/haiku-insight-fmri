#!/usr/bin/env python3
"""Within-condition FC presence tests for the archived 3-ROI FC analysis.

Reads archived condition-specific ROI-to-ROI Pearson r values, filters to the
selected 3-ROI family (PCC, L_Angular, vmPFC), Fisher-z transforms subject-level
r values, and runs one-sample t-tests against zero for each ROI pair x condition.
BH-FDR is applied across the 9 pair x condition tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RELEASE = Path(__file__).resolve().parents[1]
IN_CSV = RELEASE / "data/connectivity/condition_specific_connectivity_legacy_subject_values.csv"
OUT_DIR = RELEASE / "data/connectivity"
OUT_CSV = OUT_DIR / "fc_3roi_within_condition_fisherz_tests.csv"

PAIR_LABELS = {
    tuple(sorted(("PCC", "vmPFC"))): "PCC-vmPFC",
    tuple(sorted(("PCC", "L_Angular"))): "PCC-L_Angular",
    tuple(sorted(("L_Angular", "vmPFC"))): "L_Angular-vmPFC",
}
PAIR_ORDER = ["PCC-vmPFC", "PCC-L_Angular", "L_Angular-vmPFC"]
CONDITION_ORDER = ["CA", "JX", "OI"]


def canonical_pair(roi1: str, roi2: str) -> str | None:
    return PAIR_LABELS.get(tuple(sorted((roi1, roi2))))


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df["pair"] = [canonical_pair(a, b) for a, b in zip(df["roi1"], df["roi2"])]
    df = df[df["pair"].notna()].copy()
    df["z"] = np.arctanh(df["r"].clip(-0.999999, 0.999999))

    rows = []
    for pair in PAIR_ORDER:
        for condition in CONDITION_ORDER:
            x = df[(df["pair"] == pair) & (df["condition"] == condition)].copy()
            z = x["z"].to_numpy(float)
            r = x["r"].to_numpy(float)
            t_stat, p_val = stats.ttest_1samp(z, 0.0, nan_policy="omit")
            n = int(np.isfinite(z).sum())
            rows.append(
                {
                    "pair": pair,
                    "condition": condition,
                    "N": n,
                    "n_trs": ";".join(str(int(v)) for v in sorted(x["n_trs"].dropna().unique())),
                    "mean_r_raw": float(np.nanmean(r)),
                    "sem_r_raw": float(stats.sem(r, nan_policy="omit")),
                    "mean_fisher_z": float(np.nanmean(z)),
                    "sem_fisher_z": float(stats.sem(z, nan_policy="omit")),
                    "backtransformed_mean_r": float(np.tanh(np.nanmean(z))),
                    "t": float(t_stat),
                    "df": n - 1,
                    "p": float(p_val),
                    "dz": float(t_stat / np.sqrt(n)),
                }
            )

    out = pd.DataFrame(rows)
    out["q_3pairs_x_3conditions"] = multipletests(out["p"], method="fdr_bh")[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6g}"))


if __name__ == "__main__":
    main()
