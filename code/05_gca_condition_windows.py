#!/usr/bin/env python3
"""Condition-window GCA public-release wrapper.

The full condition-window GCA pipeline uses private fMRIPrep derivatives and
intermediate full-run deconvolved ROI time series that are not included in this
public Tier-1 release. The manuscript-matching precomputed outputs are included
under ``data/granger/`` and can be inspected directly.

Canonical private-project source script:
  /dss/studies/fmri-haiku/glm_unified/scripts/run_condition_alltimepoints_gca_legacy3roi_tr1_deconv_doi.py

Public precomputed outputs:
  data/granger/condition_window_gca_doi_summary.csv
  data/granger/condition_window_gca_omnibus.csv
  data/granger/condition_window_gca_pairwise_contrasts.csv
  data/granger/condition_window_gca_subject_directional_f.csv
  data/granger/condition_window_gca_segment_qc.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
REQUIRED = [
    RELEASE / "data/granger/condition_window_gca_doi_summary.csv",
    RELEASE / "data/granger/condition_window_gca_omnibus.csv",
    RELEASE / "data/granger/condition_window_gca_pairwise_contrasts.csv",
    RELEASE / "data/granger/condition_window_gca_subject_directional_f.csv",
    RELEASE / "data/granger/condition_window_gca_segment_qc.csv",
]


def main() -> None:
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing public GCA outputs: " + ", ".join(str(p) for p in missing))
    summary = pd.read_csv(REQUIRED[0])
    print("Condition-window GCA public outputs are present.")
    print(f"Summary rows: {len(summary)}")
    cols = [c for c in ["pair", "condition", "n", "beta_loggc_doi", "sem_loggc_doi", "q_9_within_condition_tests"] if c in summary.columns]
    if cols:
        print(summary[cols].to_string(index=False))
    print("\nFull rerun requires private fMRIPrep/intermediate derivatives; use the precomputed CSVs above for public verification.")


if __name__ == "__main__":
    main()
