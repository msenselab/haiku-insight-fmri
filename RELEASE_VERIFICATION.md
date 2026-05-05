# Release verification log

Date: 2026-05-05

Scope: final manuscript sync for `/manuscript/Poetic-Closure.docx`.

## Checks performed

- Extracted and copied the five exact manuscript-embedded figures to `figures/figure1_*.png/jpeg` through `figures/figure5_*.jpeg`.
- Added manuscript-matching source panels, processed CSVs, and reports for behavioral RT, whole-brain/ROI, brain-behavior, GCA, FC, and PPI results.
- Patched `code/02_whole_brain_maps.py` to use the final manuscript display threshold: `Z_THRESHOLD = 3.09`, `CLUSTER_THRESHOLD = 20`.
- Patched public scripts to use release-relative paths where practical; scripts requiring private fMRIPrep/raw/intermediate derivatives now write to public-release subfolders or clearly document their private-input requirement.
- Compiled all Python scripts with `python -m py_compile code/*.py`.
- Ran public-tier scripts that do not require private raw/fMRIPrep derivatives:
  - `code/01_behavioral_stats.py`
  - `code/01_plot_behavioral_rt_figure.py`
  - `code/02_whole_brain_maps.py`
  - `code/04_brain_behavior_roi_beta_rt.py`
  - `code/04_plot_brain_behavior_figure3.py`
  - `code/05_gca_condition_windows.py`
  - `code/05_plot_gca_condition_30s_bars.py`
  - `code/06_fc_3roi_within_condition_tests.py`
  - `code/06_plot_fc_3roi_condition_bars.py`
  - `code/06_mediation_analysis.py`
  - `code/07_plot_ppi_rt_modulation_3roi_bars.py`
- Verified the five top-level manuscript figure files are readable images.
- Ran a credential-pattern scan excluding `.git` and binary files; no credential-like matches were found.

## Key verification outputs

- Behavioral RT ANOVA reproduced: `F(2, 36) = 4.197`, `p = 0.0230`, partial eta squared `0.1891`.
- FC 3-ROI within-condition Fisher-z table regenerated with 9 rows, all expected pair × condition tests present.
- GCA condition-window public wrapper verified the included 9-row condition DOI summary table.
- Whole-brain map script regenerated `figures/unified_no_outlier_report.html` and contrast-specific figure folders from `data/group_zmaps/`.

## Standalone status

The release is standalone for verifying the manuscript-reported processed results and regenerating public-tier figures/tables from included CSV/NIfTI group maps.

It is not fully standalone for rerunning analyses from raw fMRI time series, because raw MRI, fMRIPrep derivatives, first-level GLM folders, event files, and intermediate deconvolved ROI time series are intentionally not shipped in this public data tier. For those analyses, the release includes:

- provenance/rerun scripts under `code/`, and
- manuscript-matching precomputed outputs under `data/` and `reports/`.

Recommendation before commit: discuss whether this public tier should be described as “standalone processed-data release” rather than “fully standalone raw-to-results reproduction package.”
