# Dataset: Haiku Insight fMRI Study

> **September 2026 revision staging (not yet published):** the current Google Doc
> manuscript has Figures 1–7 and a different Search/Pre-response analysis from
> this May release. See [`revision_2026_09/README.md`](revision_2026_09/README.md)
> for the live-Doc figure images, selected aggregate result tables and analysis
> code; its 13 model-named unthresholded group t-maps are in the separate
> [`unthresholded_tmaps/`](unthresholded_tmaps/) folder. The material described
> **below** is retained for the older manuscript, not authoritative for the
> September revision. This staging branch must not be pushed before public
> data-sharing and licensing approval.

**Final open-data release matched to `/manuscript/Poetic-Closure.docx`**

This repository contains processed data, result tables, manuscript figures, and
analysis scripts for the finalized manuscript. Raw MRI images and fMRIPrep
single-subject derivatives are not included; the release focuses on data needed
to verify the reported statistics, tables, and figures.

## What changed in the final manuscript sync

The release now includes the analyses and figures added during Siyi's final
analysis pass:

- Figure 1: RT to first insight across haiku types.
- Figure 3: ROI beta–behavior correlations for both RT and mean insight responses per trial.
- Figure 4: current reduced 3-ROI network follow-ups, including condition-window GCA DOI values and Fisher-z functional connectivity.
- Figure 5: reduced 3-ROI PCC-seed RT-modulated behavioral PPI.
- Updated processed CSVs and reports for GCA, FC, brain-behavior, and PPI.

A machine-readable file provenance table is in `MANUSCRIPT_MATCH_MANIFEST.csv`.

## Contents

```text
data_release/
├── README.md
├── MANUSCRIPT_MATCH_MANIFEST.csv
├── MANUSCRIPT_MATCH_NOTES.md
├── code/                         Final/relevant scripts by manuscript section
├── data/
│   ├── behavioral/               RT and insight-behavior tables
│   ├── group_zmaps/              Group Z-statistic maps already in the release
│   ├── wholebrain/clusters/      Cluster tables supporting Table 1
│   ├── roi/                      ROI beta estimates and ROI group statistics
│   ├── brain_behavior/           ROI beta–RT/insight correlation tables
│   ├── granger/                  Full-run and condition-window GCA result tables
│   ├── connectivity/             3-ROI condition FC result tables
│   └── ppi/                      PCC × RT behavioral-PPI result tables
├── figures/                      Exact embedded manuscript images (Figures 1–5)
│   └── source_panels/            Higher-resolution source panels/plot outputs
└── reports/                      Final analysis reports copied from project outputs
```

## Figure-to-file map

| Manuscript figure | Exact embedded image | Main source/result tables |
|---|---|---|
| Figure 1 RT to first insight | `figures/figure1_behavioral_rt.png` | `data/behavioral/figure1_rt_summary.csv`, `data/behavioral/figure1_rt_subject_values.csv` |
| Figure 2 whole-brain + ROI estimates | `figures/figure2_wholebrain_roi.jpeg` | `data/group_zmaps/`, `data/wholebrain/clusters/`, `data/roi/roi_betas_individual.csv`, `data/roi/roi_statistics_full.csv` |
| Figure 3 ROI activity vs RT/insight behavior | `figures/figure3_brain_behavior.jpeg` | `data/brain_behavior/figure3a_roi_beta_rt_correlations_fdr.csv`, `data/brain_behavior/figure3b_roi_beta_insight_correlations_fdr.csv` |
| Figure 4 GCA/FC network analyses | `figures/figure4_network_gca_fc.jpeg` | `data/granger/gca_figure_values.csv`, `data/connectivity/fc_3roi_within_condition_fisherz_tests.csv` |
| Figure 5 PCC × RT PPI | `figures/figure5_ppi_rt_modulation.jpeg` | `data/ppi/ppi_group_rt_modulation_betas_fdr.csv`, `data/ppi/figure5_ppi_rt_modulation_3roi_bar_values.csv` |

The `figures/source_panels/` directory contains higher-resolution PNG/PDF panel
outputs where available. The top-level figure files are the exact images embedded
in the final DOCX.

## Reproducing / verifying key reported results

### Behavioral RT results

```bash
python code/01_behavioral_stats.py
```

Inputs: `data/behavioral/behavioral_rt.csv`.

Reported result: RT differed across CA/JX/OI, F(2, 36) = 4.20, p = .023,
partial eta squared = .19; JX was numerically slower than CA and OI.

### Whole-brain and ROI analyses

- Group z-maps: `data/group_zmaps/`.
- Cluster tables: `data/wholebrain/clusters/`.
- ROI beta table: `data/roi/roi_betas_individual.csv`.
- ROI statistics: `data/roi/roi_statistics_full.csv`.

The manuscript reports whole-brain maps at p < .001 uncorrected with cluster
extent k >= 20, and ROI follow-ups from 6-mm spheres centered on reported peaks.

### Brain-behavior correlations

Final tables:

- RT correlations: `data/brain_behavior/figure3a_roi_beta_rt_correlations_fdr.csv`.
- Mean-insight correlations: `data/brain_behavior/figure3b_roi_beta_insight_correlations_fdr.csv`.

Key RT results: left angular CA r = -.668, p = .002, q = .021; ATL CA
r = -.60, p = .007, q = .041. No ROI × condition insight-frequency
correlation survived FDR.

### GCA and FC network analyses

GCA final tables:

- `data/granger/fullrun_gca_directional_contrasts.csv`
- `data/granger/condition_window_gca_doi_summary.csv`
- `data/granger/condition_window_gca_omnibus.csv`
- `data/granger/gca_figure_values.csv`

FC final tables:

- `data/connectivity/fc_3roi_within_condition_fisherz_tests.csv`
- `data/connectivity/figure4_fc_3roi_condition_bars_values.csv`

Interpretation: full-run GCA DOI values support general temporal asymmetry in the
reduced 3-ROI network, but condition-window GCA and FC do not show reliable
CA/JX/OI condition differences after correction.

### Behavioral PPI

Final tables:

- `data/ppi/ppi_group_rt_modulation_betas_fdr.csv`
- `data/ppi/ppi_group_rt_modulation_contrasts_fdr.csv`
- `data/ppi/ppi_subject_level_rt_modulation_betas.csv`

Figure 5 reports the reduced 3-ROI PCC × RT behavioral-PPI. Negative betas mean
longer RT was associated with weaker PCC-target coupling; equivalently, faster
responses were accompanied by stronger coupling. In the full 6 target × condition
family the effects are trend-level after FDR; the manuscript frames the CA target
family as a focused follow-up.

## Sample

N = 19 participants for the main reported analyses. The standard included sample
is `sub-001`, `sub-002`, `sub-003`, `sub-004`, `sub-006`, `sub-007`, `sub-008`,
`sub-009`, `sub-010`, `sub-011`, `sub-012`, `sub-013`, `sub-014`, `sub-016`,
`sub-018`, `sub-019`, `sub-020`, `sub-021`, and `sub-023`.

## Software

The project analyses were run in Python with Nilearn, pandas, NumPy, SciPy,
statsmodels, and Matplotlib. Some scripts require local fMRIPrep derivatives and
therefore cannot be fully rerun from this public tier alone; for those analyses,
the release includes precomputed subject/group CSVs sufficient to verify the
manuscript statistics.

## Verification status

The release was checked after the final-manuscript sync:

```bash
python -m py_compile code/*.py
python code/01_behavioral_stats.py
python code/01_plot_behavioral_rt_figure.py
python code/02_whole_brain_maps.py
python code/04_brain_behavior_roi_beta_rt.py
python code/04_plot_brain_behavior_figure3.py
python code/05_gca_condition_windows.py
python code/05_plot_gca_condition_30s_bars.py
python code/06_fc_3roi_within_condition_tests.py
python code/06_plot_fc_3roi_condition_bars.py
python code/06_mediation_analysis.py
python code/07_plot_ppi_rt_modulation_3roi_bars.py
```

These checks compile all release scripts and rerun the public-tier analyses that
do not require private raw/fMRIPrep derivatives. Scripts for raw ROI extraction,
legacy full-run GCA, legacy connectivity extraction, and behavioral PPI model
estimation are retained as provenance/rerun scripts; full execution requires a
private derivative checkout. Their manuscript-matching outputs are included as
precomputed CSVs.

`code/02_whole_brain_maps.py` uses `Z_THRESHOLD = 3.09` and
`CLUSTER_THRESHOLD = 20`, matching the final manuscript's p < .001 uncorrected
positive-tail cluster display convention.

## Citation

Chen, S., Nasemann, J., Geyer, T., Kacian, J., Pierides, S., Shi, Z., & Müller,
H. J. (in press). *Poetic Closure in Three Lines: Neural Networks of Haiku
Comprehension*.

Dataset repository: <https://github.com/msenselab/haiku-insight-fmri>

## License

Data and code are released under CC BY 4.0 unless otherwise specified. Please
cite the manuscript when using this dataset.
