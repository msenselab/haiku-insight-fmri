# Dataset: Haiku Insight fMRI Study

**"Neural signatures of poetic insight: DMN-reward coupling during haiku comprehension"**

This package contains processed data and analysis code sufficient to reproduce all
statistics, tables, and figures reported in the manuscript. Raw imaging data and
fMRIPrep derivatives are not included (see Data Availability below).

---

## Contents

```
data_release/
├── README.md               This file
├── data/
│   ├── behavioral/         RT and insight-rate data (N=19)
│   ├── group_zmaps/        Whole-brain Z-statistic maps (NIfTI, N=19, sub-005 excluded)
│   ├── roi/                ROI beta estimates and group statistics
│   ├── brain_behavior/     Brain-RT correlation data
│   ├── granger/            Granger causality results
│   ├── connectivity/       Condition-specific functional connectivity
│   └── mediation/          Mediation and path model results
├── figures/                Manuscript figures (Figures 1–5)
└── code/                   Analysis scripts (numbered by manuscript section)
```

---

## Reproducing Each Result

### Behavioral Results (Table in text, RT ANOVA)
**Script:** `code/01_behavioral_stats.py`
**Inputs:** `data/behavioral/behavioral_rt.csv`
**Reproduces:** RT descriptives (CA: M=6.4s, SD=3.0; JX: M=7.2s; OI: M=6.3s),
repeated-measures ANOVA F(2,36)=4.20, p=.023, Bonferroni post-hoc comparisons.

```bash
uv run code/01_behavioral_stats.py
```

---

### Figure 1 & Table 1 — Whole-Brain Activation Maps
**Script:** `code/02_whole_brain_maps.py`
**Inputs:** `data/group_zmaps/*.nii.gz`
**Reproduces:** Figure 1 brain maps (CA>JX, CA>OI contrasts) and cluster table.
Threshold: Z > 2.58 (p < .005), k >= 20 voxels.

```bash
uv run code/02_whole_brain_maps.py
```

**Dependencies:** `nilearn`, `numpy`, `matplotlib`

---

### Figure 2 & Table 2 — ROI Analysis
**Script:** `code/03_roi_analysis.py`  
**Pre-computed data:** `data/roi/roi_betas_individual.csv`, `data/roi/roi_statistics_full.csv`

> **Note:** This script extracts betas from first-level GLM maps (not included in
> Tier 1 release). To verify Table 2 statistics directly from pre-computed data,
> load `roi_statistics_full.csv` — it contains all group means, SEMs, t-statistics,
> and p-values matching Table 2.

Key values verified (Insight phase):
- PCC: CA=0.21±0.43, JX=-0.78±0.41, OI=-0.78±0.51 (CA>JX p<.001, CA>OI p=.008)
- L Angular: CA=1.73±0.32, JX=0.39±0.32 (CA>JX p<.001)
- R ATL: CA=0.91±0.25, JX=0.09±0.28, OI=1.16±0.39
- vmPFC: CA=1.60±0.52, JX=0.05±0.47, OI=0.59±0.47 (all pairwise p<.05)

---

### Figure 3 & Table 3 — Brain-Behavior Correlations
**Pre-computed data:** `data/brain_behavior/rt_correlations_complete.csv`

Contains Pearson r and p-values for each ROI × condition (N=19) and pooled (N=57).
All values verified against manuscript Table 3:
- L Angular × CA: r=-.67, p=.002
- R ATL × CA: r=-.60, p=.007
- ATL pooled: r=-.48, p<.001
- L Angular pooled: r=-.30, p=.023

---

### Figure 4 & Table 4 — Granger Causality
**Script:** `code/04_granger_causality.py`  
**Pre-computed data:** `data/granger/granger_group_final.csv`

> **Note:** This script requires fMRIPrep BOLD timeseries (not included).
> `granger_group_final.csv` contains all values needed to verify Table 4:
> - PCC→vmPFC: mean F=89.9, 95% significant, p<.001
> - vmPFC→PCC: mean F=29.2, 95% significant, p<.001
> - Net GC (PCC→vmPFC minus vmPFC→PCC): tested via paired t-test on
>   `data/granger/granger_all_subjects.csv`

---

### Figure 5 & Table 5 — Functional Connectivity
**Script:** `code/05_connectivity_analysis.py`  
**Pre-computed data:** `data/connectivity/condition_connectivity_comparison.csv`

> **Note:** This script requires fMRIPrep BOLD timeseries (not included).
> Key value verified: PCC-vmPFC CA=.312, JX=.210, OI=.221 (CA>JX p=.048, CA>OI p=.029)

---

### Mediation Analysis (text only, no figure)
**Script:** `code/06_mediation_analysis.py`  
**Pre-computed data:** `data/mediation/mediation_results.csv`, `data/mediation/path_comparison.csv`

> **Note:** This script requires outputs from scripts 03 and 05. Pre-computed CSV
> contains indirect effect CIs confirming null mediation.

---

## Data Availability

| Data | Availability |
|------|-------------|
| Processed data (this package) | OSF / Zenodo (DOI: TBD) |
| fMRIPrep derivatives (52GB) | OpenNeuro (accession: TBD) |
| Raw BIDS data | Available upon reasonable request |

Raw data contain unprocessed MRI scans and have not been fully anonymised for
open release. Requests for raw data should be directed to the corresponding author.

---

## Sample

N = 19 participants (sub-005 excluded: outlier JX reaction time > 3 SD above group mean;
sub-015, sub-017 excluded: incomplete data).

Included subjects: sub-001–004, sub-006–014, sub-016, sub-018–023.

---

## Software Requirements

```
Python >= 3.10
nilearn >= 0.10
numpy
pandas
scipy
matplotlib
statsmodels  # for Granger analysis
```

Install via: `uv sync` (if pyproject.toml is present) or `uv pip install nilearn pandas scipy matplotlib statsmodels`

---

## Contributors

| Name | Role |
|------|------|
| Jan Nasemann | Design, data collection |
| Siyi Chen | Design, data collection, analysis, software, visualization |
| Thomas Geyer | Design, data collection |
| Jim Kacian | Design, data collection |
| Stella Pierides | Design, data collection |
| Hermann J. Müller | Design, analysis, supervision |
| Zhuanghua Shi | Design, data collection, analysis, software, visualization, supervision |

---

## Citation

If you refer to this work, please cite:

> Chen, S., Nasemann, J., Geyer, T., Kacian, J., Pierides, S., Shi, Z., & Müller, H. J. (in press). Poetic Closure in Three Lines: Neural Networks of Haiku Comprehension. 

For the dataset specifically:

> Nasemann, J., Chen, S., Geyer, T., Kacian, J., Pierides, S., Müller, H. J., & Shi, Z. (2026). *Haiku Insight fMRI Dataset* [Data set]. GitHub. https://github.com/msenselab/haiku-insight-fmri

---

## License

Data and code are released under CC BY 4.0. Please cite the manuscript when using
this dataset.
