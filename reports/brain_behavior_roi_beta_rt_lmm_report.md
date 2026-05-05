# Original ROI beta values per condition predicting RT

## Data construction

This analysis uses the previous condition-level ROI beta table `/dss/studies/fmri-haiku/glm_unified/roi_betas_individual.csv` and merges it with `/dss/studies/fmri-haiku/glm_unified/behavioral_rt.csv`. The new LMM-ready data contain 19 participants, 57 subject × condition rows in wide format, and 228 subject × condition × ROI rows in long format.

Outputs:
- Wide original-data file: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/roi_betas_individual_rt_lmm_wide_subject_condition.csv`
- Long LMM file: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/roi_betas_individual_rt_lmm_long.csv`
- Source manifest: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/source_manifest.csv`

## LMM method

For each ROI separately, mean RT was modeled as `RT ~ roi_beta_z * haiku_type + (1 | subject)`. `roi_beta_z` is the original condition-level ROI beta z-scored within ROI across subject × condition cells. Omnibus ROI-behavior tests and beta × condition interaction tests were FDR-corrected across the 4 ROIs. Condition-specific slopes and simple Pearson correlations were FDR-corrected across the 12 ROI × condition tests.

## Results

### Omnibus ROI-behavior LRT

No omnibus ROI-behavior LRT survived FDR correction across the 4 ROIs.

- ATL: χ²(3) = 9.02, p = .029, q = 0.116.
- L_Angular: χ²(3) = 1.45, p = .694, q = 0.830.
- PCC: χ²(3) = 0.88, p = .830, q = 0.830.
- vmPFC: χ²(3) = 1.61, p = .658, q = 0.830.

### ROI beta × condition interaction LRT

No ROI beta × condition interaction survived FDR correction across the 4 ROIs.

- ATL: χ²(2) = 2.71, p = .258, q = 0.925.
- L_Angular: χ²(2) = 0.32, p = .851, q = 0.925.
- PCC: χ²(2) = 0.56, p = .755, q = 0.925.
- vmPFC: χ²(2) = 0.16, p = .925, q = 0.925.

### Condition-specific slopes

FDR-surviving LMM simple slopes: ATL CA: b = -1.17 s/SD, p = .004, q = 0.043; ATL JX: b = -1.09 s/SD, p = .007, q = 0.043.

**ATL.**
 CA: b = -1.17 s/SD, SE = 0.41, z = -2.86, p = .004, q = 0.043;
 JX: b = -1.09 s/SD, SE = 0.40, z = -2.69, p = .007, q = 0.043;
 OI: b = -0.67 s/SD, SE = 0.32, z = -2.10, p = .036, q = 0.142;

**L_Angular.**
 CA: b = -0.36 s/SD, SE = 0.31, z = -1.15, p = .250, q = 0.589;
 JX: b = -0.19 s/SD, SE = 0.31, z = -0.64, p = .523, q = 0.636;
 OI: b = -0.17 s/SD, SE = 0.32, z = -0.52, p = .603, q = 0.657;

**PCC.**
 CA: b = -0.32 s/SD, SE = 0.38, z = -0.85, p = .393, q = 0.589;
 JX: b = -0.26 s/SD, SE = 0.41, z = -0.63, p = .530, q = 0.636;
 OI: b = -0.06 s/SD, SE = 0.24, z = -0.26, p = .797, q = 0.797;

**vmPFC.**
 CA: b = -0.44 s/SD, SE = 0.36, z = -1.20, p = .230, q = 0.589;
 JX: b = -0.41 s/SD, SE = 0.40, z = -1.03, p = .302, q = 0.589;
 OI: b = -0.31 s/SD, SE = 0.36, z = -0.87, p = .384, q = 0.589;

### Simple condition-wise correlations

FDR-surviving correlations: ATL CA: r = -0.60, p = .007, q = 0.041; L_Angular CA: r = -0.67, p = .002, q = 0.021.

- ATL: CA: r = -0.60, p = .007, q = 0.041; JX: r = -0.44, p = .061, q = 0.183; OI: r = -0.41, p = .084, q = 0.202.
- L_Angular: CA: r = -0.67, p = .002, q = 0.021; JX: r = -0.12, p = .632, q = 0.843; OI: r = -0.04, p = .867, q = 0.867.
- PCC: CA: r = -0.08, p = .753, q = 0.867; JX: r = -0.14, p = .564, q = 0.843; OI: r = 0.36, p = .134, q = 0.268.
- vmPFC: CA: r = -0.45, p = .054, q = 0.183; JX: r = -0.13, p = .604, q = 0.843; OI: r = -0.04, p = .859, q = 0.867.

## Interpretation

Using the previous condition-level ROI beta values gives a stronger condition-level signal than the newly derived subject × condition beta-series means. The strongest results are the original left angular gyrus and ATL CA beta–RT associations. However, the beta × condition interaction tests do not survive FDR, so the safest wording is that the condition-wise correlations show semantic-region CA brain–behavior associations, rather than claiming a reliable cross-condition difference in slopes.

## Reproducibility

- Script: `/dss/studies/fmri-haiku/glm_unified/scripts/prepare_roi_betas_individual_rt_lmm_data.py`
- Report: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/ROI_BETAS_INDIVIDUAL_RT_LMM_REPORT.md`
- Figure: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/figures/roi_betas_individual_rt_scatter.png`
