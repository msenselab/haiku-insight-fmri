# ROI beta-RT analysis report

## Summary

This report summarizes the ROI analysis linking previous/original condition-level ROI beta values to mean RT. dmPFC was removed, and ATL was retained so its correlation results are shown alongside L_Angular, PCC, and vmPFC.

The mixed-effects omnibus tests did not provide FDR-corrected evidence that ROI beta predicted RT across conditions, nor did they show reliable ROI beta by condition interactions. The simple condition-wise cross-participant correlations showed two FDR-surviving CA associations after correction over 12 ROI×condition tests: L_Angular CA, r = -.67, p = .002, q = .021, and ATL CA, r = -.60, p = .007, q = .041. This supports condition-specific semantic-region CA brain-behavior associations, but not a reliable general ROI-to-RT prediction effect or a reliable difference in slopes across conditions.

## Figure

![Original condition ROI betas predicting RT](figures/roi_betas_individual_rt_scatter.png)

The figure shows participant-level condition means. Points and regression lines are plotted separately for CA, JX, and OI within each ROI, now including ATL. The negative L_Angular CA and ATL CA associations are the simple ROI by condition correlations surviving FDR correction.

## Detailed method

### Data sources

The analysis used the previous condition-level ROI beta table and behavioral RT table:

- ROI beta source: `/dss/studies/fmri-haiku/glm_unified/roi_betas_individual.csv`
- Behavioral RT source: `/dss/studies/fmri-haiku/glm_unified/behavioral_rt.csv`

The source ROI table contains one row per participant and condition-specific ROI beta columns, for example `L_Angular_beta_CA`, `L_Angular_beta_JX`, and `L_Angular_beta_OI`. The behavioral table contains condition-level mean RT columns, `RT_CA`, `RT_JX`, and `RT_OI`.

### ROI set

The analysis includes ATL, L_Angular, PCC, and vmPFC; dmPFC remains excluded.

### Data preparation

For each participant, the three condition-specific RT values were merged with the corresponding condition-specific ROI beta estimates. Two analysis-ready datasets were saved:

- Wide subject by condition file: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/roi_betas_individual_rt_lmm_wide_subject_condition.csv`
- Long subject by condition by ROI file: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/roi_betas_individual_rt_lmm_long.csv`

The wide file contains 57 rows (19 participants by 3 conditions). The long file contains 228 rows (19 participants by 3 conditions by 4 ROIs). ROI beta values were z-scored within each ROI across all subject by condition cells before LMM fitting, so model slopes are expressed as seconds of RT change per 1-SD increase in ROI beta.

### Mixed-effects model

For each ROI separately, RT was modeled as:

```text
RT ~ roi_beta_z * haiku_type + (1 | subject)
```

The model included fixed effects for ROI beta, condition, and their interaction, plus a participant random intercept. The random intercept accounts for stable between-participant differences in overall RT. Models were fit with maximum likelihood using `statsmodels` MixedLM.

Three sets of inferential tests were reported:

1. **Omnibus ROI-behavior LRT:** full model compared with a condition-only model, testing whether ROI beta and its condition interaction improved model fit beyond condition alone.
2. **ROI beta by condition LRT:** full model compared with an additive ROI beta + condition model, testing whether the beta-RT slope differed by condition.
3. **Condition-specific simple slopes:** estimated beta-RT slopes within CA, JX, and OI from the full LMM.

For comparison with the original brain-behavior result, simple Pearson correlations were also computed separately for each ROI by condition cell. These correlations are cross-participant condition-wise associations and do not include a participant random intercept.

### Multiple-comparison correction

Benjamini-Hochberg FDR correction was applied separately to these families:

- Omnibus ROI-behavior LRTs across 4 ROIs.
- ROI beta by condition LRTs across 4 ROIs.
- LMM condition-specific slopes across 12 ROI by condition tests.
- Simple Pearson correlations across 12 ROI by condition tests.

## Results

### Mixed-effects model: omnibus ROI-behavior tests

No omnibus ROI-behavior LRT survived FDR correction across the four ROIs, although ATL was nominal before correction.

| ROI | chi2(df=3) | p | q |
| --- | --- | --- | --- |
| ATL | 9.02 | = .029 | .116 |
| L_Angular | 1.45 | = .694 | .830 |
| PCC | 0.88 | = .830 | .830 |
| vmPFC | 1.61 | = .658 | .830 |

### Mixed-effects model: ROI beta by condition interactions

No ROI beta by condition interaction survived FDR correction across the four ROIs. Thus, the LMM does not support a reliable claim that beta-RT slopes differ between CA, JX, and OI.

| ROI | chi2(df=2) | p | q |
| --- | --- | --- | --- |
| ATL | 2.71 | = .258 | .925 |
| L_Angular | 0.32 | = .851 | .925 |
| PCC | 0.56 | = .755 | .925 |
| vmPFC | 0.16 | = .925 | .925 |

### Mixed-effects model: condition-specific slopes

ATL CA and ATL JX survived FDR correction across the 12 ROI by condition slope tests; other ROI/condition slopes did not survive.

| ROI | Condition | b (s / 1-SD beta) | SE | z | p | q |
| --- | --- | --- | --- | --- | --- | --- |
| ATL | CA | -1.17 | 0.41 | -2.86 | = .004 | .043 |
| ATL | JX | -1.09 | 0.40 | -2.69 | = .007 | .043 |
| ATL | OI | -0.67 | 0.32 | -2.10 | = .036 | .142 |
| L_Angular | CA | -0.36 | 0.31 | -1.15 | = .250 | .589 |
| L_Angular | JX | -0.19 | 0.31 | -0.64 | = .523 | .636 |
| L_Angular | OI | -0.17 | 0.32 | -0.52 | = .603 | .657 |
| PCC | CA | -0.32 | 0.38 | -0.85 | = .393 | .589 |
| PCC | JX | -0.26 | 0.41 | -0.63 | = .530 | .636 |
| PCC | OI | -0.06 | 0.24 | -0.26 | = .797 | .797 |
| vmPFC | CA | -0.44 | 0.36 | -1.20 | = .230 | .589 |
| vmPFC | JX | -0.41 | 0.40 | -1.03 | = .302 | .589 |
| vmPFC | OI | -0.31 | 0.36 | -0.87 | = .384 | .589 |

### Simple condition-wise correlations

The simple condition-wise correlations showed two FDR-surviving CA associations: L_Angular CA and ATL CA.

| ROI | Condition | r | p | q |
| --- | --- | --- | --- | --- |
| ATL | CA | -.60 | = .007 | .041 |
| ATL | JX | -.44 | = .061 | .183 |
| ATL | OI | -.41 | = .084 | .202 |
| L_Angular | CA | -.67 | = .002 | .021 |
| L_Angular | JX | -.12 | = .632 | .843 |
| L_Angular | OI | -.04 | = .867 | .867 |
| PCC | CA | -.08 | = .753 | .867 |
| PCC | JX | -.14 | = .564 | .843 |
| PCC | OI | .36 | = .134 | .268 |
| vmPFC | CA | -.45 | = .054 | .183 |
| vmPFC | JX | -.13 | = .604 | .843 |
| vmPFC | OI | -.04 | = .859 | .867 |

## Interpretation

The mixed-effects model and the simple CA correlations answer related but different questions. The LMM asks whether ROI beta explains RT across all subject by condition cells after accounting for condition effects and each participant's general RT level. Because the participant random intercept absorbs stable between-participant RT differences, the LMM mainly tests whether within-participant condition-level variation in ROI beta tracks RT.

The simple CA correlations ask a narrower question: among participants, are individual differences in semantic-region CA beta associated with individual differences in CA RT? Both L_Angular CA and ATL CA survived FDR correction. Therefore, the safest interpretation is:

> The ROI analysis did not show reliable omnibus mixed-effects evidence that ROI beta generally predicts RT across conditions, nor reliable evidence for beta by condition differences. However, the original condition-level L_Angular and ATL betas in CA showed robust cross-participant associations with CA RT after FDR correction.

This should be framed as condition-specific brain-behavior correlations, not as evidence that ROI activation generally predicts RT in the mixed-effects model.

## Reproducibility

- Analysis script: `/dss/studies/fmri-haiku/glm_unified/scripts/prepare_roi_betas_individual_rt_lmm_data.py`
- Output directory: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm`
- This report: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/report.md`
- Existing full statistical report: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/ROI_BETAS_INDIVIDUAL_RT_LMM_REPORT.md`
- Figure PNG: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/figures/roi_betas_individual_rt_scatter.png`
