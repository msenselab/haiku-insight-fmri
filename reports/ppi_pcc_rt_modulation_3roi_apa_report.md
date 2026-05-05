# PCC-seed behavioral gPPI with RT modulation: reduced 3-ROI search-onset analysis

## Purpose

This report summarizes a focused behavioral generalized psychophysiological interaction (gPPI) analysis testing whether trial-wise response time (RT) modulated PCC-seed coupling with two a priori target regions, vmPFC and left angular gyrus, during haiku comprehension. The analysis was motivated by the idea that RT captures trial-to-trial variation in the time required to reach insight or interpretive closure. Rather than testing only whether coupling differed between haiku conditions, this model asked whether slower versus faster trials showed different PCC coupling within each condition.

The reduced ROI set was used to limit the multiple-comparison burden and to align the PPI analysis with the theoretically relevant network. The PCC was treated as the seed region, and vmPFC and left angular gyrus were retained as target regions. ATL was removed from this focused PCC-seed analysis.

## Method

### Participants

The analysis included 19 participants with usable fMRI and behavioral event data: sub-001, sub-002, sub-003, sub-004, sub-006, sub-007, sub-008, sub-009, sub-010, sub-011, sub-012, sub-013, sub-014, sub-016, sub-018, sub-019, sub-020, sub-021, and sub-023.

### Regions of interest

Three 6-mm spherical ROIs were included:

- PCC: `(0, -52, 26)`
- vmPFC: `(0, 46, -10)`
- left angular gyrus: `(-48, -64, 30)`

The PCC served as the seed ROI. vmPFC and left angular gyrus served as target ROIs. PPI labels such as PCC→vmPFC refer to PCC-seed coupling expressed in the vmPFC target time series; they should not be interpreted as directed causal effects.

### Event timing and behavioral modulation

The model used the search-onset events for the three haiku conditions:

- CA: `search_CA`
- JX: `search_JX`
- OI: `search_OI`

Each condition event was modeled as a fixed 1-s search-onset event. This timing choice was used because RT was entered as an amplitude modulator, not as the modeled duration. A 1-s onset event is not interpreted as a 1-s neural process; after convolution with the canonical hemodynamic response function, it produces a full BOLD response regressor. Using fixed onsets therefore isolates whether trial-wise RT modulates coupling, whereas using RT or 30 s as the event duration would change the modeled time window and partly confound the behavioral modulator with event length.

RT was taken from the event-file `duration` column, which reflects each trial's search/response duration. RT values were z-scored within participant and condition, so the resulting PPI terms tested whether relatively slower versus faster trials within the same participant and condition showed stronger or weaker PCC-seed coupling.

### First-level gPPI model

For each participant, a first-level ROI-to-ROI behavioral gPPI model was fit. The target ROI time series was predicted from the following regressors:

- condition task regressors for CA, JX, and OI;
- condition-specific RT parametric modulators;
- the PCC seed time series;
- baseline condition PPI terms, `PCC × condition`;
- condition-specific behavioral PPI terms, `PCC × RT-within-condition`;
- nuisance confounds, a linear trend, and an intercept.

The primary coefficients of interest were the condition-specific behavioral PPI slopes: `PCC × RT-within-CA`, `PCC × RT-within-JX`, and `PCC × RT-within-OI`, separately for vmPFC and left angular gyrus.

### Group-level inference and correction strategy

Group-level inference was performed using one-sample t-tests on the participant-level behavioral PPI beta estimates. The broader reduced-family analysis corrected across all six target-by-condition tests: two targets × three conditions. Condition-difference contrasts, including CA > JX, CA > OI, CA > mean(JX, OI), and JX > OI, were tested separately and corrected across eight contrasts: two targets × four contrasts.

Because the theoretical question primarily concerned CA-related integration within the PCC-centered network, a more focused planned-family interpretation is also reported for the CA condition only. This focused family corrected across the two a priori targets, vmPFC and left angular gyrus. This narrower correction is only appropriate if CA-specific RT modulation in these two targets is declared as the primary hypothesis. The broader six-test family should be retained for transparency.

## Results

### Quality control

The first-level models were estimable for all 19 participants. Across participant-level models, the maximum absolute design correlation was 0.987, the maximum absolute correlation among RT-PPI regressors was 0.005, and the maximum design condition number was 39.2. Thus, the RT-PPI regressors themselves were not strongly collinear, although the full design matrix contained high correlations among some non-PPI regressors, as expected in event-related fMRI models with multiple convolved task regressors.

### PCC-seed RT modulation within each condition

The broader six-test family did not yield any effect that survived Benjamini-Hochberg FDR correction across all target-by-condition tests. Nevertheless, the CA condition showed the clearest negative RT-modulation pattern in both target regions.

For PCC coupling with left angular gyrus, RT negatively modulated coupling in the CA condition, *t*(18) = -2.79, *p* = .012, *q* = .073, mean beta = -0.0256, SE = 0.0092, *d*<sub>z</sub> = -0.64. The same target also showed a nominal negative RT-modulation effect in the OI condition, *t*(18) = -2.33, *p* = .032, *q* = .090, mean beta = -0.0267, SE = 0.0115, *d*<sub>z</sub> = -0.53. In contrast, the JX condition showed no evidence for RT modulation of PCC–left angular coupling, *t*(18) = 0.33, *p* = .745, *q* = .745, mean beta = 0.0036, SE = 0.0109, *d*<sub>z</sub> = 0.08.

For PCC coupling with vmPFC, the CA condition again showed a nominal negative RT-modulation effect, *t*(18) = -2.15, *p* = .045, *q* = .090, mean beta = -0.0155, SE = 0.0072, *d*<sub>z</sub> = -0.49. Neither JX nor OI showed reliable modulation in vmPFC. The JX effect was near zero, *t*(18) = -0.41, *p* = .690, *q* = .745, mean beta = -0.0026, SE = 0.0065, *d*<sub>z</sub> = -0.09. The OI effect was positive but not significant, *t*(18) = 1.04, *p* = .312, *q* = .468, mean beta = 0.0117, SE = 0.0113, *d*<sub>z</sub> = 0.24.

When the CA condition was treated as the focused primary family and correction was restricted to the two a priori targets, both CA effects survived FDR correction across the two target tests. The CA-related PCC–left angular effect survived with *q* = .024, and the CA-related PCC–vmPFC effect survived with *q* = .045. This focused correction should be interpreted as a planned-family sensitivity result rather than as confirmatory evidence, because the broader six-test correction did not produce FDR-significant effects.

### Condition differences in RT modulation

The condition-difference tests did not provide corrected evidence that the RT-modulation slopes differed reliably between conditions. For vmPFC, the CA > mean(JX, OI) contrast was negative but not significant after correction, *t*(18) = -1.69, *p* = .107, *q* = .215, mean beta = -0.0200, SE = 0.0118, *d*<sub>z</sub> = -0.39. The CA > OI contrast in vmPFC showed a similar but non-significant trend, *t*(18) = -1.76, *p* = .096, *q* = .215, mean beta = -0.0272, SE = 0.0155, *d*<sub>z</sub> = -0.40. The CA > JX contrast was weaker, *t*(18) = -1.13, *p* = .274, *q* = .362, mean beta = -0.0128, SE = 0.0114, *d*<sub>z</sub> = -0.26.

For left angular gyrus, the CA > mean(JX, OI) contrast was also not significant, *t*(18) = -1.03, *p* = .316, *q* = .362, mean beta = -0.0141, SE = 0.0136, *d*<sub>z</sub> = -0.24. The CA > JX contrast was nominally negative but did not survive correction, *t*(18) = -1.69, *p* = .108, *q* = .215, mean beta = -0.0292, SE = 0.0172, *d*<sub>z</sub> = -0.39. The JX > OI contrast showed a nominal positive trend, *t*(18) = 1.86, *p* = .079, *q* = .215, mean beta = 0.0303, SE = 0.0163, *d*<sub>z</sub> = 0.43.

Thus, the condition-difference results did not support a strong claim that CA-specific RT modulation differed reliably from JX or OI after correction.

## Interpretation

The reduced 3-ROI PCC-seed behavioral gPPI produced a coherent but statistically modest pattern. In the CA condition, longer RT was associated with more negative PCC-seed coupling expressed in both left angular gyrus and vmPFC. This pattern was strongest for the left angular target and weaker for vmPFC. The effects survived correction only under the narrow CA-only two-target family, but not under the broader six-test family including all three conditions.

The results are therefore best described as exploratory evidence that CA trials with longer response times may involve reduced PCC coupling with left angular gyrus and vmPFC. Because the between-condition contrasts were not significant, the analysis does not provide strong corrected evidence that this RT-modulation pattern is uniquely larger in CA than in JX or OI. The focused CA-only result may be useful as a sensitivity analysis, but it should not be framed as a definitive condition-specific effect.

## Suggested APA-style manuscript wording

A focused behavioral gPPI analysis tested whether trial-wise RT modulated PCC-seed coupling with vmPFC and left angular gyrus during the search period. Search-onset events were modeled as fixed 1-s events, and trial-wise RT was entered as a condition-specific amplitude modulator after z-scoring within participant and condition. In the broader reduced family correcting across two targets and three conditions, no PCC × RT modulation effect survived FDR correction. However, CA trials showed nominal negative RT modulation in both targets: left angular gyrus, *t*(18) = -2.79, *p* = .012, *q* = .073, and vmPFC, *t*(18) = -2.15, *p* = .045, *q* = .090. When correction was restricted to the planned CA-only family across the two a priori targets, both effects survived FDR correction, left angular gyrus *q* = .024 and vmPFC *q* = .045. Condition-difference contrasts did not survive correction, including the vmPFC CA > mean(JX, OI) contrast, *t*(18) = -1.69, *p* = .107, *q* = .215. These findings suggest that longer CA response times were associated with more negative PCC-seed coupling with left angular gyrus and vmPFC, but the effect should be interpreted cautiously because the broader condition-inclusive family and direct condition-difference contrasts were not significant.

## Reproducibility

Analysis script:

`/dss/studies/fmri-haiku/glm_unified/scripts/run_ppi_pcc_rt_within_condition_search_onset_3roi.py`

Helper tests:

`/dss/studies/fmri-haiku/glm_unified/scripts/test_ppi_pcc_rt_within_condition_search_onset_3roi_helpers.py`

Output folder:

`/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/`

Source result tables:

`/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/ppi_group_rt_modulation_betas_fdr.csv`

`/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/ppi_group_rt_modulation_contrasts_fdr.csv`
