# PCC behavioral gPPI: search-onset condition-specific RT modulation, reduced 3-ROI set

## Analysis status

- Completed PCC-seed RT-modulation PPI for N=19 subjects: sub-001, sub-002, sub-003, sub-004, sub-006, sub-007, sub-008, sub-009, sub-010, sub-011, sub-012, sub-013, sub-014, sub-016, sub-018, sub-019, sub-020, sub-021, sub-023.
- Output folder: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi`
- Script: `/dss/studies/fmri-haiku/glm_unified/scripts/run_ppi_pcc_rt_within_condition_search_onset_3roi.py`
- Code copy: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/code/run_ppi_pcc_rt_within_condition_search_onset_3roi.py`

## Method summary

- Seed: PCC `(0,-52,26)`, 6-mm sphere.
- Reduced ROI set: PCC seed `(0,-52,26)`, targets vmPFC `(0,46,-10)` and L_Angular `(-48,-64,30)`.
- Events: `search_CA`, `search_JX`, `search_OI`; report labels are CA/JX/OI.
- Behavioral modulators: RT/search duration from the event-file `duration` column, z-scored within subject and condition.
- Timing: fixed 1-s search-onset event for both condition sticks and RT modulators; RT is an amplitude modulator, not the modeled duration.
- GLM included condition task regressors, RT parametric regressors, PCC seed signal, baseline condition PPI terms `PCC×condition`, RT-modulation terms `PCC×RT-within-condition`, nuisance confounds, linear trend, and intercept.
- Group inference used one-sample t-tests on subject-level `PCC×RT-within-condition` beta estimates.

## QC snapshot

- Max absolute design correlation across subjects: 0.987
- Max absolute correlation among RT-PPI regressors: 0.005
- Max design condition number: 39.2

## Main tests: PCC × RT-within-condition

| target | condition | N | mean beta | t | p | q over 6 | dz |
|---|---|---:|---:|---:|---:|---:|---:|
| L_Angular | CA | 19 | -0.0256 | -2.79 | 0.01218 | 0.07308 | -0.639 |
| L_Angular | JX | 19 | 0.0036 | 0.33 | 0.7448 | 0.7448 | 0.076 |
| L_Angular | OI | 19 | -0.0267 | -2.33 | 0.03189 | 0.09035 | -0.534 |
| vmPFC | CA | 19 | -0.0155 | -2.15 | 0.04518 | 0.09035 | -0.494 |
| vmPFC | JX | 19 | -0.0026 | -0.41 | 0.6899 | 0.7448 | -0.093 |
| vmPFC | OI | 19 | 0.0117 | 1.04 | 0.3121 | 0.4682 | 0.239 |

## Condition differences in RT modulation

| target | contrast | N | mean beta | t | p | q over 8 | dz |
|---|---|---:|---:|---:|---:|---:|---:|
| L_Angular | CA_gt_JX | 19 | -0.0292 | -1.69 | 0.1077 | 0.2154 | -0.388 |
| L_Angular | CA_gt_OI | 19 | 0.0011 | 0.07 | 0.9417 | 0.9417 | 0.017 |
| L_Angular | CA_gt_nonCA | 19 | -0.0141 | -1.03 | 0.3164 | 0.3616 | -0.236 |
| L_Angular | JX_gt_OI | 19 | 0.0303 | 1.86 | 0.0794 | 0.2154 | 0.427 |
| vmPFC | CA_gt_JX | 19 | -0.0128 | -1.13 | 0.2737 | 0.3616 | -0.259 |
| vmPFC | CA_gt_OI | 19 | -0.0272 | -1.76 | 0.09613 | 0.2154 | -0.403 |
| vmPFC | CA_gt_nonCA | 19 | -0.0200 | -1.69 | 0.1074 | 0.2154 | -0.389 |
| vmPFC | JX_gt_OI | 19 | -0.0143 | -1.07 | 0.2993 | 0.3616 | -0.245 |

## FDR survivors

- No PCC×RT-within-condition effect survived BH-FDR over the 6 target×condition tests.
- No between-condition RT-modulation contrast survived BH-FDR over the 8 target×contrast tests.

## Output files

- `ppi_subject_level_rt_modulation_betas.csv` — subject-level PCC×RT-within-condition beta estimates.
- `ppi_subject_level_rt_modulation_contrasts.csv` — subject-level between-condition RT-modulation contrasts.
- `ppi_group_rt_modulation_betas_fdr.csv` — group tests for RT-modulation slopes.
- `ppi_group_rt_modulation_contrasts_fdr.csv` — group tests for condition differences in RT modulation.
- `ppi_subject_qc.csv` — event/modulator and design-QC information.

## Interpretation note

These effects are behavioral gPPI slopes: they test whether trial-to-trial RT modulates PCC-seed coupling with each target ROI. They are not causal direction estimates.
