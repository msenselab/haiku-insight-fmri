# Final manuscript/data-release match notes

Updated from `/dss/studies/fmri-haiku/manuscript/Poetic-Closure.docx`.

## Newly added/finalized analyses reflected here

1. Behavioral RT Figure 1 from `analysis_20260430_plot_behavioral_rt_results.py`.
2. Brain-behavior Figure 3 with both RT and mean-insight correlations from
   `plot_brain_behavior_roi_beta_rt_and_insight.py`.
3. Reduced 3-ROI GCA/FC Figure 4:
   - GCA DOI values from `legacy3roi_tr1_deconv_doi` and
     `legacy3roi_tr1_deconv_doi_condition_all_timepoints`.
   - FC within-condition Fisher-z tests from `fc_3roi_within_condition_fisherz_tests.csv`.
4. Reduced 3-ROI PCC × RT behavioral-PPI Figure 5 from
   `ppi_pcc_rt_within_condition_search_onset_3roi`.

## Important interpretation locks

- Whole-brain display threshold is locked to the final DOCX wording: p < .001 uncorrected, k >= 20 voxels. In the public plotting script this is implemented as `Z_THRESHOLD = 3.09` with `two_sided=False` (positive-tail maps) and `CLUSTER_THRESHOLD = 20`.
- PPI seed-target labels are not causal directions.
- GCA is exploratory BOLD temporal predictability/asymmetry, not proof of neural causality.
- Current GCA/FC condition differences are descriptive/null after corrected condition tests.
- Figure files at top level of `figures/` are the exact images embedded in the final DOCX;
  source/high-resolution panels are in `figures/source_panels/`.
