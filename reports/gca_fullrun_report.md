# Legacy full-run 3-ROI GCA with TR=1, deconvolution, and DOI

Subjects requested/processed: 19
Requested/minimal changes relative to the old full-run GCA: three legacy-coordinate ROIs only (PCC, vmPFC, L_Angular), TR verified as 1 s, fMRIPrep confound regression retained, ROI-level Wiener HRF deconvolution, FD/spike censoring for VAR rows, BIC lag selection over 1–4 TR, and three planned full-run ROI pairs.
Changed only: no condition/insight segmentation; each subject contributes one continuous full-run series. The primary interpretable result is the directional contrast DOI = logGC_forward − logGC_reverse. Raw one-direction F tests are kept below only as descriptive/legacy-compatibility checks.

Caveat: fMRIPrep preprocessed BOLD is not automatically nuisance-regressed; fMRIPrep provides confound files. This run therefore keeps explicit nuisance regression before deconvolution/GCA.

## Directional F-statistic group tests
| path            |   n |   mean_F |     sd_F |   pct_subject_directional_p_lt_05 |   t_vs_zero |       p_unc |   p_fdr_10_direction_tests | reject_fdr_10_direction_tests   |
|:----------------|----:|---------:|---------:|----------------------------------:|------------:|------------:|---------------------------:|:--------------------------------|
| PCC→vmPFC       |  19 | 14.6037  | 15.9033  |                           84.2105 |     4.00268 | 0.000834823 |                0.000908447 | True                            |
| vmPFC→PCC       |  19 |  7.75378 |  5.5131  |                           89.4737 |     6.13049 | 8.63659e-06 |                1.72732e-05 | True                            |
| L_Angular→PCC   |  19 |  8.86652 |  6.57874 |                           73.6842 |     5.87472 | 1.4581e-05  |                2.18715e-05 | True                            |
| PCC→L_Angular   |  19 |  2.5915  |  1.72388 |                           52.6316 |     6.5527  | 3.70726e-06 |                1.72732e-05 | True                            |
| L_Angular→vmPFC |  19 | 10.4963  |  7.38009 |                           89.4737 |     6.19943 | 7.51039e-06 |                1.72732e-05 | True                            |
| vmPFC→L_Angular |  19 |  3.22833 |  3.54927 |                           42.1053 |     3.96474 | 0.000908447 |                0.000908447 | True                            |

## Directional contrast summaries (forward minus reverse)
| pair            |   n |   mean_loggc_doi |   median_loggc_doi |   p_loggc_doi_unc |   q_loggc_doi_fdr_3_pairs |   mean_F_forward_minus_reverse |   p_F_forward_minus_reverse_unc |
|:----------------|----:|-----------------:|-------------------:|------------------:|--------------------------:|-------------------------------:|--------------------------------:|
| PCC_vmPFC       |  19 |        0.0185823 |          0.0153106 |       0.0281648   |               0.0281648   |                        6.84989 |                     0.0485746   |
| L_Angular_PCC   |  19 |        0.0222397 |          0.0171105 |       0.00030206  |               0.000453089 |                        6.27502 |                     0.000369201 |
| L_Angular_vmPFC |  19 |        0.0244328 |          0.0267119 |       3.39558e-05 |               0.000101868 |                        7.26798 |                     4.42502e-05 |
