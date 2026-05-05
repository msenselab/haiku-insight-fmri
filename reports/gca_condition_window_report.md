# Condition-specific full-search-window GCA

Variant: CA/JX/OI were split again, but each condition uses all time points from the full 30-s condition/viewing windows anchored at `search_{condition}` onset, after the same run-level nuisance regression, deconvolution, censoring, BIC lag selection, three legacy-coordinate ROIs (PCC, vmPFC, L_Angular), and three planned ROI-pair definitions used in the minimal-revised legacy full-run pipeline. VAR lags are segment-aware and never cross trial/search-epoch boundaries.

## Primary PCC↔vmPFC condition difference
- Complete subjects for PCC↔vmPFC across all three conditions: N = 19.
- Mean logGC DOI (PCC→vmPFC minus vmPFC→PCC): CA = 0.0349, JX = 0.0240, OI = 0.0122.
- Omnibus condition test on DOI: Friedman χ² = 0.000, p = 1.000, q(FDR over 3 pairs) = 1.000; parametric rm-ANOVA F(2, 36) = 1.321, p = .279.

### PCC↔vmPFC within-condition DOI
| condition   |   n |   mean_loggc_doi |   median_loggc_doi |   p_loggc_doi_unc |   q_ttests_9_condition_pair_doi |   p_wilcoxon_loggc_doi_unc |
|:------------|----:|-----------------:|-------------------:|------------------:|-------------------------------------:|---------------------------:|
| CA          |  19 |        0.0348781 |          0.013796  |         0.026683  |                            0.0600367 |                  0.0258217 |
| JX          |  19 |        0.0239613 |          0.0158591 |         0.0623423 |                            0.0935135 |                  0.0798759 |
| OI          |  19 |        0.012161  |          0.0133859 |         0.284978  |                            0.3206    |                  0.312408  |

### PCC↔vmPFC pairwise condition contrasts
| contrast   |   n |   mean_difference |   p_paired_t_unc |   q_pairwise_9_ttests |   p_wilcoxon_unc |
|:-----------|----:|------------------:|-----------------:|---------------------------:|-----------------:|
| CA-JX      |  19 |         0.0109168 |         0.34547  |                   0.485842 |         0.332066 |
| CA-OI      |  19 |         0.0227171 |         0.156528 |                   0.352187 |         0.441334 |
| JX-OI      |  19 |         0.0118003 |         0.439968 |                   0.494964 |         0.515278 |

### PCC↔vmPFC sample sizes
| condition   |   n |   mean_segments |   mean_samples |   mean_usable |   min_usable |   max_usable |
|:------------|----:|----------------:|---------------:|--------------:|-------------:|-------------:|
| CA          |  19 |        15.7895  |        473.684 |       344.632 |          252 |          416 |
| JX          |  19 |        15.4211  |        462.632 |       336     |          232 |          416 |
| OI          |  19 |         9.78947 |        293.684 |       219.421 |          148 |          260 |

## Raw directional F descriptive check for PCC↔vmPFC
Caveat: these are non-negative F-statistic tests against zero and are therefore descriptive/legacy checks, not primary evidence that an edge exists.
| condition   | direction   | path      |   n |   mean_F |   median_F |       p_unc |   q_direction_18_tests |
|:------------|:------------|:----------|----:|---------:|-----------:|------------:|---------------------------:|
| CA          | forward     | PCC→vmPFC |  19 |  6.75843 |    4.50439 | 0.000262892 |                0.000262892 |
| CA          | reverse     | vmPFC→PCC |  19 |  3.10873 |    2.55122 | 3.05599e-06 |                9.43628e-06 |
| JX          | forward     | PCC→vmPFC |  19 |  5.56634 |    5.44738 | 2.75466e-05 |                3.81415e-05 |
| JX          | reverse     | vmPFC→PCC |  19 |  3.21743 |    2.8997  | 1.99257e-06 |                8.96657e-06 |
| OI          | forward     | PCC→vmPFC |  19 |  3.85516 |    3.34539 | 4.31485e-06 |                9.43628e-06 |
| OI          | reverse     | vmPFC→PCC |  19 |  2.95069 |    2.70607 | 8.37391e-05 |                0.000100487 |

## Omnibus condition differences for all planned pairs
| pair            |   n_complete_subjects |   mean_CA |   mean_JX |    mean_OI |   p_friedman_condition_difference |   q_3_pair_friedman |   p_rm_anova_condition_difference |   q_3_pair_rm_anova |
|:----------------|----------------------:|----------:|----------:|-----------:|----------------------------------:|------------------------:|----------------------------------:|------------------------:|
| PCC_vmPFC       |                    19 | 0.0348781 | 0.0239613 | 0.012161   |                          1        |                1        |                         0.27949   |                0.27949  |
| L_Angular_PCC   |                    19 | 0.0164335 | 0.0246969 | 0.00463918 |                          0.331124 |                0.756733 |                         0.0897043 |                0.215654 |
| L_Angular_vmPFC |                    19 | 0.027344  | 0.0289779 | 0.0120687  |                          0.504488 |                0.756733 |                         0.14377   |                0.215654 |

## Files
- Subject-level rows: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi_condition_all_timepoints/subject_level/gca_condition_all_timepoints_subject_directional_f.csv`
- DOI summary: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi_condition_all_timepoints/group_level/gca_condition_all_timepoints_doi_summary.csv`
- Condition omnibus tests: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi_condition_all_timepoints/group_level/gca_condition_all_timepoints_condition_omnibus.csv`
- Segment QC: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi_condition_all_timepoints/segments/gca_condition_all_timepoints_segment_qc.csv`

## Suggested interpretation
Using full 30-s condition/viewing windows, PCC→vmPFC temporal asymmetry does not show a reliable CA/JX/OI difference; this argues against a strong condition-specific GCA difference in the full-window variant.
