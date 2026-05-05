# Integrated evidence report: brain-behavior coupling, PPI/FC, and GCA

## Executive summary

This report integrates the ROI brain-behavior analyses, trial-wise ROI activation follow-ups, PPI/gPPI analyses, functional connectivity (FC), and Granger-causality analysis (GCA) for the haiku fMRI project. The central question is whether the different analyses form a coherent and reliable account of the neural mechanism linking ROI activity/connectivity to RT.

**Short answer:** the cleanest order is activation/behavior first, behavioral-PPI second, and FC/GCA as contextual connectivity checks. The strongest condition-specific evidence is: (1) higher original semantic-region beta in CA is associated with faster CA RT across participants, with L_Angular r = -.67, p = .002, q = .021 and ATL r = -.60, p = .007, q = .041 after FDR over the 4-ROI x 3-condition correlation family; and (2) in a focused CA-only PCC×RT PPI family, CA RT modulation is significant for L_Angular, p = .012, q = .024, and vmPFC, p = .045, q = .045. Trial-wise ROI activation analyses provide supportive activation-behavior evidence, but condition-level aggregation of those trial-wise estimates is null. Standard gPPI, FC-to-RT, FC-to-insight-count, and condition-specific GCA do not give robust corrected evidence for a broad CA-specific connectivity mechanism.

The most defensible synthesis is:

> CA shows reliable semantic-region activation-RT associations at the condition-wise between-participant level, strongest for L_Angular and ATL. Following this brain-behavior result, focused CA-only behavioral PPI indicates that faster CA responses are accompanied by stronger PCC coupling with L_Angular and vmPFC. FC and GCA provide useful context for stable ROI coupling and general temporal asymmetry, but they do not establish CA-specific causal connectivity.

## Evidence map

| Evidence stream | Primary question | Main result | Reliability | Manuscript-safe interpretation |
|---|---|---|---|---|
| ROI beta-RT brain-behavior | Do condition-level ROI beta values predict RT? | LMM omnibus/interaction null; simple CA correlations survive FDR for L_Angular (r = -.67, q = .021) and ATL (r = -.60, q = .041) | **Moderate** for semantic-region CA correlations; weak for general ROI-to-RT prediction | Main brain-behavior result: original L_Angular/ATL CA activation is associated with faster CA RT across participants; do not claim a reliable cross-condition slope interaction. |
| Trial-wise ROI beta-series activation → RT | Do trial-level ROI responses predict trial-level RT? | vmPFC/PCC omnibus effects survive FDR; strongest simple slope is vmPFC in JX, q = .001; beta × condition interactions are not FDR-significant | **Supportive/suggestive** | Use as converging activation-behavior evidence, but not as the primary condition-level claim; aggregated trial-wise estimates are null. |
| Behavioral PPI, PCC x RT modulation, reduced 3-ROI | As a follow-up to the CA ROI-behavior result, does CA RT modulate PCC coupling with L_Angular/vmPFC? | Focused CA-only family survives FDR over two planned targets: L_Angular q = .024, vmPFC q = .045; full 6 target × condition family remains trend-level | **Moderate as focused follow-up** | Use as hypothesis-driven CA follow-up to ROI-behavior coupling; report full-family correction as sensitivity. |
| Standard gPPI, full 30-s viewing window | Does CA modulate PCC-seed coupling relative to JX/OI? | PCC seed CA > mean(JX, OI) null for vmPFC, L_Angular, ATL; all q = .612 | **Reliable null / limited power** | No reliable evidence that CA increases PCC coupling in a broad condition-modulation PPI. |
| Same-method reduced 3-ROI FC | Is FC present and/or condition-modulated under the current GCA-matched method? | Overall full-run FC is robust for all three pairs; condition-window FC does not differ reliably across CA/JX/OI | **Robust for general FC; null for condition modulation** | Use as evidence that the selected ROIs are generally coupled, not as evidence for a CA-specific FC effect. |
| Same-method FC → RT/insight prediction | Does condition-level FC predict RT or insight count? | No FC main effect or FC × haiku-type interaction survives FDR for RT or mean insights | **Null** | Do not include as a main positive result; useful as a control showing FC does not explain behavior directly. |
| Archived FC plus FDR | Is condition-specific legacy FC stronger for CA? | Old PCC-vmPFC CA > JX p = .048, but q = .287 in legacy 4-ROI family and q = .717 over all 15 pairs | **Descriptive/legacy only** | Retain as historical/descriptive context, not as corrected condition-specific evidence. |
| Minimal-revised full-run GCA | Is there general temporal asymmetry among the reduced 3-ROI network? | All three planned DOI values are positive and FDR-surviving: PCC→vmPFC β=.0186, q=.028; L_Angular→PCC β=.0222, q<.001; L_Angular→vmPFC β=.0244, q<.001 | **Exploratory supportive**, not condition-specific | Supports general full-run temporal predictability/asymmetry among reduced ROIs; do not call this CA-specific or neural causality. |
| Full 30-s condition-window GCA, current method | Do GCA DOI values differ by CA/JX/OI under the same reduced 3-ROI method? | PCC→vmPFC β values are CA=.0349, JX=.0240, OI=.0122, but condition tests are null; all three planned pair omnibus tests q≥.216 | **Descriptive only for condition differences** | Useful figure values, but no reliable CA/JX/OI-specific GCA difference. |

---

## 1. Brain-behavior coupling: ROI beta-RT analysis

### Method summary

The analysis used the original condition-level ROI beta table:

`/dss/studies/fmri-haiku/glm_unified/roi_betas_individual.csv`

and merged it with condition-level behavioral RT:

`/dss/studies/fmri-haiku/glm_unified/behavioral_rt.csv`

dmPFC was removed, but ATL was retained so that Siyi's requested ATL correlation results are shown alongside the other ROIs. The final ROI set was **ATL, L_Angular, PCC, and vmPFC**. The resulting datasets contain 19 participants, 57 subject x condition rows in wide format, and 228 subject x condition x ROI rows in long format.

For each ROI separately, the mixed-effects model was:

```text
RT ~ roi_beta_z * haiku_type + (1 | subject)
```

ROI beta values were z-scored within ROI across subject x condition cells. Omnibus ROI-behavior LRTs and ROI beta x condition LRTs were corrected over 4 ROIs. Condition-specific LMM slopes and simple Pearson correlations were corrected over 12 ROI x condition tests.

### Result

The LMM did not show reliable general ROI-to-RT prediction:

- Omnibus ROI-behavior LRTs: no ROI survived FDR over 4 ROIs; ATL was nominal (p = .029) but not corrected-significant (q = .116).
- ROI beta x condition interactions: no ROI survived FDR over 4 ROIs; all q >= .925.
- LMM condition-specific slopes: ATL CA and ATL JX survived FDR over 12 tests (CA b = -1.17 s/SD, p = .004, q = .043; JX b = -1.09 s/SD, p = .007, q = .043). Other ROI/condition slopes did not survive.

The simple condition-wise correlations showed two FDR-surviving CA associations:

- **L_Angular CA:** r = -.67, p = .002, q = .021.
- **ATL CA:** r = -.60, p = .007, q = .041.

ATL also showed negative but non-FDR-surviving condition-wise correlations for JX (r = -.44, p = .061, q = .183) and OI (r = -.41, p = .084, q = .202).

![ROI beta-RT association including ATL](figures/fig1_brain_behavior_roi_beta_rt.png)

### Interpretation

This is not contradictory. The LMM asks whether ROI beta predicts RT across all subject x condition cells while accounting for condition and each participant's general RT level. The simple CA correlations ask whether participants with stronger semantic-region CA beta also have faster CA RT. The surviving correlation results are therefore best interpreted as **condition-specific between-participant semantic-region brain-behavior associations**, strongest in L_Angular and ATL, not as reliable evidence for a cross-condition slope interaction.

### Trial-wise ROI activation follow-up

We also tested trial-wise insight-locked ROI beta-series activation as a trial-level activation-behavior analysis. This analysis used one ROI beta estimate per insight trial and modeled:

```text
response_time ~ roi_beta_z * haiku_type + (1 | subject)
```

The trial-wise analysis produced stronger activation-behavior evidence than the subject x condition aggregation: vmPFC and PCC omnibus beta/interaction tests survived FDR, and the strongest condition-specific slope was vmPFC in JX (higher vmPFC beta -> faster RT; q = .001). However, beta x haiku-type interactions were not FDR-significant, and when the same trial-wise beta estimates were aggregated to subject x condition means, no omnibus ROI-behavior test, interaction, or simple slope survived FDR.

**Interpretation:** the trial-wise beta-series result is useful converging evidence that ROI activation relates to faster responses, especially vmPFC/PCC. For the manuscript's main condition-specific claim, the original condition-level L_Angular CA association remains the cleaner result.

### Follow-up analysis: focused CA behavioral PPI

Because the ROI-behavior result is CA-specific, the most coherent PPI follow-up is also CA-specific. The reduced behavioral PPI used PCC as the seed and tested whether CA RT modulated PCC coupling with the two planned target ROIs, L_Angular and vmPFC. RT/search duration was modeled as a within-subject x condition amplitude modulator at fixed 1-s search-onset events; RT was not used as event duration.

In the focused CA-only family across the two planned targets, both PCC×RT modulation effects survived FDR correction:

| Target | Mean beta | t | p | q over CA 2-target family | Interpretation |
|---|---:|---:|---:|---:|---|
| L_Angular | -0.0256 | -2.79 | .012 | .024 | Longer CA RT was associated with weaker PCC-L_Angular coupling. |
| vmPFC | -0.0155 | -2.15 | .045 | .045 | Longer CA RT was associated with weaker PCC-vmPFC coupling. |

Equivalently, faster CA responses were accompanied by stronger PCC coupling with L_Angular and vmPFC. This makes the PPI result a useful follow-up to the CA L_Angular beta-RT correlation.

![Focused CA behavioral PPI follow-up: PCC-vmPFC activity and PCC x RT modulation](figures/fig_ca_roi_relations_rt_mediation.png)

Important transparency note: when the family is expanded to all reduced target x condition tests, these same CA effects are trend-level after FDR (L_Angular CA q = .073; vmPFC CA q = .090). Therefore, the focused CA-only PPI should be reported as a hypothesis-driven follow-up family, with the full 6-test target x condition correction reported as a sensitivity check.

---

## 2. PPI/gPPI evidence

### 2.1 Standard ROI-to-ROI gPPI, full 30-s viewing window

The standard gPPI used legacy 4 ROIs: PCC, vmPFC, L_Angular, and ATL. Events were `search_CA`, `search_JX`, and `search_OI`, with manually defined full 30-s viewing windows anchored at search onset. The model included HRF-convolved CA/JX/OI task regressors, the seed BOLD time series, gPPI regressors after canonical-HRF deconvolution/reconvolution, nuisance confounds, trend, and intercept.

Primary planned test: **PCC seed, CA > mean(JX, OI)**.

| Target | N | Mean beta | t | p | q over PCC targets |
|---|---:|---:|---:|---:|---:|
| ATL | 19 | -0.0042 | -0.52 | .610 | .613 |
| L_Angular | 19 | 0.0051 | 0.52 | .613 | .613 |
| vmPFC | 19 | -0.0100 | -1.04 | .313 | .613 |

No full-network primary path survived FDR over 12 seed-target tests. PCC-vmPFC showed a nominal omnibus condition effect, p = .040, but q = .242 over all seed-target paths. Pairwise PCC-vmPFC CA > JX was nominal, p = .031, but q = .197 over 36 pairwise tests. For visualization, the PCC-vmPFC condition-specific gPPI slopes were CA = .109, JX = .134, and OI = .104, with JX showing the steepest estimated coupling slope.

![PCC-vmPFC gPPI seed-target scatter](figures/fig_gppi_pcc_vmpfc_seed_target_scatter_conditions.png)

**Interpretation:** standard gPPI does not support a reliable CA-specific PCC coupling mechanism.

### 2.2 Behavioral PPI: PCC x RT modulation, reduced 3-ROI set

The focused CA behavioral PPI is presented directly after the ROI-behavior correlation because it is the most coherent follow-up test. For transparency, the full reduced target x condition family is summarized here.

![PCC x RT-modulation PPI, reduced 3-ROI set](figures/fig2_ppi_rt_modulation_3roi.png)

Main PCC x RT-within-condition tests in the full reduced family:

| Target | Condition | Mean beta | p | q over 6 |
|---|---|---:|---:|---:|
| L_Angular | CA | -0.0256 | .012 | .073 |
| L_Angular | OI | -0.0267 | .032 | .090 |
| vmPFC | CA | -0.0155 | .045 | .090 |

No PCC x RT-within-condition effect survived FDR over the full 6 target x condition tests, and no between-condition RT-modulation contrast survived FDR over 8 tests. Thus, the PPI is strongest when treated as a focused CA-only follow-up, not as broad evidence for all conditions.

---

## 3. Functional connectivity evidence

The current FC evidence has two roles: (i) showing that the reduced ROIs are generally coupled, and (ii) checking whether FC itself explains RT/insight behavior. The second role is mostly null.

### 3.1 Same-method reduced 3-ROI FC, matched to current GCA preprocessing

The most current FC analysis used the same reduced 3-ROI setup as the current GCA follow-up: PCC, vmPFC, and L_Angular; TR = 1 s; explicit nuisance regression; ROI-level canonical-HRF/Wiener deconvolution; censored-volume exclusion; and Fisher-z FC inference. Overall full-run FC used the uncensored continuous deconvolved ROI series, while condition FC used full 30-s windows anchored at `search_CA`, `search_JX`, and `search_OI`.

Overall full-run FC was robust for all three reduced pairs:

| Pair | Mean Fisher-z | Mean r | q |
|---|---:|---:|---:|
| PCC-vmPFC | .664 | .581 | 1.59e-9 |
| L_Angular-PCC | .512 | .471 | 5.69e-9 |
| L_Angular-vmPFC | .408 | .387 | 4.74e-9 |

However, condition-window FC was positive in each condition but did not show reliable CA/JX/OI modulation: all Friedman q ≈ .949, rm-ANOVA q ≈ .886, and pairwise q ≥ .973.

**Interpretation:** same-method FC supports general coupling among the reduced ROIs, but not a CA-specific FC modulation.

### 3.2 Same-method FC predicting RT and insight count

We then tested whether the same-method condition-level FC predicted mean RT:

```text
RT ~ fc_z * haiku_type + (1 | subject)
```

with 171 rows = 19 subjects x 3 conditions x 3 ROI pairs. No FC main effect or FC x haiku-type interaction survived FDR over the six LRTs. The largest nominal simple slope was L_Angular-vmPFC in CA (slope = -0.562 s per 1-SD FC, p = .069, q = .618), but this was not reliable. Condition-wise FC-RT correlations were also nonsignificant.

The corresponding FC-to-insight-count analysis was also null: no FC main effect, FC x haiku-type interaction, condition-specific simple slope, or condition-wise mean-insight correlation survived FDR.

**Interpretation:** FC itself does not reliably predict RT or insight count. This argues against adding a separate trialwise FC-to-RT analysis unless explicitly requested, because the better trialwise connectivity model is already the RT-modulated behavioral PPI.

### 3.3 Archived old FC plus FDR

The archived old condition-specific FC output was retained only as historical/descriptive context, with FDR correction added while leaving the old values unchanged. This is important: FC values were **not recomputed** in the final retained archived-FC report because recomputing with current event labels/TR/windowing changes the sampled BOLD time points.

Archived old FC procedure:

- ROI time series were extracted using 6-mm spheres and `NiftiSpheresMasker`.
- Signals were standardized, detrended, and band-pass filtered.
- Condition-specific FC was computed as Pearson r over pooled condition time points.
- Old condition-window samples for PCC-vmPFC were fixed at CA = 192, JX = 192, OI = 120 samples per subject.
- Group contrasts used paired tests on raw r values; the update added BH-FDR q-values.

Key retained PCC-vmPFC old result:

| Pair | CA mean r | JX mean r | OI mean r | CA > JX p | q over 6 legacy pairs | q over 15 all old pairs |
|---|---:|---:|---:|---:|---:|---:|
| PCC-vmPFC | .312 | .210 | .221 | .048 | .287 | .717 |

![Archived FC condition comparison](figures/fig3_fc_saved_old_condition_connectivity.png)

**Interpretation:** the old FC values are compatible with positive PCC-vmPFC coupling and a descriptive CA > JX pattern, but the condition contrast is not reliable after FDR correction. The archived FC result should not be used as a main positive claim.

---

## 4. GCA evidence

### Method status and caveats

The revised GCA pipeline improved several major issues relative to the old full-run GCA: TR was verified as 1 s, nuisance regression and deconvolution were applied, condition windows were explicitly defined, VAR lags were segment-aware, and directionality was summarized as **logGC DOI = forward logGC - reverse logGC**. Positive PCC-vmPFC DOI means PCC->vmPFC > vmPFC->PCC.

However, the primary revised insight-locked pipeline had poor residual-whiteness diagnostics: Ljung-Box residual-whiteness passed only 18/285 subject-level models. Thus, GCA should be treated as exploratory/complementary, not as strong neural causality evidence.

![GCA ROI nodes](figures/fig4_gca_roi_nodes_surface.png)

### 4.1 Revised insight-locked condition-specific GCA

Primary window: insight onset + 0-10 s, no additional HRF shift after deconvolution.

For primary PCC-vmPFC DOI:

| Condition | N | Mean DOI | p | q |
|---|---:|---:|---:|---:|
| CA | 19 | -0.0316 | .134 | .401 |
| JX | 19 | -0.0074 | .312 | .469 |
| OI | 17 | 0.0003 | .782 | .782 |

No planned DOI test survived BH-FDR over the 15 condition x pair tests. Window sensitivities at 8 s, 10 s, and 12 s also did not support primary CA PCC-vmPFC DOI.

**Interpretation:** no reliable insight-locked condition-specific GCA evidence.

### 4.2 Minimal-revised full-run GCA, reduced 3-ROI method

The current retained overall GCA uses the minimal-revised legacy-compatible method requested for presentation: 3 legacy-coordinate ROIs only (**PCC**, **vmPFC**, **L_Angular**), TR = 1 s, explicit nuisance regression, HRF/Wiener deconvolution, BIC lag selection over 1-4 TR, and one continuous full-run segment per participant. The primary effect value for figures is **β / mean DOI**, where DOI = forward logGC - reverse logGC.

| Direction / pair | N | β / mean DOI | Median DOI | t | p | q over 3 pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCC → vmPFC | 19 | 0.0186 | 0.0153 | 2.39 | .028 | .028 |
| L_Angular → PCC | 19 | 0.0222 | 0.0171 | 4.46 | 0.000302 | 0.000453 |
| L_Angular → vmPFC | 19 | 0.0244 | 0.0267 | 5.47 | 3.4e-05 | 0.000102 |

**Interpretation:** all three planned full-run DOI values are positive and survive FDR over the reduced 3-pair family. This supports general/full-run temporal asymmetry among the reduced ROIs, especially PCC→vmPFC and L_Angular→PCC/vmPFC, but it is not condition-specific and should be described as temporal predictability/asymmetry rather than neural causality.

### 4.3 Full 30-s condition-window GCA, same current method

The condition split keeps the same current GCA method and changes only the segmentation: all time points from full 30-s windows anchored at `search_CA`, `search_JX`, and `search_OI` are used, with segment-aware VAR rows so lags do not cross trial/window boundaries. The table below gives the condition β values for figure plotting.

| Direction / pair | Condition | N | β / mean DOI | SEM | p within | q over 9 within-condition tests |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L_Angular → PCC | CA | 19 | 0.0164 | 0.0075 | .041 | .073 |
| L_Angular → PCC | JX | 19 | 0.0247 | 0.0071 | .003 | .008 |
| L_Angular → PCC | OI | 19 | 0.0046 | 0.0078 | .561 | .561 |
| L_Angular → vmPFC | CA | 19 | 0.0273 | 0.0068 | 0.000773 | .003 |
| L_Angular → vmPFC | JX | 19 | 0.0290 | 0.0071 | 0.000657 | .003 |
| L_Angular → vmPFC | OI | 19 | 0.0121 | 0.0081 | .155 | .199 |
| PCC → vmPFC | CA | 19 | 0.0349 | 0.0145 | .027 | .060 |
| PCC → vmPFC | JX | 19 | 0.0240 | 0.0121 | .062 | .094 |
| PCC → vmPFC | OI | 19 | 0.0122 | 0.0110 | .285 | .321 |

![Full 30-s condition-window GCA DOI values](figures/fig_gca_condition_30s_bars.png)

Omnibus condition-difference tests on subject-level DOI values:

| Direction / pair | CA β | JX β | OI β | Friedman p | Friedman q | rm-ANOVA F | rm-ANOVA p | rm-ANOVA q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCC → vmPFC | 0.0349 | 0.0240 | 0.0122 | 1.000 | 1.000 | 1.32 | .279 | .279 |
| L_Angular → PCC | 0.0164 | 0.0247 | 0.0046 | .331 | .757 | 2.58 | .090 | .216 |
| L_Angular → vmPFC | 0.0273 | 0.0290 | 0.0121 | .504 | .757 | 2.05 | .144 | .216 |

Pairwise condition contrasts:

| Direction / pair | Contrast | Δβ | p | q over 9 contrasts |
| --- | --- | ---: | ---: | ---: |
| PCC → vmPFC | CA-JX | 0.0109 | .345 | .486 |
| PCC → vmPFC | CA-OI | 0.0227 | .157 | .352 |
| PCC → vmPFC | JX-OI | 0.0118 | .440 | .495 |
| L_Angular → PCC | CA-JX | -0.0083 | .378 | .486 |
| L_Angular → PCC | CA-OI | 0.0118 | .253 | .456 |
| L_Angular → PCC | JX-OI | 0.0201 | .013 | .117 |
| L_Angular → vmPFC | CA-JX | -0.0016 | .859 | .859 |
| L_Angular → vmPFC | CA-OI | 0.0153 | .103 | .309 |
| L_Angular → vmPFC | JX-OI | 0.0169 | .098 | .309 |

**Interpretation:** PCC→vmPFC is numerically largest in CA (β=.0349) and smaller in JX (β=.0240) and OI (β=.0122), but the CA/JX/OI condition difference is not reliable (Friedman p=1.000, q=1.000; rm-ANOVA p=.279, q=.279), and no pairwise contrast survives FDR. The other planned pairs also show no FDR-surviving omnibus condition difference. Thus, these condition β values are useful for the figure, but the manuscript should state that the current GCA supports general/full-run temporal asymmetry, not reliable condition-specific GCA differences.

Figure-ready values are saved here:

`/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/gca_figure_values.csv`

![Current GCA overall and condition β values](figures/fig5_gca_current_overall_condition_betas.png)

---

## 5. Are the results coherent?

Yes, but only under a conservative interpretation.

### Coherent pattern

1. **Primary local activation/behavior coupling:** L_Angular CA beta relates to CA RT across participants. This is the strongest condition-specific brain-behavior result.
2. **Supportive trial-wise activation/behavior evidence:** trial-wise vmPFC/PCC beta-series activation predicts faster RT in some models, but this should remain supportive because condition-level aggregation is null and condition interactions are not robust.
3. **Focused CA PPI follow-up:** when the PPI family is restricted to the hypothesis-relevant CA condition and the two planned PCC targets, CA RT modulation of PCC-L_Angular and PCC-vmPFC coupling survives FDR. This follows naturally from the CA-specific ROI-behavior correlation.
4. **Broader condition-specific connectivity is not robust:** standard gPPI, archived FC condition contrasts, same-method FC condition contrasts, FC-to-RT/insight prediction, the full target x condition PPI family, and condition-specific GCA do not provide strong corrected evidence for a broad CA-specific connectivity mechanism.
5. **General network structure exists:** same-method FC shows robust positive coupling among the reduced ROIs, and full-run GCA suggests possible general temporal asymmetry. These results are compatible with stable DMN-like coordination, but they do not prove CA-specific causal direction.

### What is *not* coherent enough to claim

The current evidence does **not** support a strong chain such as:

> CA increases PCC-vmPFC coupling, which causally drives faster RT through a directional PCC->vmPFC mechanism.

That claim is too strong because PPI is not directional/causal, and FC/GCA do not provide reliable CA-specific corrected evidence for causal direction. The focused PPI can support CA-related PCC coupling as a behavioral follow-up, but not causal flow.

### Better integrated claim

A safer integrated claim is:

> Behavioral facilitation in CA is associated with condition-level semantic-region activation across participants, especially L_Angular and ATL. Following this ROI-behavior association, focused CA behavioral PPI shows that faster CA responses are accompanied by stronger PCC coupling with L_Angular and vmPFC. Broader FC/GCA analyses indicate stable DMN-like coupling and possible general PCC-vmPFC temporal asymmetry, but they do not establish CA-specific causal direction.

---

## 6. Reliability assessment

### Strongest / most usable

- **Semantic-region CA simple brain-behavior correlations:** L_Angular CA (r = -.67, q = .021) and ATL CA (r = -.60, q = .041) survive FDR in the 4 ROI x 3 condition correlation family. Use as the main brain-behavior result, but note these are condition-wise and cross-participant.
- **Focused CA-only behavioral PPI:** survives FDR over the two planned CA PCC targets (L_Angular q = .024; vmPFC q = .045). Use as the main connectivity follow-up to the ROI-behavior result, with the full target x condition family reported as sensitivity.

### Supportive / contextual

- **Trial-wise ROI beta-series activation → RT:** vmPFC/PCC trial-wise activation has FDR-surviving omnibus evidence and vmPFC-JX has the strongest simple slope, but the condition interaction and subject x condition aggregation are not robust. Use as supportive activation-behavior evidence, not the main condition-specific result.
- **Same-method reduced 3-ROI FC:** robust positive overall FC among PCC, vmPFC, and L_Angular. Use as network-context evidence only, because condition modulation and FC-to-behavior prediction are null.
- **Minimal-revised full-run GCA DOI:** all three reduced planned pairs show positive full-run temporal asymmetry after FDR over the 3-pair family (PCC→vmPFC β=.0186, q=.028; L_Angular→PCC β=.0222, q<.001; L_Angular→vmPFC β=.0244, q<.001). This is not condition-specific and should be framed as temporal predictability, not causality.
- **Standard gPPI null result:** no reliable CA > non-CA PCC-seed coupling in the full-viewing-window gPPI. This constrains the mechanistic interpretation and suggests the behavioral PPI is specifically RT-modulation related.

### Weak / descriptive / null

- **Same-method FC → RT and FC → insight count:** no reliable prediction; do not use as positive evidence.
- **Archived FC CA > JX:** p = .048 but not FDR-surviving; retained values are legacy/descriptive.
- **Full target x condition behavioral PPI family:** the CA effects are trend-level after FDR over 6 tests (L_Angular CA q = .073; vmPFC CA q = .090), so the focused-family definition must be stated clearly.
- **Condition-window GCA DOI values:** some within-condition DOI tests are positive, and PCC→vmPFC is numerically CA > JX > OI, but no CA/JX/OI omnibus or pairwise condition difference survives FDR; use these as figure values/descriptives, not as condition-specific GCA evidence.

### Not reliable for main claims

- Raw one-sample tests of non-negative GCA F-statistics.
- Trialwise FC-to-RT analysis made by expanding condition-level FC to trials; that would be pseudo-replication.
- Any claim that PPI/FC/GCA conclusively demonstrates CA-specific PCC-vmPFC coupling or CA-specific directional GCA.
- Any causal wording such as “PCC drives vmPFC” without strong caveats.

---

## 7. Recommended manuscript wording

### Main result wording

> Condition-level brain-behavior analyses showed that stronger semantic-region activation in CA was associated with faster CA RT across participants, with FDR-surviving correlations in left angular gyrus (r = -.67, p = .002, q = .021) and ATL (r = -.60, p = .007, q = .041). These associations did not generalize to a significant omnibus mixed-effects ROI-to-RT prediction effect or a reliable beta × condition interaction across all conditions, indicating that the effects are best treated as condition-specific between-participant associations.

### Focused PPI follow-up wording

> To follow up this CA-specific brain-behavior association, we tested a focused CA-only behavioral PPI family using PCC as the seed and L_Angular/vmPFC as planned targets. PCC×RT modulation was significant for both L_Angular and vmPFC after FDR correction across the two CA targets. The negative modulation estimates indicate that longer CA RT was associated with weaker PCC-target coupling, or equivalently that faster CA responses were accompanied by stronger PCC coupling with L_Angular and vmPFC. The broader target × condition correction is reported as a sensitivity analysis.

### Connectivity/GCA wording

> Complementary connectivity analyses showed a focused CA behavioral-PPI effect but did not establish a broader condition-general or causal mechanism. Standard gPPI was null for the planned PCC-seed CA > non-CA contrast. Same-method reduced-ROI FC showed robust overall coupling among PCC, vmPFC, and L_Angular, but no reliable CA/JX/OI FC modulation and no reliable FC prediction of RT or insight count. The current minimal-revised GCA showed positive full-run temporal asymmetry for the three planned reduced-ROI directions, but the 30-s condition-window GCA did not show reliable CA/JX/OI differences. GCA should therefore be interpreted as exploratory BOLD temporal predictability/asymmetry rather than neural causality.

### Integrated conclusion

> Together, the results support CA-related semantic-region activation-RT associations, strongest in L_Angular and ATL, and a focused CA behavioral-PPI follow-up in which faster CA responses were accompanied by stronger PCC coupling with L_Angular and vmPFC. Trial-wise ROI activation analyses provide supportive activation-behavior evidence, and FC/GCA provide useful exploratory context for broader DMN coordination, but the manuscript should not overstate these analyses as proof of CA-specific causal connectivity.

---

## 8. Source reports and reproducibility

### Brain-behavior

- ROI beta-RT report including ATL correlations: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/report.md`
- ROI beta-RT detailed report including ATL correlations: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/roi_betas_individual_rt_lmm/ROI_BETAS_INDIVIDUAL_RT_LMM_REPORT.md`
- ROI beta-RT script: `/dss/studies/fmri-haiku/glm_unified/scripts/prepare_roi_betas_individual_rt_lmm_data.py`
- Trial-wise ROI beta-series RT report: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/trialwise_insight_beta_rt_lmm/TRIALWISE_ROI_BETA_RT_LMM_APA_REPORT.md`
- Trial-wise ROI beta-series RT script: `/dss/studies/fmri-haiku/glm_unified/scripts/run_trialwise_roi_beta_rt_lmm.py`
- Subject x condition aggregation report: `/dss/studies/fmri-haiku/glm_unified/roi_behavior_analysis/subject_condition_roi_activation_rt_lmm/SUBJECT_CONDITION_ROI_ACTIVATION_RT_LMM_REPORT.md`

### PPI

- Standard selected-ROI gPPI report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_selected_rois/PPI_SELECTED_ROIS_REPORT.md`
- Behavioral PPI reduced 3-ROI report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/ppi_pcc_rt_within_condition_search_onset_3roi/PPI_PCC_RT_WITHIN_CONDITION_SEARCH_ONSET_3ROI_REPORT.md`
- Focused CA PPI note: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/PPI_FOCUSED_CA_FAMILY_NOTE.md`
- Focused CA PPI adapted figure script: `/dss/studies/fmri-haiku/glm_unified/scripts/plot_ca_roi_relations_rt_mediation.py`
- Focused CA PPI adapted figure: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures/fig_ca_roi_relations_rt_mediation.png`
- gPPI PCC-vmPFC seed-target scatter script: `/dss/studies/fmri-haiku/glm_unified/scripts/plot_gppi_pcc_vmpfc_seed_target_scatter_conditions.py`
- gPPI PCC-vmPFC seed-target scatter figure: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures/fig_gppi_pcc_vmpfc_seed_target_scatter_conditions.png`

### FC

- Same-method 3-ROI FC report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/fc_same_method_as_gca_3roi/FC_SAME_METHOD_AS_GCA_3ROI_REPORT.md`
- Same-method 3-ROI FC script: `/dss/studies/fmri-haiku/glm_unified/scripts/run_fc_same_method_as_gca_3roi.py`
- Same-method FC-to-RT report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/fc_same_method_as_gca_3roi_rt_lmm/FC_SAME_METHOD_3ROI_RT_LMM_REPORT.md`
- Same-method FC-to-RT script: `/dss/studies/fmri-haiku/glm_unified/scripts/run_fc_same_method_3roi_rt_lmm.py`
- Same-method FC-to-insight-count report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/fc_same_method_as_gca_3roi_insight_count_lmm/FC_SAME_METHOD_3ROI_INSIGHT_COUNT_LMM_REPORT.md`
- Same-method FC-to-insight-count script: `/dss/studies/fmri-haiku/glm_unified/scripts/run_fc_same_method_3roi_insight_count_lmm.py`
- Archived FC + FDR report: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/saved_old_fc_with_fdr/FC_SAVED_OLD_VALUES_WITH_FDR_REPORT.md`
- Archived-FC companion PCC-vmPFC activity scatter script: `/dss/studies/fmri-haiku/glm_unified/scripts/plot_archived_old_fc_pcc_vmpfc_activity_scatter.py`
- Archived-FC companion PCC-vmPFC activity scatter figure: `/dss/studies/fmri-haiku/glm_unified/connectivity_analysis/saved_old_fc_with_fdr/figures/fig_archived_old_fc_pcc_vmpfc_activity_scatter_by_condition.png`

### GCA

- GCA results summary: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/GCA_RESULTS_SUMMARY_FOR_SIYI.md`
- Revised GCA validation summary: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/GCA_REVISED_RUN_VALIDATION_SUMMARY.md`
- Reduced pair-family summary: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/reduced_pair_family_summary/REDUCED_GCA_PAIR_FAMILY_SUMMARY.md`
- Minimal-revised full-run GCA report: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi/GCA_FULLRUN_FSTAT_REPORT.md`
- Minimal-revised condition-window GCA report: `/dss/studies/fmri-haiku/glm_unified/granger_analysis_full/legacy3roi_tr1_deconv_doi_condition_all_timepoints/GCA_CONDITION_ALL_TIMEPOINTS_REPORT.md`
- Figure-ready GCA values table: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/gca_figure_values.csv`
- GCA report-update script: `/dss/studies/fmri-haiku/glm_unified/scripts/update_integrated_report_gca_20260427.py`
- GCA beta figure script: `/dss/studies/fmri-haiku/glm_unified/scripts/plot_gca_current_overall_condition_betas.py`
- GCA beta figure: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures/fig5_gca_current_overall_condition_betas.png`
- Full 30-s condition-window GCA bar-plot script: `/dss/studies/fmri-haiku/glm_unified/scripts/plot_gca_condition_30s_bars.py`
- Full 30-s condition-window GCA bar plot: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures/fig_gca_condition_30s_bars.png`

### This integrated report

- Report: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/report.md`
- Figures: `/dss/studies/fmri-haiku/glm_unified/integrated_reports/brain_behavior_ppi_fc_gca_coherence/figures`
