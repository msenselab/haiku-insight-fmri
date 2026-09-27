# Haiku insight fMRI — revised-manuscript evidence

This tree is aligned to the revised manuscript's Google Doc tab `t.0` (September 2026), not to the older May manuscript. See [`revision_2026_09/README.md`](revision_2026_09/README.md) for the exact seven embedded figures, model-specific scripts, and selected aggregate result tables. The separate [`unthresholded_tmaps/`](unthresholded_tmaps/) folder contains **12 signed, unthresholded N=19 group t-maps**, named by model and contrast. The 1-s *phase interaction* is excluded at the author's request; the 1-s Pre CA>JX sensitivity is retained.

## Current analysis scope

- Figures 1–2: model timing schematic and aggregate response counts.
- Figure 3: first-response RT by CA/JX/OI. The embedded image is `revision_2026_09/figures/figure3.png`. The retained older `figures/figure1_behavioral_rt.png` has identical rendered pixels. Its existing public input/code are `code/01_behavioral_stats.py`, `code/01_plot_behavioral_rt_figure.py`, `data/behavioral/behavioral_rt.csv`, `data/behavioral/figure1_rt_subject_values.csv`, and `data/behavioral/figure1_rt_summary.csv`; higher-resolution source assets remain in `figures/source_panels/fig_behavioral_rt_first_insight_conditions*`. This is **existing public participant-level behavioral data**, not newly released participant-level coefficients.
- Figures 4–6: revised Search/Pre-response, multiple-response, and response-order activation. Whole-brain inferences use permutation **cluster FWE within each contrast map** (not across-map Holm). Figure 6C's descriptive participant traces are captured in the exact Doc image; the *displayed* 19/19/15 aggregate source summary is in `revision_2026_09/data/response_order/`. The valid but undisplayed 19/18/15 variant is retained only in the internal project. Figure 6C is post-selection description, not a second inferential test.
- Figure 7: exploratory condition-specific gPPI. Its coefficient tests and targeted correction are distinct from whole-brain map inference. The release includes aggregate gPPI tables but no new participant-level gPPI vectors.

## What is not in the current evidence tier

The only root `data/`, `code/`, and `figures/` entries retained are the existing public **Figure-3 behavioral RT** inputs, scripts, summaries, and matching images. Unmatched May processed outputs (legacy Z maps, GCA/FC, PCC×RT PPI, ROI and participant-level tables), scripts, figures, reports, and May-only provenance notes have been removed from the current tree; previously public versions remain recoverable in Git history, **not erased from the public repository's history**. No raw MRI or single-subject fMRIPrep derivatives are added. The selected model scripts need protected inputs, so this public tier cannot rerun the full fMRI pipeline or regenerate every participant-dot panel. The 12-map set is findings-focused, not all tested null contrasts and not a NeuroVault deposit.

## Verify the selected current tier

From repository root with Python, NumPy and nibabel installed:

```bash
python revision_2026_09/code/verify_release.py
```

This validates figure and t-map hashes, map dimensions/sign/finiteness, public aggregate-table integrity, script syntax, and selected manuscript result landmarks. It is **not** a raw-to-results reproduction or a data-sharing/license certification.

## License and data-sharing scope

The authors authorize **Creative Commons Attribution 4.0 International (CC BY 4.0)** for the code and data in this revised repository; see [`LICENSE`](LICENSE) and the [license legal code](https://creativecommons.org/licenses/by/4.0/legalcode). Please attribute the MSense Lab haiku-insight-fMRI repository and identify changes when reusing licensed materials. This code/data grant does not assert rights in third-party dependencies or grant separate reuse rights for manuscript figure images. The existing N=19 individual Figure-3 behavioral RT inputs are retained with author approval; no new participant-level MRI/ROI/PPI/response-order inputs are included. Removing old material from the current tree does **not** erase previously published material from Git history.
