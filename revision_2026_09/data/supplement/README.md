# Supplement data scope

This folder is an **aggregate data companion**, not a copy of the live supplementary document or a complete fMRI reproduction.

The live document's **S1** presents example haiku in three categories. Those poems are not redistributed here: the authors' permission to reproduce them in the document does not, by itself, license their republication as repository data.

The live document's **S2** reports one N=19 Early–Middle–Pre-response first-response GLM. It tests 21 two-sided contrast maps and displays one table of 41 clusters that passed **within-map** cluster-extent FWE:

- `S2_N19_clusters.csv`: all 41 numerical entries from the live S2 cluster table, with source atlas labels. A few displayed labels expand “Unassigned” or generic tissue descriptions; cluster locations, extents, signed peak statistics, and within-map p-values match after display rounding.
- `S2_N19_map_ledger.csv`: a **companion analysis ledger** for all 21 maps of the same GLM (including null maps); it is **not a table displayed in the live supplement**. Its historical family-Holm columns are retained as provenance, not used to classify the displayed within-map-FWE clusters.

The separate sample-size **sensitivity-analysis tab** contains the simulation to which the formerly mislabelled public `S1_*` grid files belonged. That tab's simulation is **not** live supplementary S1 and those CSVs have been removed from this manuscript-matched current-tree folder; they remain in earlier Git history and in the internal analysis project. The different two-phase `code/first_response/run_joint_search_pre.py` is **not** an implementation of the S2 three-phase GLM. This public aggregate tier does not provide the protected inputs and full source pipeline required to refit the S2 GLM.
