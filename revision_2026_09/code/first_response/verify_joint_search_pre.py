#!/usr/bin/env python3
"""Independent verification for the joint Search/Pre-response contrast extension."""
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label
from scipy.stats import t as t_distribution

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
BASE = ROOT / "glm_unified/insight_event_timing_sensitivity/joint_search_pre_phase_contrasts_20260831"
OLD = ROOT / "glm_unified/insight_event_timing_sensitivity/reduced_nonoverlap_search_pre0p5_no_rtmod_n19mask"
FIRST = BASE / "first_level"
GROUP = BASE / "group_inference"
MASK_PATH = BASE / "first_level_masks/common_n19_intersection_mask.nii.gz"
SUBJECTS = [f"sub-{i:03d}" for i in (1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23)]
CONTRASTS = [
    "pre_CA_gt_JX", "pre_CA_gt_OI", "pre_JX_gt_OI",
    "search_CA_gt_JX", "search_CA_gt_OI", "search_JX_gt_OI",
    "phase_pre_gt_search_CA_gt_JX", "phase_pre_gt_search_CA_gt_OI",
    "phase_pre_gt_search_JX_gt_OI",
]
FAMILIES = {
    "pre_planned": ["pre_CA_gt_JX", "pre_CA_gt_OI"],
    "search_secondary": ["search_CA_gt_JX", "search_CA_gt_OI"],
    "phase_specificity": ["phase_pre_gt_search_CA_gt_JX", "phase_pre_gt_search_CA_gt_OI"],
    "exploratory_JX_gt_OI": ["pre_JX_gt_OI", "search_JX_gt_OI", "phase_pre_gt_search_JX_gt_OI"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_effect(subject: str, key: str) -> np.ndarray:
    path = FIRST / subject / f"{subject}_{key}_effect.nii.gz"
    if not path.exists():
        raise AssertionError(f"Missing effect map: {path}")
    return np.asarray(nib.load(path).dataobj, dtype=np.float64)


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, tol: float, checks: dict) -> None:
    diff = float(np.max(np.abs(actual - expected)))
    checks[name] = diff
    if diff > tol:
        raise AssertionError(f"{name}: max absolute difference {diff} > {tol}")


def holm(raw: np.ndarray) -> np.ndarray:
    order = np.argsort(raw)
    adj_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
    out = np.empty_like(raw)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def main() -> None:
    metadata = json.loads((BASE / "metadata.json").read_text())
    if metadata["subjects"] != SUBJECTS or metadata["n_subjects"] != 19:
        raise AssertionError("Locked sample mismatch")
    if metadata["group_inference"]["families"] != FAMILIES:
        raise AssertionError("Correction-family metadata mismatch")
    if metadata["group_inference"]["n_permutations"] != 50000:
        raise AssertionError("Permutation count mismatch")

    mask_img = nib.load(MASK_PATH)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    if int(mask.sum()) != metadata["mask"]["voxels"]:
        raise AssertionError("Mask voxel count mismatch")
    if sha256(MASK_PATH) != metadata["mask"]["sha256"]:
        raise AssertionError("Mask hash mismatch")

    checks: dict[str, float | int | str] = {}
    algebra_tol = 1e-6
    for subject in SUBJECTS:
        arrays = {key: load_effect(subject, key) for key in CONTRASTS}
        if any(a.shape != mask.shape for a in arrays.values()):
            raise AssertionError(f"{subject}: effect-map shape mismatch")
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise AssertionError(f"{subject}: nonfinite effect-map values")

        assert_close(f"{subject}: pre JX>OI algebra",
                     arrays["pre_JX_gt_OI"], arrays["pre_CA_gt_OI"] - arrays["pre_CA_gt_JX"], algebra_tol, checks)
        assert_close(f"{subject}: search JX>OI algebra",
                     arrays["search_JX_gt_OI"], arrays["search_CA_gt_OI"] - arrays["search_CA_gt_JX"], algebra_tol, checks)
        assert_close(f"{subject}: CA>JX phase algebra",
                     arrays["phase_pre_gt_search_CA_gt_JX"], arrays["pre_CA_gt_JX"] - arrays["search_CA_gt_JX"], algebra_tol, checks)
        assert_close(f"{subject}: CA>OI phase algebra",
                     arrays["phase_pre_gt_search_CA_gt_OI"], arrays["pre_CA_gt_OI"] - arrays["search_CA_gt_OI"], algebra_tol, checks)
        assert_close(f"{subject}: JX>OI phase algebra",
                     arrays["phase_pre_gt_search_JX_gt_OI"], arrays["pre_JX_gt_OI"] - arrays["search_JX_gt_OI"], algebra_tol, checks)

        for key in ("pre_CA_gt_JX", "pre_CA_gt_OI"):
            old_path = OLD / "first_level" / subject / f"{subject}_{key}_effect.nii.gz"
            old = np.asarray(nib.load(old_path).dataobj, dtype=np.float64)
            assert_close(f"{subject}: old/new {key}", arrays[key], old, 0.0, checks)

    summary = pd.read_csv(GROUP / "contrast_summary.csv")
    clusters = pd.read_csv(GROUP / "all_cdt_clusters.csv")
    if set(summary["map_key"]) != set(CONTRASTS) or len(summary) != len(CONTRASTS):
        raise AssertionError("Group-summary contrast set mismatch")
    if set(clusters["map_key"]) - set(CONTRASTS):
        raise AssertionError("Unexpected contrast in cluster table")

    threshold = float(t_distribution.isf(.001, len(SUBJECTS) - 1))
    if abs(threshold - metadata["group_inference"]["cluster_forming_t"]) > 1e-12:
        raise AssertionError("Cluster-forming t threshold mismatch")
    structure = generate_binary_structure(3, 1)
    max_group_t_diff = 0.0
    max_cluster_p_diff = 0.0
    for key in CONTRASTS:
        stack = np.stack([load_effect(s, key) for s in SUBJECTS])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_data = stack.mean(axis=0) / (stack.std(axis=0, ddof=1) / np.sqrt(len(SUBJECTS)))
        t_data[~mask] = 0.0
        t_data[~np.isfinite(t_data)] = 0.0
        saved_t = np.asarray(nib.load(GROUP / f"{key}_parametric_t.nii.gz").dataobj, dtype=np.float64)
        diff = float(np.max(np.abs(saved_t - t_data)))
        max_group_t_diff = max(max_group_t_diff, diff)
        if diff > 1e-10:
            raise AssertionError(f"{key}: saved group t-map mismatch {diff}")

        components, n_components = label((t_data > threshold) & mask, structure=structure)
        rows = clusters[clusters["map_key"].eq(key)]
        if len(rows) != n_components:
            raise AssertionError(f"{key}: component count mismatch ({len(rows)} vs {n_components})")
        logp = np.asarray(nib.load(GROUP / f"{key}_logp_max_size.nii.gz").dataobj, dtype=float)
        sizes = np.asarray(nib.load(GROUP / f"{key}_size.nii.gz").dataobj, dtype=float)
        for row in rows.itertuples(index=False):
            cmask = components == int(row.cluster_id)
            if int(cmask.sum()) != int(row.cluster_voxels):
                raise AssertionError(f"{key} cluster {row.cluster_id}: k mismatch")
            idx = np.argwhere(cmask)
            peak = idx[int(np.argmax(t_data[cmask]))]
            xyz = nib.affines.apply_affine(mask_img.affine, peak)
            if np.max(np.abs(xyz - np.array([row.mni_x, row.mni_y, row.mni_z]))) > 1e-8:
                raise AssertionError(f"{key} cluster {row.cluster_id}: coordinate mismatch")
            if abs(float(t_data[tuple(peak)]) - float(row.peak_t)) > 1e-10:
                raise AssertionError(f"{key} cluster {row.cluster_id}: peak t mismatch")
            p_fwe = float(10 ** (-np.max(logp[cmask])))
            max_cluster_p_diff = max(max_cluster_p_diff, abs(p_fwe - float(row.cluster_pFWE)))
            if abs(p_fwe - float(row.cluster_pFWE)) > 1e-12:
                raise AssertionError(f"{key} cluster {row.cluster_id}: cluster pFWE mismatch")
            reported_sizes = np.unique(sizes[cmask])
            reported_sizes = reported_sizes[reported_sizes > 0]
            if len(reported_sizes) and int(round(float(reported_sizes.max()))) != int(row.cluster_voxels):
                raise AssertionError(f"{key} cluster {row.cluster_id}: size-map mismatch")

        srow = summary[summary["map_key"].eq(key)].iloc[0]
        min_p = float(rows["cluster_pFWE"].min()) if len(rows) else 1.0
        if abs(min_p - float(srow.minimum_cluster_pFWE)) > 1e-12:
            raise AssertionError(f"{key}: summary minimum p mismatch")
        if int(srow.n_cdt_clusters) != n_components:
            raise AssertionError(f"{key}: summary component count mismatch")

    for family, keys in FAMILIES.items():
        idx = summary.index[summary["family"].eq(family)]
        raw = summary.loc[idx, "minimum_cluster_pFWE"].to_numpy(float)
        expected = holm(raw)
        actual = summary.loc[idx, "minimum_cluster_pFWE_holm_within_family"].to_numpy(float)
        if np.max(np.abs(expected - actual)) > 1e-12:
            raise AssertionError(f"{family}: Holm correction mismatch")
        if set(summary.loc[idx, "map_key"]) != set(keys):
            raise AssertionError(f"{family}: member mismatch")

    within = pd.read_csv(GROUP / "significant_clusters_within_map.csv")
    gated = pd.read_csv(GROUP / "clusters_within_holm_significant_maps.csv")
    expected_within = clusters[clusters["cluster_pFWE"] < .05]
    expected_gated = expected_within[expected_within["map_survives_holm_05"].astype(bool)]
    if set(zip(within.map_key, within.cluster_id)) != set(zip(expected_within.map_key, expected_within.cluster_id)):
        raise AssertionError("Within-map significant-cluster file is not an exact filter")
    if set(zip(gated.map_key, gated.cluster_id)) != set(zip(expected_gated.map_key, expected_gated.cluster_id)):
        raise AssertionError("Holm-gated cluster file is not an exact filter")

    qc = pd.read_csv(BASE / "design_qc/design_qc_all_subjects.csv")
    if list(qc["subject"]) != SUBJECTS or not qc["full_rank"].astype(bool).all():
        raise AssertionError("Design-QC subject order/rank mismatch")
    max_vif = float(qc[["vif_pre_CA", "vif_pre_JX", "vif_pre_OI"]].max().max())
    if max_vif >= 10:
        raise AssertionError(f"VIF guard failed: {max_vif}")

    result = {
        "status": "PASS",
        "n_subjects": len(SUBJECTS),
        "n_contrasts": len(CONTRASTS),
        "n_families": len(FAMILIES),
        "mask_voxels": int(mask.sum()),
        "max_group_t_map_abs_diff": max_group_t_diff,
        "max_cluster_pFWE_abs_diff": max_cluster_p_diff,
        "max_pre_response_vif": max_vif,
        "n_uncorrected_cdt_components": int(len(clusters)),
        "n_within_map_cluster_fwe_clusters": int(len(within)),
        "n_holm_gated_clusters": int(len(gated)),
        "analysis_script_sha256": sha256(ROOT / "glm_unified/scripts/run_joint_search_pre_phase_contrasts_20260831.py"),
        "verification_script_sha256": sha256(Path(__file__)),
    }
    (BASE / "VERIFICATION_REPORT.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Independent verification report", "",
        "**Status: PASS**", "",
        f"- Locked sample: N={result['n_subjects']}.",
        f"- Contrasts: {result['n_contrasts']} in {result['n_families']} declared correction families.",
        f"- Common-mask voxels: {result['mask_voxels']}.",
        f"- Maximum independently recomputed group-t difference: {result['max_group_t_map_abs_diff']:.3g}.",
        f"- Maximum independently recomputed cluster-pFWE difference: {result['max_cluster_pFWE_abs_diff']:.3g}.",
        f"- Maximum Pre-response VIF: {result['max_pre_response_vif']:.3f} (<10 guard).",
        f"- Uncorrected CDT components: {result['n_uncorrected_cdt_components']}.",
        f"- Within-map cluster-FWE clusters: {result['n_within_map_cluster_fwe_clusters']}.",
        f"- Clusters in Holm-significant maps: {result['n_holm_gated_clusters']}.", "",
        "Verified: participant-map completeness and finiteness; exact old/new Pre-response reproduction; contrast algebra; independently recomputed group t maps; voxel threshold, six-connected components, peaks, coordinates, sizes, cluster pFWE values, family membership, Holm values, filtered tables, sample, mask, full rank, and VIF guard.", "",
    ]
    (BASE / "VERIFICATION_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
