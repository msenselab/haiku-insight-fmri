#!/usr/bin/env python3
"""Rebuild manuscript Table 1 using permutation cluster-size FWE inference.

Primary specification
---------------------
- Input: N=19 subject-level unified-GLM effect maps (sub-005 excluded;
  sub-022 has no unified first-level map).
- Contrasts: insight_CA_gt_JX and insight_CA_gt_OI.
- Group test: one-sample, positive-direction t test.
- Cluster-defining threshold: one-sided voxelwise p < .001.
- Inference: sign-flip permutations of the one-sample intercept.
- Cluster statistic: maximum cluster extent (6-neighbour connectivity).
- Significance: cluster-level p_FWE < .05 within each contrast.

The script saves all permutation output maps, all CDT clusters, the surviving
cluster table, exact input provenance, and manuscript-ready Markdown.
"""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn import plotting
from scipy import ndimage
from scipy.stats import t as t_distribution

BASE = Path(os.environ["HAIKU_PROJECT_ROOT"])
GLM = BASE / "glm_unified"
FIRST_LEVEL = GLM / "first_level"
GROUP_LEGACY = GLM / "group_results_no_outlier"
DEFAULT_OUTPUT = GLM / "permutation_cluster_fwe_table1_20260825"

SUBJECTS = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-006",
    "sub-007", "sub-008", "sub-009", "sub-010", "sub-011",
    "sub-012", "sub-013", "sub-014", "sub-016", "sub-018",
    "sub-019", "sub-020", "sub-021", "sub-023",
]

CONTRASTS = {
    "insight_CA_gt_JX": "CA > JX",
    "insight_CA_gt_OI": "CA > OI",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def region_label(x: float, y: float, z: float) -> str:
    """Conservative coordinate-based labels matching the manuscript vocabulary."""
    if x <= -35 and -72 <= y <= -40 and 10 <= z <= 50:
        return "L Angular Gyrus"
    if abs(x) <= 18 and -75 <= y <= -38 and 20 <= z <= 55:
        return "PCC/Precuneus"
    if x >= 24 and 5 <= y <= 30 and z <= -20:
        return "R Anterior Temporal Lobe/OFC"
    if abs(x) <= 15 and 30 <= y <= 60 and -30 <= z <= 5:
        return "vmPFC"
    if x >= 30 and y <= -70:
        return "R Visual Cortex"
    if x <= -30 and y <= -70:
        return "L Visual Cortex"
    if x >= 35 and y > 0:
        return "R Frontal Cortex"
    if x <= -35 and y > 0:
        return "L Frontal Cortex"
    if abs(x) <= 18 and y <= -35:
        return "Medial Posterior Cortex"
    return "Unlabeled peak"


def p_string(p: float, n_perm: int) -> str:
    resolution = 1.0 / (n_perm + 1)
    if p <= resolution * 1.000001:
        return f"<{2 * resolution:.5f}"
    if p < 0.001:
        return f"{p:.5f}"
    return f"{p:.3f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional tabulate."""
    columns = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows))
        for i in range(len(columns))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    header = render(columns)
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    return "\n".join([header, divider, *(render(row) for row in rows)])


def extract_clusters(
    contrast_key: str,
    contrast_label: str,
    t_img: nib.Nifti1Image,
    z_img: nib.Nifti1Image,
    size_img: nib.Nifti1Image,
    logp_size_img: nib.Nifti1Image,
    t_threshold: float,
    n_perm: int,
) -> pd.DataFrame:
    t_data = np.asanyarray(t_img.dataobj)
    z_data = np.asanyarray(z_img.dataobj)
    size_data = np.asanyarray(size_img.dataobj)
    logp_data = np.asanyarray(logp_size_img.dataobj)

    # scipy's default 3D structure is face-connected (6-neighbour), matching
    # nilearn's cluster-extent implementation.
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, n_clusters = ndimage.label(t_data > t_threshold, structure=structure)
    voxel_volume = float(abs(np.linalg.det(t_img.affine[:3, :3])))

    rows: list[dict[str, object]] = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labels == cluster_id
        n_voxels = int(cluster_mask.sum())
        if n_voxels == 0:
            continue

        cluster_indices = np.argwhere(cluster_mask)
        cluster_t = t_data[cluster_mask]
        peak_local = int(np.nanargmax(cluster_t))
        peak_ijk = cluster_indices[peak_local]
        peak_xyz = nib.affines.apply_affine(t_img.affine, peak_ijk)
        peak_t = float(t_data[tuple(peak_ijk)])
        peak_z = float(z_data[tuple(peak_ijk)])

        nilearn_sizes = np.unique(size_data[cluster_mask])
        nilearn_sizes = nilearn_sizes[nilearn_sizes > 0]
        size_from_map = int(round(float(nilearn_sizes.max()))) if len(nilearn_sizes) else 0
        if size_from_map != n_voxels:
            raise RuntimeError(
                f"Cluster-size mismatch in {contrast_key}, cluster {cluster_id}: "
                f"label={n_voxels}, nilearn={size_from_map}"
            )

        cluster_logp = float(np.nanmax(logp_data[cluster_mask]))
        cluster_p = float(10.0 ** (-cluster_logp))

        x, y, z = (float(v) for v in peak_xyz)
        rows.append(
            {
                "Contrast_key": contrast_key,
                "Contrast": contrast_label,
                "Cluster_ID": cluster_id,
                "Region": region_label(x, y, z),
                "MNI_x": x,
                "MNI_y": y,
                "MNI_z": z,
                "Peak_t": peak_t,
                "Peak_Z": peak_z,
                "Cluster_size_voxels": n_voxels,
                "Cluster_size_mm3": n_voxels * voxel_volume,
                "Cluster_pFWE": cluster_p,
                "Cluster_pFWE_display": p_string(cluster_p, n_perm),
                "Survives_pFWE_lt_05": cluster_p < 0.05,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Survives_pFWE_lt_05", "Cluster_pFWE", "Cluster_size_voxels"],
        ascending=[False, True, False],
    )


def markdown_table(sig: pd.DataFrame, metadata: dict[str, object]) -> str:
    lines = [
        "# Rebuilt Table 1: Permutation cluster-level FWE results",
        "",
        "**Table 1. Significant whole-brain activations for insight-phase contrasts "
        "using permutation-based cluster-level FWE correction.**",
        "",
    ]
    if sig.empty:
        lines.append("No clusters survived cluster-level $p_{FWE} < .05$.")
    else:
        display = sig.copy()
        display["MNI (x, y, z)"] = display.apply(
            lambda r: f"({r.MNI_x:.0f}, {r.MNI_y:.0f}, {r.MNI_z:.0f})", axis=1
        )
        display["Peak t"] = display["Peak_t"].map(lambda x: f"{x:.2f}")
        display["Peak Z"] = display["Peak_Z"].map(lambda x: f"{x:.2f}")
        display["Cluster size (voxels)"] = display["Cluster_size_voxels"].astype(int)
        display["Cluster size (mm³)"] = display["Cluster_size_mm3"].map(lambda x: f"{x:,.0f}")
        display["Cluster pFWE"] = display["Cluster_pFWE_display"]
        display = display[
            [
                "Contrast", "Region", "MNI (x, y, z)", "Peak t", "Peak Z",
                "Cluster size (voxels)", "Cluster size (mm³)", "Cluster pFWE",
            ]
        ]
        lines.append(dataframe_to_markdown(display))

    lines.extend(
        [
            "",
            "**Note.** Positive-direction, one-sample second-level tests were evaluated "
            f"using {metadata['n_permutations']:,} sign-flip permutations. Clusters were "
            "formed at one-sided voxelwise *p* < .001 and retained at cluster-level "
            "*p*FWE < .05 based on the permutation distribution of the maximum cluster "
            "extent. Cluster FWE was controlled separately within each contrast. "
            f"The analysis included *N* = {metadata['n_subjects']} participants; "
            f"df = {metadata['degrees_of_freedom']}; CDT *t* > "
            f"{metadata['cluster_forming_t']:.4f}. Peak Z values are parametric "
            "one-sample Z transformations reported for continuity with the original "
            "table; corrected inference is based on the cluster *p*FWE values. "
            "Connectivity used 6-neighbour voxel adjacency.",
            "",
            "## Analysis provenance",
            "",
            f"- Nilearn: {metadata['software']['nilearn']}",
            f"- Random seed: {metadata['random_seed']}",
            f"- Permutation p-value resolution: {metadata['permutation_p_resolution']:.8f}",
            f"- Analysis-mask voxels: {metadata['mask_voxels']:,}",
            f"- Voxel volume: {metadata['voxel_volume_mm3']:.6f} mm³",
            "- Subject-level contrast-map paths and SHA-256 checksums are in `metadata.json`.",
            "- `all_cdt_clusters.csv` contains every cluster crossing the voxelwise CDT, "
            "including clusters that did not survive cluster-level FWE correction.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=20260825)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = len(SUBJECTS)
    df = n_subjects - 1
    cdt_p = 0.001
    t_threshold = float(t_distribution.isf(cdt_p, df))

    input_records: list[dict[str, object]] = []
    all_tables: list[pd.DataFrame] = []
    parametric_diff_checks: dict[str, dict[str, float]] = {}
    mask_img = None
    mask_voxels = None
    voxel_volume = None

    for contrast_key, contrast_label in CONTRASTS.items():
        print(f"\n=== {contrast_label} ({contrast_key}) ===", flush=True)
        effect_maps: list[str] = []
        for subject in SUBJECTS:
            path = FIRST_LEVEL / subject / f"{subject}_{contrast_key}_effect.nii.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            effect_maps.append(str(path))
            input_records.append(
                {
                    "subject": subject,
                    "contrast": contrast_key,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        if len(effect_maps) != 19:
            raise RuntimeError(f"Expected 19 maps for {contrast_key}; got {len(effect_maps)}")

        design = pd.DataFrame({"intercept": np.ones(n_subjects, dtype=float)})
        parametric = SecondLevelModel(smoothing_fwhm=None).fit(
            effect_maps, design_matrix=design
        )
        current_mask = parametric.masker_.mask_img_
        current_mask_data = np.asanyarray(current_mask.dataobj).astype(bool)
        if mask_img is None:
            mask_img = current_mask
            mask_voxels = int(current_mask_data.sum())
            voxel_volume = float(abs(np.linalg.det(current_mask.affine[:3, :3])))
            nib.save(mask_img, output_dir / "analysis_mask.nii.gz")
        elif not np.array_equal(current_mask_data, np.asanyarray(mask_img.dataobj).astype(bool)):
            raise RuntimeError("Automatically computed masks differ between contrasts")

        z_img = parametric.compute_contrast("intercept", output_type="z_score")
        t_param_img = parametric.compute_contrast("intercept", output_type="stat")
        nib.save(z_img, output_dir / f"{contrast_key}_parametric_z.nii.gz")
        nib.save(t_param_img, output_dir / f"{contrast_key}_parametric_t.nii.gz")

        legacy_path = GROUP_LEGACY / f"group_{contrast_key}_zmap.nii.gz"
        legacy = nib.load(legacy_path)
        z_data = np.asanyarray(z_img.dataobj)
        legacy_data = np.asanyarray(legacy.dataobj)
        valid = np.isfinite(z_data) & np.isfinite(legacy_data)
        parametric_diff_checks[contrast_key] = {
            "max_abs_z_difference_from_legacy": float(
                np.max(np.abs(z_data[valid] - legacy_data[valid]))
            ),
            "z_correlation_with_legacy": float(
                np.corrcoef(z_data[valid].ravel(), legacy_data[valid].ravel())[0, 1]
            ),
        }

        outputs = non_parametric_inference(
            effect_maps,
            design_matrix=design,
            second_level_contrast="intercept",
            mask=current_mask,
            smoothing_fwhm=None,
            model_intercept=True,
            n_perm=args.n_perm,
            two_sided_test=False,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=1,
            threshold=cdt_p,
            tfce=False,
        )

        for key, img in outputs.items():
            nib.save(img, output_dir / f"{contrast_key}_{key}.nii.gz")

        permutation_t = np.asanyarray(outputs["t"].dataobj)
        parametric_t = np.asanyarray(t_param_img.dataobj)
        if not np.allclose(permutation_t, parametric_t, rtol=1e-5, atol=1e-5):
            max_diff = float(np.nanmax(np.abs(permutation_t - parametric_t)))
            raise RuntimeError(f"Permutation and parametric t maps differ: {max_diff}")

        clusters = extract_clusters(
            contrast_key,
            contrast_label,
            outputs["t"],
            z_img,
            outputs["size"],
            outputs["logp_max_size"],
            t_threshold,
            args.n_perm,
        )
        clusters.to_csv(output_dir / f"{contrast_key}_all_cdt_clusters.csv", index=False)
        all_tables.append(clusters)

        threshold_logp = -math.log10(0.05)
        significant_mask = (
            np.asanyarray(outputs["logp_max_size"].dataobj) > threshold_logp
        ) & (np.asanyarray(outputs["t"].dataobj) > t_threshold)
        significant_t = np.where(significant_mask, np.asanyarray(outputs["t"].dataobj), 0.0)
        sig_img = nib.Nifti1Image(significant_t.astype(np.float32), outputs["t"].affine, outputs["t"].header)
        nib.save(sig_img, output_dir / f"{contrast_key}_clusterFWE_p_lt_05_t.nii.gz")

        if significant_mask.any():
            plotting.plot_stat_map(
                sig_img,
                threshold=t_threshold,
                display_mode="ortho",
                colorbar=True,
                draw_cross=False,
                title=f"{contrast_label}: permutation cluster FWE p < .05",
                output_file=str(output_dir / f"{contrast_key}_clusterFWE_p_lt_05.png"),
            )

        n_sig = int(clusters["Survives_pFWE_lt_05"].sum()) if not clusters.empty else 0
        print(f"CDT clusters: {len(clusters)}; surviving cluster FWE: {n_sig}", flush=True)

    all_clusters = pd.concat(all_tables, ignore_index=True)
    significant = all_clusters.loc[all_clusters["Survives_pFWE_lt_05"]].copy()
    significant = significant.sort_values(["Contrast", "Cluster_pFWE", "Cluster_size_voxels"])
    all_clusters.to_csv(output_dir / "all_cdt_clusters.csv", index=False)
    significant.to_csv(output_dir / "table1_clusterFWE_significant.csv", index=False)

    metadata: dict[str, object] = {
        "analysis": "permutation maximum-cluster-extent FWE",
        "n_subjects": n_subjects,
        "subjects": SUBJECTS,
        "degrees_of_freedom": df,
        "contrasts": CONTRASTS,
        "test_direction": "positive, one-sided",
        "cluster_forming_p_uncorrected": cdt_p,
        "cluster_forming_t": t_threshold,
        "cluster_connectivity": "6-neighbour (face-connected)",
        "cluster_alpha_fwe": 0.05,
        "cluster_statistic": "extent in voxels",
        "fwe_scope": "separately within each whole-brain contrast",
        "n_permutations": args.n_perm,
        "random_seed": args.random_state,
        "permutation_p_resolution": 1.0 / (args.n_perm + 1),
        "n_jobs": args.n_jobs,
        "mask_voxels": mask_voxels,
        "voxel_volume_mm3": voxel_volume,
        "parametric_reproduction_checks": parametric_diff_checks,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
            "nilearn": nilearn.__version__,
        },
        "inputs": input_records,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "TABLE1_clusterFWE.md").write_text(markdown_table(significant, metadata))

    print("\n=== Final surviving clusters ===")
    if significant.empty:
        print("None")
    else:
        print(
            significant[
                [
                    "Contrast", "Region", "MNI_x", "MNI_y", "MNI_z", "Peak_t",
                    "Peak_Z", "Cluster_size_voxels", "Cluster_size_mm3", "Cluster_pFWE",
                ]
            ].to_string(index=False)
        )
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
