#!/usr/bin/env python3
"""Joint Search + Pre-response GLM contrast extension for haiku insight.

Responded trials are partitioned into two condition-specific epochs:

- Search: stimulus onset to 0.5 s before the first button press
- Pre-response: final 0.5 s before the first button press

There are no Search x RT amplitude modulators. No-insight trials are represented
by one pooled nuisance column, capped at the next trial onset. The locked sample
is N=19 voxelwise through an intersection of the actual first-level masks. This
non-destructive extension saves condition contrasts for Search and Pre-response,
plus direct Pre-response-minus-Search condition interactions. Group inference
uses one-sided sign-flip permutations, voxelwise CDT p<.001,
maximum-cluster-extent pFWE<.05 within map, and Holm correction within four
declared map families. Every voxel-p<.001 component is retained for simultaneous
uncorrected and corrected reporting.
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from scipy import ndimage
from scipy.stats import t as t_distribution

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
FMRIPREP = ROOT / "derivatives/fmriprep"
BEHAVIOR = ROOT / "manuscript/behavioral_analysis/behavioral_data_all_subjects.csv"
ANALYSIS_VERSION = "joint_search_pre_phase_contrasts_v1_20260831"
OUT = ROOT / "glm_unified/insight_event_timing_sensitivity/joint_search_pre_phase_contrasts_20260831"
FIRST = OUT / "first_level"
GROUP = OUT / "group_inference"
DESIGN_QC = OUT / "design_qc"
MASKS = OUT / "first_level_masks"
COMMON_MASK = MASKS / "common_n19_intersection_mask.nii.gz"
for directory in (OUT, FIRST, GROUP, DESIGN_QC, MASKS):
    directory.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub-{i:03d}" for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
CONDITIONS = ["CA", "JX", "OI"]
COND_MAP = {"context-action": "CA", "juxtaposition": "JX", "One-image": "OI"}
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
TR = 1.0
FWHM = 6.0
HRF = "spm"
HIGH_PASS = 1 / 128.0
NOISE = "ar1"
PRE_DURATION = 0.5
NO_INSIGHT_DURATION = 29.0
CDT_P = 0.001

CONTRASTS = {
    "pre_CA_gt_JX": {"pre_CA": 1.0, "pre_JX": -1.0},
    "pre_CA_gt_OI": {"pre_CA": 1.0, "pre_OI": -1.0},
    "pre_JX_gt_OI": {"pre_JX": 1.0, "pre_OI": -1.0},
    "search_CA_gt_JX": {"search_CA": 1.0, "search_JX": -1.0},
    "search_CA_gt_OI": {"search_CA": 1.0, "search_OI": -1.0},
    "search_JX_gt_OI": {"search_JX": 1.0, "search_OI": -1.0},
    "phase_pre_gt_search_CA_gt_JX": {
        "pre_CA": 1.0, "pre_JX": -1.0,
        "search_CA": -1.0, "search_JX": 1.0,
    },
    "phase_pre_gt_search_CA_gt_OI": {
        "pre_CA": 1.0, "pre_OI": -1.0,
        "search_CA": -1.0, "search_OI": 1.0,
    },
    "phase_pre_gt_search_JX_gt_OI": {
        "pre_JX": 1.0, "pre_OI": -1.0,
        "search_JX": -1.0, "search_OI": 1.0,
    },
}
GROUP_MAPS = {
    "pre_CA_gt_JX": "Pre-response CA > JX",
    "pre_CA_gt_OI": "Pre-response CA > OI",
    "pre_JX_gt_OI": "Pre-response JX > OI",
    "search_CA_gt_JX": "Search CA > JX",
    "search_CA_gt_OI": "Search CA > OI",
    "search_JX_gt_OI": "Search JX > OI",
    "phase_pre_gt_search_CA_gt_JX": "[Pre CA−JX] > [Search CA−JX]",
    "phase_pre_gt_search_CA_gt_OI": "[Pre CA−OI] > [Search CA−OI]",
    "phase_pre_gt_search_JX_gt_OI": "[Pre JX−OI] > [Search JX−OI]",
}
MAP_FAMILIES = {
    "pre_planned": ["pre_CA_gt_JX", "pre_CA_gt_OI"],
    "search_secondary": ["search_CA_gt_JX", "search_CA_gt_OI"],
    "phase_specificity": ["phase_pre_gt_search_CA_gt_JX", "phase_pre_gt_search_CA_gt_OI"],
    "exploratory_JX_gt_OI": ["pre_JX_gt_OI", "search_JX_gt_OI", "phase_pre_gt_search_JX_gt_OI"],
}
MAP_TO_FAMILY = {
    map_key: family
    for family, map_keys in MAP_FAMILIES.items()
    for map_key in map_keys
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def as_float64(img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float64), img.affine)


def load_subject_inputs(subject: str) -> tuple[Path, Path, Path]:
    func_dir = FMRIPREP / subject / "func"
    func = func_dir / f"{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    meta = func.with_suffix("").with_suffix(".json")
    conf = func_dir / f"{subject}_task-haiku_desc-confounds_timeseries.tsv"
    for path in (func, meta, conf):
        if not path.exists():
            raise FileNotFoundError(path)
    return func, meta, conf


def build_events(subject: str, behavioral: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Partition each responded trial into nonoverlapping Search and Pre epochs."""
    if behavioral is None:
        behavioral = pd.read_csv(BEHAVIOR)
    sdf = behavioral[behavioral["subject_id"].eq(subject)].copy().sort_values("onset").reset_index(drop=True)
    if len(sdf) != 42:
        raise RuntimeError(f"{subject}: expected 42 trials, found {len(sdf)}")

    rows: list[dict[str, object]] = []
    qc: dict[str, Any] = {"subject": subject, "n_trials": len(sdf), "n_insight": 0, "n_no_insight": 0}
    for cond in CONDITIONS:
        qc[f"n_insight_{cond}"] = 0
    min_rt = np.inf
    for trial_idx, trial in sdf.iterrows():
        onset = float(trial["onset"])
        next_onset = float(sdf.iloc[trial_idx + 1]["onset"]) if trial_idx + 1 < len(sdf) else np.inf
        original = str(trial["haiku_type"])
        cond = COND_MAP[original]
        has = bool(trial["has_insight"])
        if has and pd.notna(trial["first_insight_rt"]):
            rt = float(trial["first_insight_rt"])
            if not np.isfinite(rt) or rt <= PRE_DURATION:
                raise RuntimeError(f"{subject}: invalid first response RT={rt}")
            min_rt = min(min_rt, rt)
            rows.extend([
                {"onset": onset, "duration": rt - PRE_DURATION,
                 "trial_type": f"search_{cond}", "modulation": 1.0},
                {"onset": onset + rt - PRE_DURATION, "duration": PRE_DURATION,
                 "trial_type": f"pre_{cond}", "modulation": 1.0},
            ])
            qc["n_insight"] = int(qc["n_insight"]) + 1
            qc[f"n_insight_{cond}"] = int(qc[f"n_insight_{cond}"]) + 1
        else:
            duration = min(NO_INSIGHT_DURATION, next_onset - onset)
            if not np.isfinite(duration):
                duration = NO_INSIGHT_DURATION
            if duration <= 0:
                raise RuntimeError(f"{subject}: non-positive no-insight duration {duration}")
            rows.append({"onset": onset, "duration": duration,
                         "trial_type": "no_insight", "modulation": 1.0})
            qc["n_no_insight"] = int(qc["n_no_insight"]) + 1
            qc["n_no_insight_capped_at_next_trial"] = int(qc.get("n_no_insight_capped_at_next_trial", 0)) + int(duration < NO_INSIGHT_DURATION)
    qc["minimum_first_response_rt_s"] = float(min_rt)
    events = pd.DataFrame(rows).sort_values(["onset", "trial_type"]).reset_index(drop=True)
    if not np.isfinite(events[["onset", "duration", "modulation"]].to_numpy(float)).all():
        raise RuntimeError(f"{subject}: non-finite event value")
    return events, qc


def load_confounds(path: Path, n_scans: int) -> pd.DataFrame:
    """Match the submitted unified GLM: six rigid-body motion regressors."""
    c = pd.read_csv(path, sep="\t")
    missing = [name for name in MOTION if name not in c.columns]
    if missing:
        raise RuntimeError(f"{path}: missing motion columns {missing}")
    out = c[MOTION].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(out) != n_scans:
        raise RuntimeError(f"{path}: confounds rows {len(out)} != scans {n_scans}")
    return out


def contrast_vector(columns: list[str], weights: dict[str, float]) -> np.ndarray:
    missing = [name for name in weights if name not in columns]
    if missing:
        raise RuntimeError(f"Missing contrast columns: {missing}")
    out = np.zeros(len(columns), dtype=float)
    for name, weight in weights.items():
        out[columns.index(name)] = weight
    return out


def vif_for_column(dm: pd.DataFrame, column: str) -> float:
    y = dm[column].to_numpy(float)
    x = dm.drop(columns=[column]).to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(residual @ residual) / denom
    return float(1.0 / max(1.0 - r2, np.finfo(float).eps))


def first_level_subject(subject: str) -> dict[str, Any]:
    print(f"[first] {subject}", flush=True)
    func, meta, conf_path = load_subject_inputs(subject)
    tr = float(json.loads(meta.read_text())["RepetitionTime"])
    if not np.isclose(tr, TR):
        raise RuntimeError(f"{subject}: TR={tr}, expected {TR}")
    img = nib.load(func)
    n_scans = int(img.shape[-1])
    events, event_qc = build_events(subject)
    confounds = load_confounds(conf_path, n_scans)

    model = FirstLevelModel(
        t_r=tr, smoothing_fwhm=FWHM, hrf_model=HRF, high_pass=HIGH_PASS,
        noise_model=NOISE, standardize=True, signal_scaling=0,
        minimize_memory=True, n_jobs=1,
        memory=str(OUT / "_nilearn_cache"), memory_level=1,
    )
    model.fit(str(func), events=events, confounds=confounds)
    dm = model.design_matrices_[0]
    columns = dm.columns.tolist()
    required = [f"{kind}_{cond}" for kind in ("search", "pre") for cond in CONDITIONS]
    missing = [name for name in required if name not in columns]
    if missing:
        raise RuntimeError(f"{subject}: missing task columns {missing}")

    task_cols = required + (["no_insight"] if "no_insight" in columns else [])
    corr = dm[task_cols].corr()
    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, np.nan)
    rank = int(np.linalg.matrix_rank(dm.to_numpy()))
    qc = {
        **event_qc,
        "n_scans": n_scans,
        "n_design_columns": len(columns),
        "design_rank": rank,
        "full_rank": bool(rank == len(columns)),
        "max_abs_task_correlation": float(np.nanmax(np.abs(corr_values))),
    }
    for cond in CONDITIONS:
        qc[f"corr_search_pre_{cond}"] = float(corr.loc[f"search_{cond}", f"pre_{cond}"])
        qc[f"vif_pre_{cond}"] = vif_for_column(dm, f"pre_{cond}")

    subject_out = FIRST / subject
    subject_out.mkdir(parents=True, exist_ok=True)
    events.to_csv(subject_out / f"{subject}_events.tsv", sep="\t", index=False)
    dm.to_csv(DESIGN_QC / f"{subject}_design_matrix.csv", index=False)
    (DESIGN_QC / f"{subject}_design_columns.json").write_text(json.dumps(columns, indent=2))

    fitted_mask = model.masker_.mask_img_
    if fitted_mask is None:
        raise RuntimeError(f"{subject}: fitted first-level mask is unavailable")
    mask_data = np.asarray(fitted_mask.dataobj).astype(bool)
    mask_path = MASKS / f"{subject}_first_level_mask.nii.gz"
    nib.Nifti1Image(mask_data.astype(np.uint8), fitted_mask.affine, fitted_mask.header).to_filename(mask_path)
    qc["first_level_mask_voxels"] = int(mask_data.sum())

    for contrast, weights in CONTRASTS.items():
        vec = contrast_vector(columns, weights)
        effect = as_float64(model.compute_contrast(vec, output_type="effect_size"))
        path = subject_out / f"{subject}_{contrast}_effect.nii.gz"
        effect.to_filename(path)
        reloaded = nib.load(path)
        if reloaded.get_data_dtype() != np.dtype("float64"):
            raise RuntimeError(f"{path}: expected float64, got {reloaded.get_data_dtype()}")
        if np.unique(np.asarray(reloaded.dataobj)).size <= 256:
            raise RuntimeError(f"{path}: possible quantization")
    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "subject": subject,
        "implementation_sha256": sha256(Path(__file__)),
        "behavior_sha256": sha256(BEHAVIOR),
        "bold_sha256": sha256(func),
        "bold_json_sha256": sha256(meta),
        "confounds_sha256": sha256(conf_path),
        "events_sha256": sha256(subject_out / f"{subject}_events.tsv"),
        "first_level_mask_sha256": sha256(mask_path),
        "contrast_sha256": {
            contrast: sha256(subject_out / f"{subject}_{contrast}_effect.nii.gz")
            for contrast in CONTRASTS
        },
    }
    (subject_out / f"{subject}_first_level_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[first] {subject} done", flush=True)
    return qc


def group_input_paths(map_key: str) -> list[Path]:
    if map_key not in GROUP_MAPS:
        raise KeyError(map_key)
    return [FIRST / subject / f"{subject}_{map_key}_effect.nii.gz" for subject in SUBJECTS]


def validate_and_build_common_mask() -> dict[str, Any]:
    """Require one coherent N=19 first-level build and intersect its fitted masks."""
    expected_script_hash = sha256(Path(__file__))
    mask_arrays: list[np.ndarray] = []
    reference_img = None
    manifest_rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        subject_out = FIRST / subject
        provenance_path = subject_out / f"{subject}_first_level_provenance.json"
        mask_path = MASKS / f"{subject}_first_level_mask.nii.gz"
        if not provenance_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"{subject}: missing provenance or first-level mask")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("analysis_version") != ANALYSIS_VERSION:
            raise RuntimeError(f"{subject}: stale analysis version in provenance")
        if provenance.get("implementation_sha256") != expected_script_hash:
            raise RuntimeError(f"{subject}: first-level maps were generated by a different implementation")
        if provenance.get("behavior_sha256") != sha256(BEHAVIOR):
            raise RuntimeError(f"{subject}: behavioral input changed after first-level fit")
        for contrast in CONTRASTS:
            path = subject_out / f"{subject}_{contrast}_effect.nii.gz"
            expected = provenance.get("contrast_sha256", {}).get(contrast)
            if not path.exists() or expected != sha256(path):
                raise RuntimeError(f"{subject}: stale or modified first-level map for {contrast}")
        if provenance.get("first_level_mask_sha256") != sha256(mask_path):
            raise RuntimeError(f"{subject}: first-level mask hash mismatch")
        img = nib.load(mask_path)
        data = np.asarray(img.dataobj).astype(bool)
        if reference_img is None:
            reference_img = img
        elif data.shape != mask_arrays[0].shape or not np.allclose(img.affine, reference_img.affine):
            raise RuntimeError(f"{subject}: first-level mask grid mismatch")
        mask_arrays.append(data)
        manifest_rows.append({
            "subject": subject, "mask_path": str(mask_path), "mask_sha256": sha256(mask_path),
            "mask_voxels": int(data.sum()), "provenance_path": str(provenance_path),
            "provenance_sha256": sha256(provenance_path),
        })
    assert reference_img is not None
    intersection = np.logical_and.reduce(mask_arrays)
    if int(intersection.sum()) == 0:
        raise RuntimeError("N=19 first-level mask intersection is empty")
    nib.Nifti1Image(intersection.astype(np.uint8), reference_img.affine, reference_img.header).to_filename(COMMON_MASK)
    for data in mask_arrays:
        if not np.all(data[intersection]):
            raise RuntimeError("Common mask is not a subset of every first-level mask")
    pd.DataFrame(manifest_rows).to_csv(MASKS / "first_level_mask_manifest.csv", index=False)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "n_subjects": len(SUBJECTS),
        "voxelwise_n": len(SUBJECTS),
        "intersection_voxels": int(intersection.sum()),
        "minimum_subject_mask_voxels": int(min(data.sum() for data in mask_arrays)),
        "maximum_subject_mask_voxels": int(max(data.sum() for data in mask_arrays)),
        "common_mask_sha256": sha256(COMMON_MASK),
    }
    (MASKS / "common_mask_qc.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def extract_clusters(map_key: str, t_img, z_img, size_img, logp_img, threshold: float, n_perm: int) -> pd.DataFrame:
    t_data = np.asarray(t_img.dataobj)
    z_data = np.asarray(z_img.dataobj)
    size_data = np.asarray(size_img.dataobj)
    logp_data = np.asarray(logp_img.dataobj)
    labels, n_clusters = ndimage.label(t_data > threshold, ndimage.generate_binary_structure(3, 1))
    voxel_volume = float(abs(np.linalg.det(t_img.affine[:3, :3])))
    rows = []
    for cid in range(1, n_clusters + 1):
        mask = labels == cid
        idx = np.argwhere(mask)
        if len(idx) == 0:
            continue
        peak_ijk = idx[int(np.argmax(t_data[mask]))]
        xyz = nib.affines.apply_affine(t_img.affine, peak_ijk)
        p_fwe = float(10 ** (-np.max(logp_data[mask])))
        n_vox = int(mask.sum())
        reported = np.unique(size_data[mask])
        reported = reported[reported > 0]
        if len(reported) and int(round(float(reported.max()))) != n_vox:
            raise RuntimeError(f"{map_key} cluster {cid}: cluster-size mismatch")
        rows.append({
            "map_key": map_key, "map_label": GROUP_MAPS[map_key], "cluster_id": cid,
            "mni_x": float(xyz[0]), "mni_y": float(xyz[1]), "mni_z": float(xyz[2]),
            "peak_t": float(t_data[tuple(peak_ijk)]), "peak_z": float(z_data[tuple(peak_ijk)]),
            "peak_p_uncorrected_one_sided": float(
                t_distribution.sf(float(t_data[tuple(peak_ijk)]), len(SUBJECTS) - 1)
            ),
            "cluster_voxels": n_vox, "cluster_mm3": n_vox * voxel_volume,
            "cluster_pFWE": p_fwe, "survives_cluster_pFWE_05": bool(p_fwe < .05),
            "permutation_resolution": 1.0 / n_perm,
        })
    return pd.DataFrame(rows)


def run_group(n_perm: int, n_jobs: int, random_state: int) -> None:
    common_mask_qc = validate_and_build_common_mask()
    design = pd.DataFrame({"intercept": np.ones(len(SUBJECTS), dtype=float)})
    t_threshold = float(t_distribution.isf(CDT_P, len(SUBJECTS) - 1))
    all_clusters: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []

    for idx, (map_key, label) in enumerate(GROUP_MAPS.items()):
        print(f"[group] {label}", flush=True)
        paths = group_input_paths(map_key)
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        maps = [str(path) for path in paths]
        for subject, path in zip(SUBJECTS, paths):
            provenance_path = FIRST / subject / f"{subject}_first_level_provenance.json"
            manifests.append({"analysis_version": ANALYSIS_VERSION, "map_key": map_key,
                              "subject": subject, "path": str(path), "sha256": sha256(path),
                              "size_bytes": path.stat().st_size, "provenance_path": str(provenance_path),
                              "provenance_sha256": sha256(provenance_path)})
        param = SecondLevelModel(mask_img=str(COMMON_MASK), smoothing_fwhm=None).fit(maps, design_matrix=design)
        t_img = as_float64(param.compute_contrast("intercept", output_type="stat"))
        z_img = as_float64(param.compute_contrast("intercept", output_type="z_score"))
        t_img.to_filename(GROUP / f"{map_key}_parametric_t.nii.gz")
        z_img.to_filename(GROUP / f"{map_key}_parametric_z.nii.gz")

        seed = random_state + idx * 1009
        outputs = non_parametric_inference(
            maps, design_matrix=design, second_level_contrast="intercept",
            mask=str(COMMON_MASK), smoothing_fwhm=None, model_intercept=True,
            n_perm=n_perm, two_sided_test=False, random_state=seed,
            n_jobs=n_jobs, verbose=1, threshold=CDT_P, tfce=False,
        )
        for key, image in outputs.items():
            as_float64(image).to_filename(GROUP / f"{map_key}_{key}.nii.gz")
        if not np.allclose(np.asarray(outputs["t"].dataobj), np.asarray(t_img.dataobj), rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"{map_key}: parametric and permutation t maps differ")
        clusters = extract_clusters(map_key, t_img, z_img, outputs["size"],
                                    outputs["logp_max_size"], t_threshold, n_perm)
        all_clusters.append(clusters)
        sig = clusters[clusters["survives_cluster_pFWE_05"]] if not clusters.empty else clusters
        summaries.append({
            "map_key": map_key, "map_label": label, "n_cdt_clusters": len(clusters),
            "n_significant_clusters": len(sig),
            "minimum_cluster_pFWE": float(clusters["cluster_pFWE"].min()) if not clusters.empty else 1.0,
            "random_seed": seed, "family": MAP_TO_FAMILY[map_key],
        })

    summary_df = pd.DataFrame(summaries)
    family_sizes = {k: len(v) for k, v in MAP_FAMILIES.items()}
    summary_df["family_size"] = [family_sizes[str(x)] for x in summary_df["family"]]
    summary_df["minimum_cluster_pFWE_holm_within_family"] = np.nan
    for family, family_maps in MAP_FAMILIES.items():
        idx = summary_df.index[summary_df["family"].eq(family)].to_numpy()
        if set(summary_df.loc[idx, "map_key"]) != set(family_maps):
            raise RuntimeError(f"Incomplete family {family}")
        raw = summary_df.loc[idx, "minimum_cluster_pFWE"].to_numpy(float)
        order = np.argsort(raw)
        adjusted_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
        adjusted = np.empty_like(raw)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
        summary_df.loc[idx, "minimum_cluster_pFWE_holm_within_family"] = adjusted
    summary_df["survives_holm_05"] = summary_df["minimum_cluster_pFWE_holm_within_family"] < .05
    summary_df.to_csv(GROUP / "contrast_summary.csv", index=False)
    cluster_df = pd.concat(all_clusters, ignore_index=True) if all_clusters else pd.DataFrame()
    if not cluster_df.empty:
        family_by_map = dict(zip(summary_df["map_key"], summary_df["family"]))
        family_size_by_map = dict(zip(summary_df["map_key"], summary_df["family_size"]))
        holm_by_map = dict(zip(
            summary_df["map_key"], summary_df["minimum_cluster_pFWE_holm_within_family"]
        ))
        survives_by_map = dict(zip(summary_df["map_key"], summary_df["survives_holm_05"]))
        cluster_df["family"] = [family_by_map[str(x)] for x in cluster_df["map_key"]]
        cluster_df["family_size"] = [family_size_by_map[str(x)] for x in cluster_df["map_key"]]
        cluster_df["map_minimum_cluster_pFWE_holm_within_family"] = [
            holm_by_map[str(x)] for x in cluster_df["map_key"]
        ]
        cluster_df["map_survives_holm_05"] = [survives_by_map[str(x)] for x in cluster_df["map_key"]]
        cluster_df["cluster_pFWE_bonferroni_within_family"] = np.minimum(
            cluster_df["cluster_pFWE"] * cluster_df["family_size"], 1.0
        )
        cluster_df["cluster_survives_bonferroni_within_family_05"] = (
            cluster_df["cluster_pFWE_bonferroni_within_family"] < .05
        )
    cluster_df.to_csv(GROUP / "all_cdt_clusters.csv", index=False)
    within_map = cluster_df[cluster_df["survives_cluster_pFWE_05"]] if not cluster_df.empty else cluster_df
    within_map.to_csv(GROUP / "significant_clusters_within_map.csv", index=False)
    holm_gated = within_map[within_map["map_survives_holm_05"]] if not within_map.empty else within_map
    holm_gated.to_csv(GROUP / "clusters_within_holm_significant_maps.csv", index=False)
    pd.DataFrame(manifests).to_csv(GROUP / "input_manifest.csv", index=False)

    mask = nib.load(COMMON_MASK)
    metadata = {
        "analysis": "joint nonoverlapping Search + Pre-response contrast extension without Search x RT modulators",
        "subjects": SUBJECTS, "n_subjects": len(SUBJECTS), "degrees_of_freedom": len(SUBJECTS)-1,
        "event_definition": {
            "search": "condition-specific boxcar from stimulus onset to first button press minus 0.5 s",
            "pre_response": "condition-specific 0.5-s boxcar ending at first button press",
            "pre_duration_s": PRE_DURATION,
            "search_pre_overlap": "none in neural event specification; HRF-convolved columns remain correlated",
            "search_rt_amplitude_modulators": "omitted",
            "no_insight": "one pooled nuisance regressor, duration up to 29 s and capped at the next trial onset",
            "post_first_response": "not modeled; inference is response-proximal rather than uniquely pre-insight",
        },
        "first_level": {"TR": TR, "smoothing_fwhm_mm": FWHM, "hrf_model": HRF,
                        "noise_model": NOISE, "high_pass_hz": HIGH_PASS,
                        "confounds": "six rigid-body motion regressors"},
        "group_inference": {"n_permutations": n_perm, "one_sided": True, "cdt_p": CDT_P,
                            "cluster_forming_t": t_threshold, "cluster_connectivity": 6,
                            "cluster_significance": "pFWE < .05 based on maximum cluster extent within map",
                            "uncorrected_reporting": "all positive-direction connected components at one-sided voxel p < .001, with no extent filter",
                            "across_map_correction": "Holm within each declared family; cluster-specific Bonferroni-within-family values also saved",
                            "families": MAP_FAMILIES,
                            "contrast_definitions": CONTRASTS,
                            "random_seed_base": random_state},
        "mask": {"path": str(COMMON_MASK), "sha256": sha256(COMMON_MASK),
                 "voxels": int(np.asarray(mask.dataobj).astype(bool).sum()),
                 "voxelwise_n": len(SUBJECTS), "construction": "intersection of the 19 fitted first-level masks",
                 "voxel_volume_mm3": float(abs(np.linalg.det(mask.affine[:3, :3])))},
        "first_level_build": {"analysis_version": ANALYSIS_VERSION,
                              "implementation_sha256": sha256(Path(__file__)),
                              "behavior_sha256": sha256(BEHAVIOR),
                              "common_mask_qc": common_mask_qc},
        "software": {"python": platform.python_version(), "nilearn": nilearn.__version__,
                     "nibabel": nib.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[group] outputs: {GROUP}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["first", "group", "all"], default="all")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--n-jobs-first", type=int, default=2)
    parser.add_argument("--n-jobs-group", type=int, default=8)
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=20260827)
    args = parser.parse_args()

    if args.step in ("first", "all"):
        selected = args.subjects if args.subjects else SUBJECTS
        invalid = sorted(set(selected) - set(SUBJECTS))
        if invalid:
            raise ValueError(f"Subjects outside locked N=19 sample: {invalid}")
        qcs = Parallel(n_jobs=args.n_jobs_first)(delayed(first_level_subject)(subject) for subject in selected)
        qc_path = DESIGN_QC / "design_qc_all_subjects.csv"
        if qc_path.exists() and args.subjects:
            old = pd.read_csv(qc_path)
            new = pd.DataFrame(qcs)
            out = pd.concat([old[~old["subject"].isin(new["subject"])], new], ignore_index=True)
        else:
            out = pd.DataFrame(qcs)
        out.sort_values("subject").to_csv(qc_path, index=False)
        print(out[["subject", "full_rank", "max_abs_task_correlation",
                   "corr_search_pre_CA", "corr_search_pre_JX", "corr_search_pre_OI",
                   "vif_pre_CA", "vif_pre_JX", "vif_pre_OI"]].to_string(index=False))
        if not args.subjects:
            validate_and_build_common_mask()
    if args.step in ("group", "all"):
        run_group(args.n_perm, args.n_jobs_group, args.random_state)


if __name__ == "__main__":
    main()
