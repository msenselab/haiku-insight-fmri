#!/usr/bin/env python3
"""Frozen N=19 same-specification group sensitivity for archived Whole Search.

This intentionally reuses only the archived N=20 Search first-level effect maps,
drops sub-005, and fixes the N=20 analysis mask.  It never fits BOLD data or
creates events/design matrices.
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
from scipy import ndimage
from scipy.stats import t as t_distribution

BASE = Path(os.environ["HAIKU_PROJECT_ROOT"])
SOURCE = BASE / 'glm_unified/search_only_glm_cluster_fwe_20260905'
OUT = BASE / 'glm_unified/whole_search_matched_n19_20260916'
N20_META = SOURCE / 'ca_jx/metadata.json'
N20_MASK = SOURCE / 'ca_jx/analysis_mask.nii.gz'
SOURCE_RUNNER = SOURCE / 'run_search_only_glm_cluster_fwe.py'
SOURCE_FIRST_LEVEL_CODE = BASE / 'code/insight_search_glm.py'
EXCLUDE = 'sub-005'
CONTRASTS = {'CA_gt_JX': 'Search CA > JX', 'CA_gt_OI': 'Search CA > OI'}
N_PERM = 50_000
RANDOM_SEED = 20260916
N_JOBS = 8
CDT_P = 0.001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_subjects() -> list[str]:
    subjects = json.loads(N20_META.read_text())['subjects']
    if len(subjects) != 20 or EXCLUDE not in subjects:
        raise RuntimeError('Unexpected N=20 source cohort; cannot derive matched N=19.')
    return [s for s in subjects if s != EXCLUDE]


def input_records(subjects: list[str]) -> list[dict]:
    records = []
    for key in CONTRASTS:
        source_meta = json.loads((SOURCE / ('ca_jx' if key == 'CA_gt_JX' else 'ca_oi') / 'metadata.json').read_text())
        source_by_subject = {row['subject']: row for row in source_meta['inputs']}
        for subject in subjects:
            row = source_by_subject[subject]
            path = Path(row['path'])
            records.append({
                'contrast': key, 'subject': subject, 'path': str(path),
                'source_n20_sha256': row['sha256'], 'current_sha256': sha256(path),
                'size_bytes': path.stat().st_size,
            })
    return records


def frozen_plan() -> dict:
    subjects = source_subjects()
    m2_meta = BASE / 'glm_unified/insight_event_timing_sensitivity/joint_search_pre_phase_contrasts_20260831/metadata.json'
    m2_subjects = json.loads(m2_meta.read_text())['subjects']
    return {
        'analysis_id': 'whole_search_matched_n19_20260916',
        'frozen_before_group': True,
        'purpose': 'sample-drop sensitivity of original Whole Search only',
        'cohort': {
            'n20_source_metadata': str(N20_META), 'n20_subjects': json.loads(N20_META.read_text())['subjects'],
            'excluded_from_n20': EXCLUDE, 'subjects_n19': subjects, 'n_subjects': len(subjects),
            'm2_subject_metadata': str(m2_meta), 'm2_subjects': m2_subjects,
            'exact_id_equality_with_m2': subjects == m2_subjects,
        },
        'first_level_provenance': {
            'map_source': 'archived glm_insight_search effect maps used by original N20 Search FWE output',
            'source_code': str(SOURCE_FIRST_LEVEL_CODE), 'source_code_sha256': sha256(SOURCE_FIRST_LEVEL_CODE),
            'source_runner': str(SOURCE_RUNNER), 'source_runner_sha256': sha256(SOURCE_RUNNER),
            'event_timing': 'responded Search: onset=behavioral trial onset, duration=first_insight_rt; no-insight: onset=trial onset, duration=29 s',
            'SearchRT': 'preserved: condition-specific centered first_insight_rt parametric regressors remain in each archived first-level design',
            'not_m2': 'No response-pre epoch, no M2 common mask, no historical unified Search-phase substitution.',
            'no_bold_refit_or_nuisance_change': True,
        },
        'mask': {
            'path': str(N20_MASK), 'sha256': sha256(N20_MASK),
            'rule': 'fixed original N20 automatic group analysis mask for both contrasts; no N19 recomputation',
            'voxels': int(np.asanyarray(nib.load(N20_MASK).dataobj).astype(bool).sum()),
        },
        'group_inference': {
            'contrasts': CONTRASTS, 'one_sided_positive': True, 'cdt_p': CDT_P,
            'cluster_connectivity': '6-neighbour face-connected', 'cluster_statistic': 'maximum extent',
            'n_permutations': N_PERM, 'random_seed': RANDOM_SEED, 'n_jobs': N_JOBS,
            'within_map_cluster_fwe': True, 'across_map_family': 'Holm across CA_gt_JX and CA_gt_OI map-minimum pFWE values',
        },
        'inputs': input_records(subjects),
        'runner_sha256': sha256(Path(__file__)),
    }


def write_freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan()
    (OUT / 'analysis_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    pd.DataFrame(plan['inputs']).to_csv(OUT / 'input_manifest.csv', index=False)
    print(f'Frozen plan: {OUT / "analysis_plan.json"}')


def load_checked_plan() -> dict:
    plan_path = OUT / 'analysis_plan.json'
    if not plan_path.exists():
        raise RuntimeError('Missing frozen analysis_plan.json; run --freeze first.')
    plan = json.loads(plan_path.read_text())
    if not plan['frozen_before_group'] or plan['runner_sha256'] != sha256(Path(__file__)):
        raise RuntimeError('Frozen plan is absent or runner hash differs; do not group.')
    return plan


def one_sample_t(stack: np.ndarray) -> tuple[np.ndarray, int]:
    """One-sample t, assigning 0 to all-zero fixed-mask voxels.

    Removing sub-005 leaves 91 voxels with exactly zero values in all 19 maps.
    Their 0/0 t statistic is undefined mathematically but represents no effect;
    nilearn's group implementation treats these constant-zero locations as zero.
    A nonzero constant voxel is an error, never silently coerced.
    """
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1)
    zero_sd = sd == 0
    if np.any(zero_sd & (mean != 0)):
        raise RuntimeError('Nonzero constant voxel encountered in fixed N20 mask.')
    t = np.zeros_like(mean, dtype=np.float64)
    np.divide(mean, sd / np.sqrt(stack.shape[0]), out=t, where=~zero_sd)
    return t, int(zero_sd.sum())


def preflight() -> dict:
    plan = load_checked_plan()
    subjects = source_subjects()
    checks = []
    def check(name, passed, detail): checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
    check('exact_n19_equals_m2_ids', subjects == plan['cohort']['m2_subjects'] and plan['cohort']['exact_id_equality_with_m2'], subjects)
    check('n19_is_n20_minus_sub005', subjects == [s for s in plan['cohort']['n20_subjects'] if s != EXCLUDE], EXCLUDE)
    check('original_n20_mask_hash', sha256(N20_MASK) == plan['mask']['sha256'], sha256(N20_MASK))
    mask_img = nib.load(N20_MASK); mask = np.asanyarray(mask_img.dataobj).astype(bool)
    check('original_n20_mask_voxel_count', int(mask.sum()) == plan['mask']['voxels'], int(mask.sum()))
    records = plan['inputs']
    for row in records:
        path = Path(row['path']); current = sha256(path) if path.exists() else None
        check(f"input_hash:{row['contrast']}:{row['subject']}", current == row['current_sha256'] == row['source_n20_sha256'], current)
        if path.exists():
            img = nib.load(path)
            check(f"input_geometry:{row['contrast']}:{row['subject']}", img.shape == mask_img.shape and np.allclose(img.affine, mask_img.affine), {'shape': img.shape})
    # Manual group-t calculation before any permutations, from all N19 source maps in fixed mask.
    t_checks = {}
    for key in CONTRASTS:
        paths = [Path(r['path']) for r in records if r['contrast'] == key]
        stack = np.stack([np.asanyarray(nib.load(p).dataobj)[mask] for p in paths]).astype(np.float64)
        finite = np.isfinite(stack).all()
        t, constant_zero_voxels = one_sample_t(stack)
        t_checks[key] = {'n_maps': len(paths), 'finite_stack': bool(finite), 'manual_t_finite': bool(np.isfinite(t).all()),
                         'constant_zero_voxels_fixed_to_zero': constant_zero_voxels,
                         'manual_t_max': float(np.max(t)), 'manual_t_min': float(np.min(t))}
        check(f'manual_t_stack:{key}', len(paths) == 19 and finite and np.isfinite(t).all(), t_checks[key])
    result = {'passed': all(x['passed'] for x in checks), 'checks': checks, 'manual_t_preflight': t_checks}
    (OUT / 'preflight_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']: raise RuntimeError('Preflight failed; see preflight_verification.json')
    print('Preflight PASS')
    return plan


def components(t_data, threshold, affine, size_data, logp_data, contrast, label, nperm):
    structure = ndimage.generate_binary_structure(3, 1)
    lab, n = ndimage.label(t_data > threshold, structure=structure)
    rows = []; voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    for cid in range(1, n + 1):
        cmask = lab == cid; ijk = np.argwhere(cmask); vals = t_data[cmask]; peak_idx = ijk[int(np.nanargmax(vals))]
        size = int(cmask.sum()); declared_sizes = np.unique(size_data[cmask]); declared_sizes = declared_sizes[declared_sizes > 0]
        if len(declared_sizes) != 1 or int(round(float(declared_sizes[0]))) != size: raise RuntimeError(f'size-map mismatch {contrast}/{cid}')
        xyz = nib.affines.apply_affine(affine, peak_idx); p = float(10 ** (-float(np.nanmax(logp_data[cmask]))))
        rows.append({'Contrast_key': contrast, 'Contrast': label, 'Cluster_ID': cid, 'MNI_x': float(xyz[0]), 'MNI_y': float(xyz[1]), 'MNI_z': float(xyz[2]), 'Peak_t': float(t_data[tuple(peak_idx)]), 'Cluster_size_voxels': size, 'Cluster_size_mm3': size*voxel_volume, 'Cluster_pFWE': p, 'Cluster_pFWE_display': f'{p:.5f}' if p < .001 else f'{p:.3f}', 'Survives_pFWE_lt_05': p < .05})
    return pd.DataFrame(rows, columns=['Contrast_key','Contrast','Cluster_ID','MNI_x','MNI_y','MNI_z','Peak_t','Cluster_size_voxels','Cluster_size_mm3','Cluster_pFWE','Cluster_pFWE_display','Survives_pFWE_lt_05'])


def save_float(img, path):
    hdr=img.header.copy(); hdr.set_data_dtype(np.float64)
    nib.save(nib.Nifti1Image(img.get_fdata(),img.affine,hdr),path)

def run_group() -> None:
    plan = preflight()
    subjects = plan['cohort']['subjects_n19']; mask_img = nib.load(N20_MASK); threshold = float(t_distribution.isf(CDT_P, len(subjects)-1))
    all_clusters=[]; map_min=[]
    for key, label in CONTRASTS.items():
        cdir=OUT/key.lower(); cdir.mkdir(exist_ok=True)
        paths=[Path(r['path']) for r in plan['inputs'] if r['contrast']==key]
        stack=np.stack([np.asanyarray(nib.load(p).dataobj)[np.asanyarray(mask_img.dataobj).astype(bool)] for p in paths]).astype(np.float64)
        np.savez_compressed(cdir/'input_effect_stack.npz', subjects=np.array(subjects), values=stack)
        manual_t, constant_zero_voxels = one_sample_t(stack)
        manual_vol=np.zeros(mask_img.shape,dtype=np.float64); manual_vol[np.asanyarray(mask_img.dataobj).astype(bool)]=manual_t
        save_float(nib.Nifti1Image(manual_vol, mask_img.affine, mask_img.header), cdir/f'{key}_manual_t_from_n19_stack.nii.gz')
        design=pd.DataFrame({'intercept':np.ones(len(subjects))})
        parametric=SecondLevelModel(mask_img=mask_img,smoothing_fwhm=None).fit([str(p) for p in paths],design_matrix=design)
        tparam=parametric.compute_contrast('intercept',output_type='stat'); zimg=parametric.compute_contrast('intercept',output_type='z_score')
        maxerr=float(np.max(np.abs(np.asanyarray(tparam.dataobj)[np.asanyarray(mask_img.dataobj).astype(bool)]-manual_t)))
        if maxerr > 1e-8: raise RuntimeError(f'manual t mismatch {key}: {maxerr}')
        save_float(tparam,cdir/f'{key}_parametric_t.nii.gz'); save_float(zimg,cdir/f'{key}_parametric_z.nii.gz')
        outputs=non_parametric_inference([str(p) for p in paths],design_matrix=design,second_level_contrast='intercept',mask=mask_img,smoothing_fwhm=None,model_intercept=True,n_perm=N_PERM,two_sided_test=False,random_state=RANDOM_SEED,n_jobs=N_JOBS,verbose=1,threshold=CDT_P,tfce=False)
        for output_key,img in outputs.items(): save_float(img,cdir/f'{key}_{output_key}.nii.gz')
        permt=np.asanyarray(outputs['t'].dataobj)
        inside = np.asanyarray(mask_img.dataobj).astype(bool)
        finite_perm = np.isfinite(permt[inside])
        expected_finite = np.ones_like(finite_perm, dtype=bool)
        expected_finite[stack.std(axis=0, ddof=1) == 0] = False
        if not np.array_equal(finite_perm, expected_finite):
            raise RuntimeError(f'unexpected non-finite permutation t locations for {key}')
        permerr=float(np.max(np.abs(permt[inside][finite_perm]-manual_t[finite_perm])))
        if permerr > 1e-5: raise RuntimeError(f'permutation t mismatch {key}: {permerr}')
        table=components(permt,threshold,mask_img.affine,np.asanyarray(outputs['size'].dataobj),np.asanyarray(outputs['logp_max_size'].dataobj),key,label,N_PERM)
        table.to_csv(cdir/f'{key}_all_cdt_clusters.csv',index=False); table.to_csv(cdir/'all_cdt_clusters.csv',index=False)
        sig=table.loc[table.Survives_pFWE_lt_05].copy(); sig.to_csv(cdir/'search_clusterFWE_significant.csv',index=False)
        all_clusters.append(table); map_min.append({'map_key':key,'contrast':label,'cdt_clusters':len(table),'within_map_fwe_clusters':int(len(sig)),'minimum_cluster_pFWE':float(table.Cluster_pFWE.min()) if len(table) else 1.0,'constant_zero_voxels_fixed_mask':constant_zero_voxels,'manual_t_max_abs_error':maxerr,'permutation_t_max_abs_error_nonconstant_voxels':permerr})
    all_df=pd.concat(all_clusters,ignore_index=True); all_df.to_csv(OUT/'all_cdt_clusters.csv',index=False)
    summary=pd.DataFrame(map_min); p=summary.minimum_cluster_pFWE.to_numpy(); order=np.argsort(p); holm=np.empty(len(p)); running=0.
    for rank,idx in enumerate(order): running=max(running,(len(p)-rank)*p[idx]); holm[idx]=min(1.,running)
    summary['holm_p_across_two_search_CA_maps']=holm; summary['survives_two_map_holm_05']=holm < .05; summary.to_csv(OUT/'contrast_summary_holm.csv',index=False)
    metadata={'analysis':'permutation maximum-cluster-extent FWE, N20-mask-fixed N19 sample-drop sensitivity','n_subjects':len(subjects),'subjects':subjects,'degrees_of_freedom':len(subjects)-1,'contrasts':CONTRASTS,'test_direction':'positive, one-sided','cluster_forming_p_uncorrected':CDT_P,'cluster_forming_t':threshold,'cluster_connectivity':'6-neighbour (face-connected)','cluster_alpha_fwe':.05,'cluster_statistic':'extent in voxels','fwe_scope':'separately within each contrast','n_permutations':N_PERM,'random_seed':RANDOM_SEED,'n_jobs':N_JOBS,'mask_path':str(N20_MASK),'mask_sha256':sha256(N20_MASK),'mask_voxels':int(np.asanyarray(mask_img.dataobj).astype(bool).sum()),'input_plan_sha256':sha256(OUT/'analysis_plan.json'),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'nibabel':nib.__version__,'nilearn':nilearn.__version__}}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary.to_string(index=False)); print(f'Complete: {OUT}')

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true'); ap.add_argument('--preflight',action='store_true'); ap.add_argument('--group',action='store_true'); args=ap.parse_args()
    if args.freeze: write_freeze()
    elif args.preflight: preflight()
    elif args.group: run_group()
    else: ap.error('choose --freeze, --preflight, or --group')
