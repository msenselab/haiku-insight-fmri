#!/usr/bin/env python3
"""Execution adapter for frozen CSV-limited all-button joint Search/Pre contrasts.

It reads immutable N19 and completed sub-005 AR(1) caches, validates the saved
pre-BOLD designs against every cache regression model, reproduces archived Pre
maps before using other coefficients, and performs the frozen 50k group tests.
"""
from __future__ import annotations
import os
import argparse, hashlib, json, os, re, shutil
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import joblib
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from nilearn.glm.contrasts import compute_contrast
from nilearn.maskers import NiftiMasker
from nilearn.glm.second_level import non_parametric_inference
from scipy import ndimage
from scipy.stats import norm, t as tdist

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
OUT = ROOT / 'glm_unified/allbutton_joint_complete_contrasts_20260916'
SRC = ROOT / 'glm_unified/preresponse_with_search_first_vs_all_buttons_20260903'
EXT = ROOT / 'glm_unified/allbutton_joint_n20_20260916'
SRC_SCRIPT = ROOT / 'glm_unified/scripts/run_preresponse_with_search_first_vs_all_buttons_20260903.py'
SRC_CACHE = SRC / '_nilearn_cache/joblib/nilearn/glm/first_level/first_level/run_glm'
EXT_CACHE = EXT / '_nilearn_cache/joblib/nilearn/glm/first_level/first_level/run_glm'
N19 = [f'sub-{x:03d}' for x in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
N20 = N19[:4] + ['sub-005'] + N19[4:]
PAIRS = [('CA','JX'), ('CA','OI'), ('JX','OI')]
# Keep the historical pair ordering while declaring every directed phase map.
PHASE = []
for phase in ('search', 'pre'):
    for a,b in PAIRS:
        PHASE.extend([(f'{phase}_{a}_gt_{b}', phase, {f'{phase}_{a}':1., f'{phase}_{b}':-1.}, False),
                      (f'{phase}_{b}_gt_{a}', phase, {f'{phase}_{b}':1., f'{phase}_{a}':-1.}, False)])
INTER = [(f'pre_minus_search_{a}_gt_{b}', 'interaction',
          {f'pre_{a}':1., f'pre_{b}':-1., f'search_{a}':-1., f'search_{b}':1.}, True)
         for a,b in PAIRS]
CONTRASTS = PHASE + INTER
KEYS = [x[0] for x in CONTRASTS]
MAPINFO = {x[0]: {'phase':x[1], 'weights':x[2], 'two_sided':x[3]} for x in CONTRASTS}
N_PERM = 50000
CDT = .001
NJOBS = 8
SEEDS = {k:2026091600+i*9973 for i,k in enumerate(KEYS)}
CLUSTER_COLUMNS = ['contrast_key','cluster_id','cluster_sign','cluster_voxels','mni_x','mni_y','mni_z',
                   'peak_t','peak_z','cluster_pFWE_raw','within_map_cluster_FWE05','two_sided','cdt_t']
CLUSTER_DTYPES = {'contrast_key':'string','cluster_id':'int64','cluster_sign':'string','cluster_voxels':'int64',
                  'mni_x':'float64','mni_y':'float64','mni_z':'float64','peak_t':'float64','peak_z':'float64',
                  'cluster_pFWE_raw':'float64','within_map_cluster_FWE05':'bool','two_sided':'bool','cdt_t':'float64'}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stable_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save(array, ref, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RuntimeError(f'nonfinite output {path}')
    nib.Nifti1Image(array, ref.affine).to_filename(path)
    image = nib.load(path)
    if image.get_data_dtype() != np.dtype('float64') or not np.isfinite(np.asarray(image.dataobj)).all():
        raise RuntimeError(f'float64 readback failed {path}')


def grid(a, b) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-8, rtol=0)


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    result = np.empty_like(values)
    result[order] = np.minimum(1, np.maximum.accumulate((len(values)-np.arange(len(values))) * values[order]))
    return result


def mask_for(subject):
    return ROOT/'derivatives/fmriprep'/subject/'func'/f'{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'


def bold_for(subject):
    return ROOT/'derivatives/fmriprep'/subject/'func'/f'{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'


def dm_for(subject):
    return EXT/'first_level/sub-005/sub-005_design_matrix.csv' if subject == 'sub-005' else SRC/'design_only/all_buttons'/f'{subject}_design_matrix.csv'


def known_pre(subject, key):
    return EXT/'first_level/sub-005'/f'sub-005_{key}_effect.nii.gz' if subject == 'sub-005' else SRC/'first_level/all_buttons'/subject/f'{subject}_{key}_effect.nii.gz'


def out_effect(subject, key):
    return OUT/'first_level'/subject/f'{subject}_{key}_effect.nii.gz'


def _raw_cache_records(cache: Path):
    records = []
    for meta in sorted(cache.glob('*/metadata.json')):
        data = json.loads(meta.read_text())
        args = data.get('input_args', {})
        y, x = args.get('Y', ''), args.get('X', '')
        ym = re.search(r'shape=\((\d+),\s*(\d+)\)', y)
        xm = re.search(r'shape=\((\d+),\s*(\d+)\)', x)
        output = meta.parent/'output.pkl'
        if ym and xm and output.exists():
            stat = output.stat()
            records.append({'p':str(output), 'nscan':int(ym.group(1)), 'nvox':int(ym.group(2)),
                            'ncol':int(xm.group(2)), 'metadata_sha256':sha(meta),
                            'p_size':stat.st_size, 'p_mtime_ns':stat.st_mtime_ns})
    return records


def cache_records() -> tuple[list[dict], list[dict]]:
    """Hash each cache PKL once, then reuse hash records on all later stages/runs."""
    OUT.mkdir(parents=True, exist_ok=True)
    record_path = OUT/'cache_records.json'
    existing = json.loads(record_path.read_text()) if record_path.exists() else {'schema_version':2, 'source':[], 'sub005':[]}
    source_now, sub_now = _raw_cache_records(SRC_CACHE), _raw_cache_records(EXT_CACHE)
    for label, now in [('source', source_now), ('sub005', sub_now)]:
        prior = {r['p']:r for r in existing.get(label, [])}
        for record in now:
            old = prior.get(record['p'])
            if old and all(old.get(k) == record[k] for k in ('nscan','nvox','ncol','metadata_sha256','p_size','p_mtime_ns')) and old.get('cache_sha256'):
                record['cache_sha256'] = old['cache_sha256']
            else:
                record['cache_sha256'] = sha(Path(record['p']))
                # Persist completed hashes incrementally: interruption never forces a full rehash.
                current = {r['p']:r for r in existing.get(label, [])}
                current[record['p']] = record.copy()
                existing[label] = [current[k] for k in sorted(current)]
                existing['schema_version'] = 2
                write_json(record_path, existing)
        existing[label] = [dict(r) for r in now]
        existing['schema_version'] = 2
        write_json(record_path, existing)
    if len(source_now) != 38 or len(sub_now) != 1:
        raise RuntimeError(f'unexpected cache records: source={len(source_now)}, sub005={len(sub_now)}')
    return source_now, sub_now


def select_cache(subject, records):
    dm = pd.read_csv(dm_for(subject))
    nscan = nib.load(bold_for(subject)).shape[-1]
    nvox = int(np.asarray(nib.load(mask_for(subject)).dataobj).astype(bool).sum())
    hits = [r for r in records if (r['nscan'],r['nvox'],r['ncol']) == (nscan,nvox,len(dm.columns))]
    if len(hits) != 1:
        raise RuntimeError(f'{subject}: expected one cache record, got {len(hits)}')
    return hits[0], dm


def validate_cached_design(subject, labels, results, dm):
    expected = dm.to_numpy(dtype=np.float64)
    if labels.shape[0] != int(np.asarray(nib.load(mask_for(subject)).dataobj).astype(bool).sum()):
        raise RuntimeError(f'{subject}: cached labels do not match subject mask')
    if not set(np.asarray(labels).astype(str)).issubset(set(map(str, results))):
        raise RuntimeError(f'{subject}: cached labels/results mismatch')
    checks = []
    for rho, result in results.items():
        got = np.asarray(result.model.design, dtype=np.float64)
        if got.shape != expected.shape:
            raise RuntimeError(f'{subject}/{rho}: cache design shape {got.shape} != saved {expected.shape}')
        err = float(np.max(np.abs(got - expected))) if got.size else 0.
        if err > 1e-12:
            raise RuntimeError(f'{subject}/{rho}: cache design differs from saved design ({err})')
        checks.append({'subject':subject, 'rho':str(rho), 'nscan':got.shape[0], 'ncol':got.shape[1],
                       'max_abs_difference_cache_vs_saved_design':err, 'passes_exact':True})
    if not checks:
        raise RuntimeError(f'{subject}: empty cache regression results')
    return checks


def effects_from_loaded_cache(subject, labels, results, dm, masker):
    arrays = {}
    for key, _, weights, _ in CONTRASTS:
        contrast = np.asarray([weights.get(column, 0.) for column in dm.columns], dtype=float)
        if not np.any(contrast):
            raise RuntimeError(f'{subject} {key}: zero contrast')
        effect = compute_contrast(labels, results, contrast, 't').effect_size()
        arrays[key] = np.asarray(masker.inverse_transform(effect).get_fdata(), dtype=np.float64)
    return arrays


def contrast_plan():
    return [{'key':k, 'phase':phase, 'weights':weights, 'two_sided':two}
            for k,phase,weights,two in CONTRASTS]


def preflight(njobs):
    OUT.mkdir(parents=True, exist_ok=True)
    source_freeze = json.loads((SRC/'design_only/frozen_design.json').read_text())
    if source_freeze['subjects'] != N19 or sha(SRC_SCRIPT) != source_freeze['implementation_sha256']:
        raise RuntimeError('immutable source identity/hash failed')
    n19mask = nib.load(SRC/'masks/common_n19_input_mask_intersection.nii.gz')
    n20mask = nib.load(EXT/'masks/new_n20_input_mask_intersection.nii.gz')
    a, b = np.asarray(n19mask.dataobj).astype(bool), np.asarray(n20mask.dataobj).astype(bool)
    if a.sum() != 52787 or b.sum() != 52486 or not np.all(~b | a):
        raise RuntimeError('frozen masks invalid')
    sourceq = pd.read_csv(SRC/'design_only/design_qc_all_buttons.csv')
    if not bool(sourceq.passes_design_gate.all()):
        raise RuntimeError('source N19 original-basis gate failed')
    subq = json.loads((EXT/'FROZEN_PREFLIGHT.json').read_text())['sub005_design_gate']
    if not (subq['full_rank'] and subq['target_columns_complete'] and subq['passes_before_BOLD']):
        raise RuntimeError('sub005 gate failed')
    for subject in N20:
        dm = pd.read_csv(dm_for(subject)); x = dm.to_numpy(float)
        needed = [f'{phase}_{condition}' for phase in ('search','pre') for condition in ('CA','JX','OI')]
        if any(col not in dm for col in needed) or np.linalg.matrix_rank(x) != len(dm.columns):
            raise RuntimeError(f'{subject}: required columns/rank failed')
    records, rec5 = cache_records()
    runner = Path(__file__)
    adapter = {'status':'FROZEN', 'adapter_role':'cache-object execution adapter only',
               'previous_runner':str(OUT/'run_complete_contrasts_before_engine_fix.py'),
               'previous_runner_sha256':sha(OUT/'run_complete_contrasts_before_engine_fix.py'),
               'runner':str(runner), 'runner_sha256':sha(runner),
               'unchanged_analysis_parameters':{'n_permutations':N_PERM,'CDT_p':CDT,'n_jobs':njobs,
                 'source_specification':'CSV clock; 30-s no-response duration; no RT modifier',
                 'samples':{'N19':N19,'N20':N20}}}
    write_json(OUT/'execution_adapter.json', adapter)
    report = {'status':'PASS','runner_sha256':sha(runner),'execution_adapter_sha256':sha(OUT/'execution_adapter.json'),
              'source_runner_sha256':sha(SRC_SCRIPT),'source_frozen_design_sha256':sha(SRC/'design_only/frozen_design.json'),
              'n19_subjects':N19,'n20_subjects':N20,'mask_n19_voxels':int(a.sum()),'mask_n20_voxels':int(b.sum()),
              'no_zero_padding':True,'source_cache_records':len(records),'sub005_cache_records':len(rec5),
              'n_permutations':N_PERM,'CDT_p':CDT,'cluster_connectivity':6,'n_jobs':njobs,
              'contrast_plan':contrast_plan(),'contrast_plan_sha256':stable_sha(contrast_plan()),
              'families_frozen':{'phase_directed_search':6,'phase_directed_pre':6,'phase_directed_all':12,
                                  'interactions_two_sided':3,'expanded_all_maps_per_sample':15},'seeds':SEEDS,
              'sub005_caveat':'8 responses (CA3/JX3/OI2), 34/42 no-response; sparse-response precision caveat, not an exclusion'}
    write_json(OUT/'FROZEN_PREFLIGHT.json', report)
    shutil.copy2(SRC_SCRIPT, OUT/'code_snapshot_source_runner.py')
    shutil.copy2(runner, OUT/'code_snapshot_runner.py')
    print(json.dumps({'stage':'preflight','status':'PASS','source_cache_records':len(records),'sub005_cache_records':len(rec5),
                      'runner_sha256':report['runner_sha256'],'contrast_plan_sha256':report['contrast_plan_sha256']}, indent=2))


def check_frozen(njobs):
    frozen = json.loads((OUT/'FROZEN_PREFLIGHT.json').read_text())
    if frozen.get('status') != 'PASS' or frozen.get('runner_sha256') != sha(Path(__file__)):
        raise RuntimeError('runner was not frozen before analysis')
    if frozen.get('n_jobs') != njobs or frozen.get('contrast_plan_sha256') != stable_sha(contrast_plan()):
        raise RuntimeError('frozen n_jobs or contrast plan differs')
    return frozen


def extract(njobs):
    check_frozen(njobs)
    rows, reproduction, design_checks = [], [], []
    source_records, sub005_records = cache_records()
    for subject in N20:
        record, dm = select_cache(subject, sub005_records if subject == 'sub-005' else source_records)
        labels, results = joblib.load(record['p'])  # exactly one load per subject
        design_checks.extend(validate_cached_design(subject, labels, results, dm))
        masker = NiftiMasker(mask_img=str(mask_for(subject))).fit()
        arrays = effects_from_loaded_cache(subject, labels, results, dm, masker)
        # Trust no other cache-derived coefficient until both archived positive Pre maps reproduce exactly.
        for key in ('pre_CA_gt_JX', 'pre_CA_gt_OI'):
            diff = float(np.max(np.abs(arrays[key] - np.asarray(nib.load(known_pre(subject,key)).dataobj, dtype=np.float64))))
            reproduction.append({'subject':subject,'contrast_key':key,'max_abs_difference_saved_positive_pre':diff,
                                 'passes_exact':diff <= 1e-10})
            if diff > 1e-10:
                raise RuntimeError(f'{subject} {key}: cache extraction did not reproduce archived map ({diff})')
        for key in KEYS:
            path = out_effect(subject, key)
            save(arrays[key], nib.load(mask_for(subject)), path)
            rows.append({'subject':subject,'contrast_key':key,'path':str(path),'sha256':sha(path),'cache_path':record['p'],
                         'cache_sha256':record['cache_sha256'],'cache_metadata_sha256':record['metadata_sha256'],
                         'two_sided':MAPINFO[key]['two_sided']})
        del arrays, masker, labels, results
    if len(rows) != 300 or len(reproduction) != 40 or not all(x['passes_exact'] for x in reproduction):
        raise RuntimeError(f'extraction completeness failed: maps={len(rows)} reproduction={len(reproduction)}')
    pd.DataFrame(rows).to_csv(OUT/'first_level_manifest.csv', index=False)
    pd.DataFrame(reproduction).to_csv(OUT/'pre_reproduction.csv', index=False)
    pd.DataFrame(design_checks).to_csv(OUT/'cache_design_validation.csv', index=False)
    verification = {'status':'PASS','n_effect_maps':len(rows),'n_subjects':len(N20),'n_contrasts':len(KEYS),
                    'n_pre_reproductions':len(reproduction),'max_abs_difference':max(x['max_abs_difference_saved_positive_pre'] for x in reproduction),
                    'source_n19_subjects_with_cache_design_validation':len(set(x['subject'] for x in design_checks if x['subject'] != 'sub-005')),
                    'sub005_cache_design_validation':any(x['subject']=='sub-005' for x in design_checks),
                    'cache_regression_models_checked':len(design_checks),'cache_loaded_once_per_subject':True}
    write_json(OUT/'cache_extraction_verification.json', verification)
    print(json.dumps({'stage':'extract', **verification}, indent=2))


def manual_t(paths, mask):
    support = np.asarray(mask.dataobj).astype(bool)
    stack = np.stack([np.asarray(nib.load(path).dataobj, dtype=np.float64) for path in paths])
    sd = stack.std(0, ddof=1)
    tmap = np.divide(stack.mean(0), sd/np.sqrt(len(paths)), out=np.zeros_like(sd), where=sd > 0)
    tmap[~support] = 0.
    return tmap, stack


def signed_components(tmap, support, threshold, two_sided):
    structure = ndimage.generate_binary_structure(3, 1)
    pos, npos = ndimage.label((tmap > threshold) & support, structure)
    if not two_sided:
        return pos, npos
    neg, nneg = ndimage.label((tmap < -threshold) & support, structure)
    labels = pos.copy()
    labels[neg > 0] = neg[neg > 0] + npos
    return labels, npos+nneg


def typed_clusters(rows):
    frame = pd.DataFrame(rows, columns=CLUSTER_COLUMNS)
    for column, dtype in CLUSTER_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    return frame


def clusters(key, tmap, zmap, size, logp, mask, two_sided, df):
    support = np.asarray(mask.dataobj).astype(bool)
    threshold = float(tdist.isf(CDT/(2 if two_sided else 1), df))
    labels, nclusters = signed_components(tmap, support, threshold, two_sided)
    saved_size, saved_logp = np.asarray(size.get_fdata(),float), np.asarray(logp.get_fdata(),float)
    if not np.array_equal(labels > 0, saved_size > 0):
        raise RuntimeError(f'{key}: signed manual/Nilearn restricted-mask CDT mismatch')
    rows = []
    for cluster_id in range(1, nclusters+1):
        component = labels == cluster_id
        extent = int(component.sum())
        unique_size = np.unique(saved_size[component])
        if len(unique_size) != 1 or int(round(unique_size[0])) != extent:
            raise RuntimeError(f'{key}: within-mask exact cluster-size mismatch')
        sign = 'positive' if float(np.mean(tmap[component])) > 0 else 'negative'
        score = tmap if sign == 'positive' else -tmap
        peak = np.argwhere(component)[int(np.argmax(score[component]))]
        xyz = nib.affines.apply_affine(mask.affine, peak)
        pvalue = float(10 ** (-saved_logp[component].max()))
        rows.append({'contrast_key':key,'cluster_id':cluster_id,'cluster_sign':sign,'cluster_voxels':extent,
                     'mni_x':xyz[0],'mni_y':xyz[1],'mni_z':xyz[2],'peak_t':tmap[tuple(peak)],'peak_z':zmap[tuple(peak)],
                     'cluster_pFWE_raw':pvalue,'within_map_cluster_FWE05':pvalue < .05,'two_sided':two_sided,'cdt_t':threshold})
    return typed_clusters(rows), threshold


def reconstruct_group(sample, base, subjects, mask):
    """Recreate all sidecars from map directories on every resume, never trust sentinels alone."""
    summaries, cluster_frames, manifests = [], [], []
    support = np.asarray(mask.dataobj).astype(bool)
    for key in KEYS:
        folder = base/key
        done_path = folder/'completion.json'
        if not done_path.exists():
            raise RuntimeError(f'{sample} {key}: missing completion')
        done = json.loads(done_path.read_text())
        if done.get('summary',{}).get('contrast_key') != key or done.get('n_perm') != N_PERM:
            raise RuntimeError(f'{sample} {key}: invalid completion record')
        expected_outputs = done.get('output_keys', [])
        required = [folder/'manual_t.nii.gz',folder/'manual_z.nii.gz',folder/'input_effect_stack.npz',folder/'all_cdt_clusters.csv'] + [folder/f'{name}_finite_float64.nii.gz' for name in expected_outputs]
        if not all(path.exists() for path in required):
            raise RuntimeError(f'{sample} {key}: incomplete map directory')
        stack = np.load(folder/'input_effect_stack.npz', allow_pickle=False)
        if not np.array_equal(stack['subjects'].astype(str), np.asarray(subjects)) or stack['effects'].shape != (len(subjects), int(support.sum())):
            raise RuntimeError(f'{sample} {key}: input checkpoint incomplete')
        cluster = pd.read_csv(folder/'all_cdt_clusters.csv', dtype=CLUSTER_DTYPES)
        if list(cluster.columns) != CLUSTER_COLUMNS:
            raise RuntimeError(f'{sample} {key}: cluster CSV schema invalid')
        cluster_frames.append(cluster)
        input_hashes = done.get('input_hashes', {})
        if len(input_hashes) != len(subjects):
            raise RuntimeError(f'{sample} {key}: incomplete input hash record')
        for subject in subjects:
            path = out_effect(subject,key)
            if str(path) not in input_hashes or not path.exists() or sha(path) != input_hashes[str(path)]:
                raise RuntimeError(f'{sample} {key}: input hash mismatch {subject}')
            manifests.append({'sample':sample,'subject':subject,'contrast_key':key,'path':str(path),'sha256':input_hashes[str(path)]})
        summaries.append(done['summary'])
    summary = pd.DataFrame(summaries).sort_values('contrast_key').reset_index(drop=True)
    raw = summary.minimum_cluster_pFWE_raw.to_numpy(float)
    summary['holm_all_15_maps'] = holm(raw)
    summary['holm_phase_directed_6'] = np.nan
    for phase in ('search','pre'):
        idx = (summary.phase == phase).to_numpy()
        summary.loc[idx,'holm_phase_directed_6'] = holm(raw[idx])
    idx = summary.phase.eq('interaction').to_numpy()
    summary.loc[idx,'holm_interaction_two_sided_3'] = holm(raw[idx])
    summary.to_csv(base/'contrast_summary_correction_ledgers.csv', index=False)
    all_clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else typed_clusters([])
    all_clusters.to_csv(base/'all_cdt_clusters.csv', index=False)
    manifest = pd.DataFrame(manifests)
    if len(manifest) != len(subjects)*len(KEYS):
        raise RuntimeError(f'{sample}: reconstructed manifest has {len(manifest)} rows')
    manifest.to_csv(base/'input_manifest.csv', index=False)
    write_json(base/'metadata.json', {'status':'COMPLETE','sample':sample,'subjects':subjects,'mask_voxels':int(support.sum()),
               'n_perm':N_PERM,'CDT_p':CDT,'cluster_connectivity':6,'families':{'all_maps':15,'search_directed':6,'pre_directed':6,'interactions_two_sided':3},
               'seeds':{k:SEEDS[k]+(0 if sample=='N19' else 500000) for k in KEYS},'n_jobs':json.loads((OUT/'FROZEN_PREFLIGHT.json').read_text())['n_jobs'],
               'reconstructed_from_completed_map_directories':True,'input_manifest_rows':len(manifest)})
    return summary


def group(sample, njobs):
    check_frozen(njobs)
    subjects = N19 if sample == 'N19' else N20
    mask = nib.load(SRC/'masks/common_n19_input_mask_intersection.nii.gz' if sample == 'N19' else EXT/'masks/new_n20_input_mask_intersection.nii.gz')
    support = np.asarray(mask.dataobj).astype(bool)
    base = OUT/'group_inference'/sample
    base.mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        info, folder = MAPINFO[key], base/key
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'completion.json').exists():
            continue
        paths = [out_effect(subject,key) for subject in subjects]
        for path in paths:
            image = nib.load(path)
            if not (grid(image, mask) and image.get_data_dtype() == np.dtype('float64') and np.isfinite(np.asarray(image.dataobj)).all()):
                raise RuntimeError(f'invalid input {path}')
        tmap, stack = manual_t(paths, mask)
        pvalue = tdist.sf(tmap, len(subjects)-1)
        zmap = norm.isf(np.clip(pvalue, np.finfo(float).tiny, 1-np.finfo(float).eps))
        zmap[~support] = 0.
        save(tmap, mask, folder/'manual_t.nii.gz')
        save(zmap, mask, folder/'manual_z.nii.gz')
        np.savez_compressed(folder/'input_effect_stack.npz', effects=stack[:,support], subjects=np.asarray(subjects),
                            mask_sha256=sha(SRC/'masks/common_n19_input_mask_intersection.nii.gz' if sample == 'N19' else EXT/'masks/new_n20_input_mask_intersection.nii.gz'))
        outcome = non_parametric_inference([str(p) for p in paths], design_matrix=pd.DataFrame({'intercept':np.ones(len(subjects))}),
                  second_level_contrast='intercept', mask=str(SRC/'masks/common_n19_input_mask_intersection.nii.gz' if sample == 'N19' else EXT/'masks/new_n20_input_mask_intersection.nii.gz'),
                  smoothing_fwhm=None, model_intercept=True, n_perm=N_PERM, two_sided_test=info['two_sided'],
                  random_state=SEEDS[key]+(0 if sample == 'N19' else 500000), n_jobs=njobs, verbose=1, threshold=CDT, tfce=False)
        perm_t = np.nan_to_num(outcome['t'].get_fdata(), nan=0.)
        max_diff = float(np.max(np.abs(perm_t[support]-tmap[support])))
        if max_diff > 1e-7:
            raise RuntimeError(f'{sample} {key}: manual t mismatch {max_diff}')
        for name,image in outcome.items():
            output = np.nan_to_num(image.get_fdata(), nan=0., posinf=0., neginf=0.)
            output[~support] = 0.
            save(output, mask, folder/f'{name}_finite_float64.nii.gz')
        cluster_frame, threshold = clusters(key,tmap,zmap,outcome['size'],outcome['logp_max_size'],mask,info['two_sided'],len(subjects)-1)
        cluster_frame.to_csv(folder/'all_cdt_clusters.csv', index=False)
        summary = {'contrast_key':key,'phase':info['phase'],'two_sided':info['two_sided'],'n_cdt_clusters':len(cluster_frame),
                   'minimum_cluster_pFWE_raw':float(cluster_frame.cluster_pFWE_raw.min()) if len(cluster_frame) else 1.,
                   'n_within_map_significant_clusters':int(cluster_frame.within_map_cluster_FWE05.sum()) if len(cluster_frame) else 0,
                   'seed':SEEDS[key]+(0 if sample == 'N19' else 500000),'manual_permutation_t_max_abs_diff':max_diff,'cdt_t':threshold}
        write_json(folder/'completion.json', {'summary':summary,'input_hashes':{str(path):sha(path) for path in paths},
                   'n_perm':N_PERM,'n_jobs':njobs,'output_keys':sorted(outcome.keys()),
                   'raw_null_note':'Nilearn non_parametric_inference does not expose max-extent draws; size/logp maps are its 50k-draw FWE result.'})
    summary = reconstruct_group(sample, base, subjects, mask)
    print(json.dumps({'stage':f'group-{sample.lower()}','status':'COMPLETE','map_count':len(summary),
                      'manifest_rows':len(subjects)*len(KEYS),'cluster_rows':int(pd.read_csv(base/'all_cdt_clusters.csv').shape[0])}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight','extract','group-n19','group-n20','all'], default='all')
    parser.add_argument('--n-jobs', type=int, default=NJOBS)
    args = parser.parse_args()
    if args.n_jobs != NJOBS:
        raise RuntimeError(f'frozen job partition is {NJOBS}; requested {args.n_jobs}')
    if args.stage in ('preflight','all'): preflight(args.n_jobs)
    if args.stage in ('extract','all'): extract(args.n_jobs)
    if args.stage in ('group-n19','all'): group('N19',args.n_jobs)
    if args.stage in ('group-n20','all'): group('N20',args.n_jobs)

if __name__ == '__main__':
    main()
