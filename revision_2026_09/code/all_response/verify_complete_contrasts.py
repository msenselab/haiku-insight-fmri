#!/usr/bin/env python3
"""Independent readback verifier for all 30 completed whole-brain contrast maps.

Does not import the execution runner. It rebuilds group t maps from first-level
files, checks signed CDT components/extent maps, all 15-map Holm ledgers, and
resume-reconstructed manifests.
"""
from __future__ import annotations
import os
import hashlib, json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import t as tdist

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
OUT = ROOT/'glm_unified/allbutton_joint_complete_contrasts_20260916'
SRC = ROOT/'glm_unified/preresponse_with_search_first_vs_all_buttons_20260903'
EXT = ROOT/'glm_unified/allbutton_joint_n20_20260916'
N19 = [f'sub-{x:03d}' for x in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
N20 = N19[:4] + ['sub-005'] + N19[4:]
PAIRS = [('CA','JX'),('CA','OI'),('JX','OI')]
KEYS, TWO_SIDED = [], {}
for phase in ('search','pre'):
    for a,b in PAIRS:
        KEYS.extend([f'{phase}_{a}_gt_{b}', f'{phase}_{b}_gt_{a}'])
for a,b in PAIRS:
    key = f'pre_minus_search_{a}_gt_{b}'; KEYS.append(key); TWO_SIDED[key] = True
for key in KEYS:
    TWO_SIDED.setdefault(key, False)
CDT = .001
N_PERM = 50000
CLUSTER_COLUMNS = ['contrast_key','cluster_id','cluster_sign','cluster_voxels','mni_x','mni_y','mni_z',
                   'peak_t','peak_z','cluster_pFWE_raw','within_map_cluster_FWE05','two_sided','cdt_t']
CLUSTER_DTYPES = {'contrast_key':'string','cluster_id':'int64','cluster_sign':'string','cluster_voxels':'int64',
                  'mni_x':'float64','mni_y':'float64','mni_z':'float64','peak_t':'float64','peak_z':'float64',
                  'cluster_pFWE_raw':'float64','within_map_cluster_FWE05':'bool','two_sided':'bool','cdt_t':'float64'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine,b.affine,atol=1e-8,rtol=0)


def holm(values):
    values=np.asarray(values,float); order=np.argsort(values); result=np.empty_like(values)
    result[order]=np.minimum(1,np.maximum.accumulate((len(values)-np.arange(len(values)))*values[order]))
    return result


def signed_components(tmap, support, threshold, two_sided):
    structure=ndimage.generate_binary_structure(3,1)
    positive,npositive=ndimage.label((tmap>threshold)&support,structure)
    if not two_sided:
        return positive,npositive
    negative,nnegative=ndimage.label((tmap<-threshold)&support,structure)
    labels=positive.copy(); labels[negative>0]=negative[negative>0]+npositive
    return labels,npositive+nnegative


def verify_cluster_map(key, tmap, support, affine, folder, df):
    size=np.asarray(nib.load(folder/'size_finite_float64.nii.gz').dataobj,dtype=np.float64)
    logp=np.asarray(nib.load(folder/'logp_max_size_finite_float64.nii.gz').dataobj,dtype=np.float64)
    threshold=float(tdist.isf(CDT/(2 if TWO_SIDED[key] else 1),df))
    labels,ncomponents=signed_components(tmap,support,threshold,TWO_SIDED[key])
    assert np.array_equal(labels>0,size>0), f'{key}: signed CDT support differs from Nilearn size map'
    saved=pd.read_csv(folder/'all_cdt_clusters.csv',dtype=CLUSTER_DTYPES)
    assert list(saved.columns)==CLUSTER_COLUMNS, f'{key}: cluster CSV loses typed empty schema'
    assert len(saved)==ncomponents, f'{key}: CSV/component count mismatch'
    expected=[]
    for cid in range(1,ncomponents+1):
        component=labels==cid; extent=int(component.sum()); unique=np.unique(size[component])
        assert len(unique)==1 and int(round(unique[0]))==extent, f'{key}: within-mask extent mismatch'
        sign='positive' if float(np.mean(tmap[component]))>0 else 'negative'
        score=tmap if sign=='positive' else -tmap
        peak=np.argwhere(component)[int(np.argmax(score[component]))]
        xyz=nib.affines.apply_affine(affine,peak); p=float(10**(-logp[component].max()))
        expected.append({'cluster_id':cid,'cluster_sign':sign,'cluster_voxels':extent,'mni_x':xyz[0],'mni_y':xyz[1],'mni_z':xyz[2],
                         'peak_t':tmap[tuple(peak)],'cluster_pFWE_raw':p})
    expected=pd.DataFrame(expected)
    if ncomponents:
        columns=['cluster_id','cluster_voxels','mni_x','mni_y','mni_z','peak_t','cluster_pFWE_raw']
        got=saved[columns].sort_values('cluster_id').reset_index(drop=True)
        want=expected[columns].sort_values('cluster_id').reset_index(drop=True)
        assert np.allclose(got.to_numpy(float),want.to_numpy(float),atol=1e-10,rtol=0), f'{key}: cluster CSV differs'
        assert (saved.cluster_sign.to_numpy()==expected.sort_values('cluster_id').cluster_sign.to_numpy()).all(), f'{key}: signed cluster labels differ'
    return {'n_cdt_clusters':ncomponents,'cdt_t':threshold,'signed_components_exact':True,'within_mask_sizes_exact':True,
            'typed_empty_csv_valid':ncomponents != 0 or list(saved.columns)==CLUSTER_COLUMNS}


def verify_sample(sample, subjects, mask_path):
    mask=nib.load(mask_path); support=np.asarray(mask.dataobj).astype(bool); base=OUT/'group_inference'/sample
    metadata=json.loads((base/'metadata.json').read_text())
    assert metadata['status']=='COMPLETE' and metadata['n_perm']==N_PERM and metadata['input_manifest_rows']==len(subjects)*15
    manifest=pd.read_csv(base/'input_manifest.csv')
    assert len(manifest)==len(subjects)*15 and not manifest.duplicated(['subject','contrast_key']).any()
    map_checks=[]; raw=[]; zero_checked=0
    for key in KEYS:
        folder=base/key; done=json.loads((folder/'completion.json').read_text())
        assert done['n_perm']==N_PERM and done['summary']['contrast_key']==key and len(done['input_hashes'])==len(subjects)
        subset=manifest[manifest.contrast_key==key].set_index('subject').loc[subjects].reset_index()
        assert len(subset)==len(subjects)
        images=[nib.load(path) for path in subset.path]
        assert all(grid(mask,image) for image in images)
        stack=np.stack([np.asarray(image.dataobj,dtype=np.float64)[support] for image in images])
        assert all(np.isfinite(values).all() for values in stack)
        assert all(sha(path)==expected for path,expected in zip(subset.path,subset.sha256))
        checkpoint=np.load(folder/'input_effect_stack.npz',allow_pickle=False)
        assert np.array_equal(checkpoint['subjects'].astype(str),np.asarray(subjects)) and np.array_equal(checkpoint['effects'],stack)
        sd=stack.std(0,ddof=1); reconstructed=np.divide(stack.mean(0),sd/np.sqrt(len(subjects)),out=np.zeros_like(sd),where=sd>0)
        manual=np.asarray(nib.load(folder/'manual_t.nii.gz').dataobj,dtype=np.float64)
        permutation=np.asarray(nib.load(folder/'t_finite_float64.nii.gz').dataobj,dtype=np.float64)
        assert np.all(manual[~support]==0) and np.all(permutation[~support]==0), f'{sample} {key}: zero padding outside group mask'
        for output_key in done['output_keys']:
            output=np.asarray(nib.load(folder/f'{output_key}_finite_float64.nii.gz').dataobj,dtype=np.float64)
            assert output.dtype==np.float64 and np.isfinite(output).all() and np.all(output[~support]==0)
            zero_checked += 1
        manual_diff=float(np.max(np.abs(manual[support]-reconstructed)))
        perm_diff=float(np.max(np.abs(permutation[support]-reconstructed)))
        assert manual_diff<1e-12 and perm_diff<1e-7, f'{sample} {key}: t reconstruction mismatch'
        cluster_check=verify_cluster_map(key,manual,support,mask.affine,folder,len(subjects)-1)
        raw.append(done['summary']['minimum_cluster_pFWE_raw'])
        map_checks.append({'contrast_key':key,'manual_t_max_abs_diff':manual_diff,'permutation_t_max_abs_diff':perm_diff,
                           'input_vectors_exact':True,'input_hashes_exact':True,'no_zero_padding':True,**cluster_check})
    ledger=pd.read_csv(base/'contrast_summary_correction_ledgers.csv').sort_values('contrast_key').reset_index(drop=True)
    ordering=sorted(KEYS); raw_by_key=dict(zip(KEYS,raw))
    expected=holm(np.asarray([raw_by_key[key] for key in ordering]))
    got=ledger.set_index('contrast_key').loc[ordering,'holm_all_15_maps'].to_numpy(float)
    assert len(ledger)==15 and np.allclose(got,expected,atol=1e-12,rtol=0), f'{sample}: 15-map Holm ledger differs'
    return {'sample':sample,'maps_verified':len(map_checks),'manifest_rows':len(manifest),'mask_voxels':int(support.sum()),
            'holm_all_15_exact':True,'all_group_outputs_float64_finite_and_zero_outside_mask':zero_checked,'map_checks':map_checks}


def main():
    frozen=json.loads((OUT/'FROZEN_PREFLIGHT.json').read_text())
    assert frozen['status']=='PASS' and frozen['n_permutations']==N_PERM and len(frozen['contrast_plan'])==15
    assert sha(OUT/'run_complete_contrasts.py')==frozen['runner_sha256'], 'runner changed after freeze'
    first=pd.read_csv(OUT/'first_level_manifest.csv')
    reproduction=pd.read_csv(OUT/'pre_reproduction.csv')
    designs=pd.read_csv(OUT/'cache_design_validation.csv')
    assert len(first)==300 and first.subject.nunique()==20 and first.contrast_key.nunique()==15
    assert len(reproduction)==40 and bool(reproduction.passes_exact.all()) and float(reproduction.max_abs_difference_saved_positive_pre.max())<=1e-10
    assert designs.passes_exact.all() and set(N19).issubset(set(designs.subject)) and 'sub-005' in set(designs.subject)
    n19mask= nib.load(SRC/'masks/common_n19_input_mask_intersection.nii.gz')
    n20mask= nib.load(EXT/'masks/new_n20_input_mask_intersection.nii.gz')
    support19=np.asarray(n19mask.dataobj).astype(bool); support20=np.asarray(n20mask.dataobj).astype(bool)
    assert support19.sum()==52787 and support20.sum()==52486 and np.all(~support20|support19)
    n19=verify_sample('N19',N19,SRC/'masks/common_n19_input_mask_intersection.nii.gz')
    n20=verify_sample('N20',N20,EXT/'masks/new_n20_input_mask_intersection.nii.gz')
    result={'status':'PASS','runner_imported':False,'first_level_effect_maps':len(first),'archived_pre_maps_reproduced':len(reproduction),
            'max_pre_map_abs_error':float(reproduction.max_abs_difference_saved_positive_pre.max()),
            'cache_design_subjects_validated':sorted(designs.subject.unique().tolist()),'source_n19_and_sub005_design_validated':True,
            'source_mask_no_zero_padding':True,'n19_mask_voxels':int(support19.sum()),'n20_mask_voxels':int(support20.sum()),
            'group_maps_verified_total':n19['maps_verified']+n20['maps_verified'],'holm_map_count_total':30,'samples':[n19,n20]}
    (OUT/'independent_verification_complete_contrasts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','first_level_effect_maps':len(first),'pre_reproductions':len(reproduction),
                      'group_maps_verified_total':30,'holm_map_count_total':30,'source_mask_no_zero_padding':True},indent=2))

if __name__=='__main__':
    main()
