"""Retrospective one-sided within-map FWE readback of frozen W10 CA-JX interaction.

The positive tail was selected after inspecting a null two-sided analysis;
this is exploratory, not a predeclared directional confirmation.
Requires protected first-level inputs; writes only to a separate scratch directory.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.stats import t as tdist
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
root=Path(os.environ['HAIKU_PROJECT_ROOT']).resolve()/'glm_unified'
public=root.parent/'data_release'
if root == public or public in root.parents:
    raise RuntimeError('Protected analysis inputs must be outside the public checkout')
w1=root/'w1_comprehensive_group_only_twosided_20260918'
mask=w1.parent/'section3_duration_sensitivity_20260908/common_n19_intersection_mask.nii.gz'
assert hashlib.sha256(mask.read_bytes()).hexdigest()=='2c497f2aef874bd586b0346df22322af4564edc0afcbeb73894b26a0013b71e6'
subjects=[1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]
paths=[w1/'participant_contrasts'/f'sub-{s:03d}_interaction_CA_minus_JX_effect.nii.gz' for s in subjects]
assert len(paths)==19 and all(p.is_file() for p in paths)
manifest=pd.read_csv(w1/'participant_contrast_manifest.csv')
for s,p in zip(subjects,paths):
    row=manifest[(manifest.subject==f'sub-{s:03d}')&(manifest.map_key=='interaction_CA_minus_JX')]
    assert len(row)==1 and row.iloc[0]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
design=pd.DataFrame({'intercept':np.ones(len(paths))})
ni=nib.load(mask);mask_data=np.asarray(ni.dataobj,dtype=bool)
assert int(mask_data.sum())==48279
param=SecondLevelModel(mask_img=str(mask),smoothing_fwhm=None).fit([str(p) for p in paths],design_matrix=design)
observed=np.asarray(param.compute_contrast('intercept',output_type='stat').dataobj,dtype=float)
reference=np.asarray(nib.load(w1/'group/interaction_CA_minus_JX_parametric_t.nii.gz').dataobj,dtype=float)
max_err=float(np.max(np.abs(observed[mask_data]-reference[mask_data])))
assert max_err<1e-7,max_err
n_perm=50000;seed=int(os.environ.get('HAIKU_SEED','20266972'));cdt=.001
if seed not in (20266972, 20260918):
    raise ValueError('Only the frozen primary and independent-check seeds are accepted')
result=non_parametric_inference([str(p) for p in paths],design_matrix=design,second_level_contrast='intercept',mask=str(mask),smoothing_fwhm=None,model_intercept=True,n_perm=n_perm,two_sided_test=False,random_state=seed,n_jobs=8,verbose=1,threshold=cdt,tfce=False)
tmap=np.asarray(result['t'].dataobj,dtype=float)
assert np.allclose(tmap[mask_data],observed[mask_data],atol=1e-7,rtol=1e-7)
threshold=float(tdist.isf(cdt,len(paths)-1));labels,n=ndimage.label((tmap>threshold)&mask_data,ndimage.generate_binary_structure(3,1));logp=np.asarray(result['logp_max_size'].dataobj,dtype=float)
clusters=[]
for i in range(1,n+1):
    cm=labels==i;idx=np.argwhere(cm);peak=idx[np.argmax(tmap[cm])];ijk=tuple(int(v) for v in peak)
    clusters.append({'k':int(cm.sum()),'pFWE':float(10**(-np.max(logp[cm]))),'peak_t':float(tmap[ijk]),'peak_mni':[float(x) for x in nib.affines.apply_affine(ni.affine,peak)]})
clusters.sort(key=lambda x:(x['pFWE'],-x['k']))
output={'analysis':'retrospective one-sided positive Pre(CA-JX)-Search(CA-JX) on frozen W10 cell effects','n':len(paths),'n_perm':n_perm,'random_seed':seed,'voxel_cdt_p_one_sided':cdt,'threshold_t':threshold,'mask_voxels':int(mask_data.sum()),'observed_t_max_abs_difference_vs_two_sided_source':max_err,'n_cdt_clusters':n,'n_pFWE_lt_0p05':sum(x['pFWE']<.05 for x in clusters),'top_clusters':clusters[:20]}
path=Path(os.environ['HAIKU_W10_OUTPUT_DIR']).resolve()/f'haiku_w10_one_sided_seed{seed}.json'
if path == public or public in path.parents or path == root or root in path.parents:
    raise RuntimeError('Analysis output must be outside the public checkout and protected source tree')
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(output,indent=2)+'\n');print('FINAL RESULT',json.dumps(output,indent=2))
