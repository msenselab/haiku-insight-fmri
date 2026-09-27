#!/usr/bin/env python3
"""Frozen W1 group-only nine-map, two-sided cluster-FWE suite; no first-level fit."""
from __future__ import annotations
import argparse, hashlib, json, os, platform
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import nibabel as nib
import nilearn, numpy as np, pandas as pd, scipy
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from scipy import ndimage
from scipy.stats import t as tdist

ROOT=Path(os.environ['HAIKU_PROJECT_ROOT']).expanduser().resolve()
OUT=ROOT/'glm_unified/w1_comprehensive_group_only_twosided_20260918'
if Path(__file__).resolve().parents[3] in OUT.parents:
    raise RuntimeError('Analysis output must be outside the public checkout')
PLAN=OUT/'PLAN.md'
SOURCE=ROOT/'glm_unified/section3_duration_sensitivity_20260908'
MASK=SOURCE/'common_n19_intersection_mask.nii.gz'
SUBJECTS=[f'sub-{x:03d}' for x in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
CELLS=[f'{phase}_{condition}' for phase in ('search','pre') for condition in ('CA','JX','OI')]
PAIRS=[('CA','JX'),('CA','OI'),('JX','OI')]
CONTRASTS=[]
for family, phase in [('Search','search'),('Pre','pre')]:
    for a,b in PAIRS:
        key=f'{phase}_{a}_minus_{b}'
        CONTRASTS.append(dict(key=key,label=f'{family} {a}−{b}',family=family,weights={f'{phase}_{a}':1.,f'{phase}_{b}':-1.}))
for a,b in PAIRS:
    key=f'interaction_{a}_minus_{b}'
    CONTRASTS.append(dict(key=key,label=f'[Pre({a}−{b})]−[Search({a}−{b})]',family='Interaction',weights={f'pre_{a}':1.,f'pre_{b}':-1.,f'search_{a}':-1.,f'search_{b}':1.}))
N_PERM=50000; CDT=.001; SEED=20260918; EXPECTED_MASK_HASH='2c497f2aef874bd586b0346df22322af4564edc0afcbeb73894b26a0013b71e6'
CLUSTER_COLUMNS=['map_key','map_label','family','cluster_sign','direction','cluster_id_within_sign','mni_x','mni_y','mni_z','peak_t','peak_z','cluster_voxels','cluster_mm3','cluster_pFWE_raw','survives_within_map_cluster_FWE_05']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(data,aff,path):
    x=np.asarray(data,dtype=np.float64); assert np.isfinite(x).all()
    nib.Nifti1Image(x,aff).to_filename(path)
    assert nib.load(path).get_data_dtype()==np.dtype('float64')
def holm(v):
    v=np.asarray(v,float); order=np.argsort(v); ranked=v[order]
    corrected=np.minimum(1.,np.maximum.accumulate((len(v)-np.arange(len(v)))*ranked))
    out=np.empty_like(corrected); out[order]=corrected; return out
def cellpath(subject,cell): return SOURCE/'first_level'/subject/f'w10_{cell}_effect.nii.gz'
def require_freeze():
    f=json.loads((OUT/'frozen_run.json').read_text())
    if sha(PLAN)!=f['plan_sha256'] or sha(__file__)!=f['runner_sha256']: raise RuntimeError('Frozen plan/runner changed')
    return f

def prepare():
    if (OUT/'frozen_run.json').exists() or (OUT/'group').exists() or (OUT/'participant_contrasts').exists():
        raise RuntimeError('Refusing to overwrite/reprepare frozen run')
    if not PLAN.exists(): raise RuntimeError('Missing frozen PLAN.md')
    mask_img=nib.load(MASK); mask=np.asarray(mask_img.dataobj).astype(bool)
    if sha(MASK)!=EXPECTED_MASK_HASH or int(mask.sum())!=48279: raise RuntimeError('Common N19 mask identity failed')
    outmaps=OUT/'participant_contrasts';outmaps.mkdir()
    source_rows=[];direct_rows=[];ref=None
    for subject in SUBJECTS:
        provenance=json.loads((SOURCE/'first_level'/subject/'provenance.json').read_text())
        if provenance['subject']!=subject: raise RuntimeError('Subject provenance mismatch')
        output_hashes=provenance['output_sha256']
        cells={}
        for cell in CELLS:
            p=cellpath(subject,cell); rel=str(p.relative_to(SOURCE))
            actual=sha(p); expected=output_hashes.get(rel)
            if actual!=expected: raise RuntimeError(f'Source hash mismatch {subject} {cell}')
            img=nib.load(p); data=np.asarray(img.dataobj,dtype=float)
            if ref is None: ref=img
            if data.shape!=mask_img.shape or not np.allclose(img.affine,mask_img.affine): raise RuntimeError('Cell grid/mask mismatch')
            if not np.isfinite(data).all() or not np.all(data[~mask]==0): raise RuntimeError('Invalid source map')
            cells[cell]=data
            source_rows.append(dict(subject=subject,cell=cell,path=str(p),sha256=actual,first_level_provenance_path=str(SOURCE/'first_level'/subject/'provenance.json'),provenance_sha256=sha(SOURCE/'first_level'/subject/'provenance.json'),provenance_expected_sha256=expected,hash_matches_saved_provenance=True))
        for con in CONTRASTS:
            data=sum(cells[c]*float(w) for c,w in con['weights'].items())
            p=outmaps/f'{subject}_{con["key"]}_effect.nii.gz';save(data,mask_img.affine,p)
            direct_rows.append(dict(subject=subject,map_key=con['key'],path=str(p),sha256=sha(p),max_abs_cell_algebra_error=0.0,weights=json.dumps(con['weights'],sort_keys=True)))
    pd.DataFrame(source_rows).to_csv(OUT/'source_cell_hash_manifest.csv',index=False)
    pd.DataFrame(direct_rows).to_csv(OUT/'participant_contrast_manifest.csv',index=False)
    frozen=dict(status='FROZEN_BEFORE_NEW_GROUP_MAP_INSPECTION',plan_sha256=sha(PLAN),runner_sha256=sha(__file__),source_directory=str(SOURCE),source_mask=str(MASK),source_mask_sha256=sha(MASK),source_mask_voxels=int(mask.sum()),subjects=SUBJECTS,cells=CELLS,contrasts=CONTRASTS,n_perm=N_PERM,cdt_two_sided_p=CDT,df=18,connectivity=6,within_map='two-sided signed sign-flips; maximum cluster extent over positive and negative signs',family_correction='Holm across 3 map minima per Search/Pre/Interaction family',global_sensitivity='Holm across 9 map minima',seed_base=SEED,software=dict(python=platform.python_version(),nilearn=nilearn.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,nibabel=nib.__version__))
    dump(OUT/'frozen_run.json',frozen)
    print(json.dumps({'status':frozen['status'],'n_subjects':19,'n_maps':9,'mask_voxels':int(mask.sum()),'n_source_cells':len(source_rows),'n_perm':N_PERM},indent=2))

def clusters(con,t,z,size,logp,mask_img,threshold):
    m=np.asarray(mask_img.dataobj).astype(bool);struct=ndimage.generate_binary_structure(3,1);volume=abs(np.linalg.det(mask_img.affine[:3,:3]));rows=[]
    for sign,selected,chooser,direction in [('positive',t>threshold,np.argmax,'stated positive contrast direction'),('negative',t<-threshold,np.argmin,'opposite of stated positive contrast direction')]:
        labels,n=ndimage.label(selected&m,struct)
        for cid in range(1,n+1):
            cm=labels==cid; indices=np.argwhere(cm); peak=indices[int(chooser(t[cm]))]; xyz=nib.affines.apply_affine(mask_img.affine,peak);extent=int(cm.sum())
            reported=np.asarray(size)[cm]; reported=reported[reported>0]
            if len(reported) and int(round(reported.max()))!=extent: raise RuntimeError('Cluster extent map mismatch')
            p=float(10**(-np.max(logp[cm])))
            rows.append(dict(map_key=con['key'],map_label=con['label'],family=con['family'],cluster_sign=sign,direction=direction,cluster_id_within_sign=cid,mni_x=float(xyz[0]),mni_y=float(xyz[1]),mni_z=float(xyz[2]),peak_t=float(t[tuple(peak)]),peak_z=float(z[tuple(peak)]),cluster_voxels=extent,cluster_mm3=extent*volume,cluster_pFWE_raw=p,survives_within_map_cluster_FWE_05=bool(p<.05)))
    return pd.DataFrame(rows,columns=CLUSTER_COLUMNS)

def group(n_jobs):
    frozen=require_freeze(); mask_img=nib.load(MASK); mask=np.asarray(mask_img.dataobj).astype(bool);out=OUT/'group';out.mkdir(exist_ok=False)
    source_manifest=pd.read_csv(OUT/'source_cell_hash_manifest.csv');direct_manifest=pd.read_csv(OUT/'participant_contrast_manifest.csv')
    for r in source_manifest.itertuples():
        if sha(r.path)!=r.sha256 or not r.hash_matches_saved_provenance: raise RuntimeError('Frozen source input changed')
    allrows=[]; summaries=[]; manual_rows=[]
    des=pd.DataFrame({'intercept':np.ones(19)}); threshold=float(tdist.isf(CDT/2,18))
    for i,con in enumerate(CONTRASTS):
        key=con['key'];paths=[]
        for s in SUBJECTS:
            row=direct_manifest[(direct_manifest.subject==s)&(direct_manifest.map_key==key)]
            if len(row)!=1 or sha(row.iloc[0].path)!=row.iloc[0].sha256: raise RuntimeError('Participant contrast changed')
            paths.append(row.iloc[0].path)
        sec=SecondLevelModel(mask_img=str(MASK),smoothing_fwhm=None).fit(paths,design_matrix=des)
        t=np.asarray(sec.compute_contrast('intercept',output_type='stat').dataobj,dtype=float);z=np.asarray(sec.compute_contrast('intercept',output_type='z_score').dataobj,dtype=float)
        stack=np.stack([np.asarray(nib.load(p).dataobj,dtype=float) for p in paths]);mean=stack.mean(axis=0);sd=stack.std(axis=0,ddof=1);manual=np.divide(mean,sd/np.sqrt(19),out=np.zeros_like(mean),where=sd>0)
        err=float(np.max(np.abs(t[mask]-manual[mask]))); save(t,mask_img.affine,out/f'{key}_parametric_t.nii.gz');save(z,mask_img.affine,out/f'{key}_parametric_z.nii.gz');save(manual,mask_img.affine,out/f'{key}_manual_t.nii.gz')
        seed=SEED+i*1009
        print(f'GROUP {i+1}/9 {key}: 50000 two-sided sign-flips',flush=True)
        result=non_parametric_inference(paths,design_matrix=des,second_level_contrast='intercept',mask=str(MASK),smoothing_fwhm=None,model_intercept=True,n_perm=N_PERM,two_sided_test=True,random_state=seed,n_jobs=n_jobs,verbose=1,threshold=CDT,tfce=False)
        for name,img in result.items(): save(np.asarray(img.dataobj),img.affine,out/f'{key}_{name}.nii.gz')
        if not np.allclose(t[mask],np.asarray(result['t'].dataobj)[mask],atol=1e-7,rtol=1e-7): raise RuntimeError('Observed t mismatch')
        tab=clusters(con,t,z,np.asarray(result['size'].dataobj),np.asarray(result['logp_max_size'].dataobj),mask_img,threshold); allrows.append(tab)
        summaries.append(dict(map_key=key,map_label=con['label'],family=con['family'],n_cdt_clusters=len(tab),n_positive_cdt_clusters=int((tab.cluster_sign=='positive').sum()),n_negative_cdt_clusters=int((tab.cluster_sign=='negative').sum()),n_within_map_significant_clusters=int(tab.survives_within_map_cluster_FWE_05.sum()),minimum_cluster_pFWE_raw=float(tab.cluster_pFWE_raw.min()) if len(tab) else 1.0,within_map_null=not bool(tab.survives_within_map_cluster_FWE_05.any()),random_seed=seed,manual_group_t_max_abs_diff=err))
        manual_rows.append(dict(map_key=key,manual_group_t_max_abs_diff=err,passes_le_1e_10=err<=1e-10))
    summary=pd.DataFrame(summaries)
    summary['minimum_cluster_pFWE_holm_family3']=np.nan
    for family in ('Search','Pre','Interaction'):
        ix=summary.family.eq(family); summary.loc[ix,'minimum_cluster_pFWE_holm_family3']=holm(summary.loc[ix,'minimum_cluster_pFWE_raw'])
    summary['map_survives_holm_family3_05']=summary.minimum_cluster_pFWE_holm_family3<.05
    summary['minimum_cluster_pFWE_holm_global9_sensitivity']=holm(summary.minimum_cluster_pFWE_raw)
    summary['map_survives_holm_global9_sensitivity_05']=summary.minimum_cluster_pFWE_holm_global9_sensitivity<.05
    cluster_table=pd.concat(allrows,ignore_index=True) if allrows else pd.DataFrame(columns=CLUSTER_COLUMNS)
    cluster_table=cluster_table.merge(summary[['map_key','minimum_cluster_pFWE_holm_family3','map_survives_holm_family3_05','minimum_cluster_pFWE_holm_global9_sensitivity','map_survives_holm_global9_sensitivity_05']],on='map_key',how='left')
    summary.to_csv(out/'group_summary.csv',index=False);cluster_table.to_csv(out/'all_cdt_clusters.csv',index=False);cluster_table[cluster_table.survives_within_map_cluster_FWE_05].to_csv(out/'significant_clusters_within_map.csv',index=False);pd.DataFrame(manual_rows).to_csv(out/'manual_t_qc.csv',index=False)
    meta=dict(n_subjects=19,n_maps=9,n_perm=N_PERM,two_sided=True,cdt_p=CDT,cdt_abs_t=threshold,df=18,connectivity=6,cluster_fwe='maximum cluster extent over both signs; within map',seed_base=SEED,runner_sha256=sha(__file__),frozen_run_sha256=sha(OUT/'frozen_run.json'),software=frozen['software'])
    dump(out/'group_metadata.json',meta)
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    dump(out/'result_hashes.json',hashes)
    print(summary.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','group','all']);p.add_argument('--n-jobs',type=int,default=8);a=p.parse_args()
    if not 1<=a.n_jobs<=8: raise SystemExit('--n-jobs must be 1..8')
    if a.stage in ('prepare','all'): prepare()
    if a.stage in ('group','all'): group(a.n_jobs)
if __name__=='__main__': main()
