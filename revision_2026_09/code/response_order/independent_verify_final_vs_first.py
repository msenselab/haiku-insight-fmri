#!/usr/bin/env python3
"""Standalone verifier for the frozen CSV-limited final-versus-first extension.

This script deliberately does not import the analysis runner.
"""
from __future__ import annotations
import os
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy import ndimage
from scipy.stats import t as t_distribution

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
OUT = ROOT / 'glm_unified/preresponse_with_search_final_vs_first_20260921_r1'
SOURCE = ROOT / 'glm_unified/preresponse_with_search_first_vs_all_buttons_20260903'
DESIGN, FIRST, GROUP, MAN = OUT/'design_gate', OUT/'first_level', OUT/'group_inference', OUT/'manifests'
EVENTS = SOURCE/'design_only/all_buttons'; MASK = SOURCE/'masks/common_n19_input_mask_intersection.nii.gz'
BEHAVIOR = ROOT/'manuscript/behavioral_analysis/behavioral_data_all_subjects.csv'; FMRIPREP = ROOT/'derivatives/fmriprep'
SOURCE_FILES = [ROOT/'glm_unified/scripts/run_preresponse_with_search_first_vs_all_buttons_20260903.py', ROOT/'glm_unified/scripts/run_adapted_preresponse_later_buttons_20260902.py', ROOT/'glm_unified/analysis_plans/PRE_RESPONSE_WITH_SEARCH_FIRST_VS_ALL_BUTTONS_PLAN_20260903.md', SOURCE/'design_only/frozen_design.json', MASK, BEHAVIOR, OUT/'FROZEN_PLAN.md']
SUBJECTS = [f'sub-{i:03d}' for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
CONDS = ['CA','JX','OI']; COND = {'context-action':'CA','juxtaposition':'JX','One-image':'OI'}
BUTTONS = ['first_insight_rt','insight2','insight3','insight4']; MOTION = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
FOCAL = {'pre_final_gt_first': 'pre_final_vs_first', 'search_final_gt_first': 'search_final_vs_first', 'pre_first_gt_final': 'pre_final_vs_first', 'search_first_gt_final': 'search_final_vs_first'}
FAMILY = {'pre_final_gt_first':'primary_final_gt_first','search_final_gt_first':'primary_final_gt_first','pre_first_gt_final':'secondary_first_gt_final','search_first_gt_final':'secondary_first_gt_final'}
TR, HP, CDT, VIF = 1., 1/128., .001, 10.

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def equal_df(a: pd.DataFrame,b: pd.DataFrame) -> bool:
    if a.columns.tolist()!=b.columns.tolist() or len(a)!=len(b): return False
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]):
            if not np.allclose(a[col].to_numpy(float),b[col].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True): return False
        elif not a[col].astype(str).equals(b[col].astype(str)): return False
    return True

def trial_end(onset: float, nxt: float, frame: float) -> float:
    return min([onset+30.,frame]+([nxt] if np.isfinite(nxt) else []))

def rebuild_source(subject: str) -> pd.DataFrame:
    d=pd.read_csv(BEHAVIOR); d=d[d.subject_id.eq(subject)].sort_values('onset').reset_index(drop=True)
    if len(d)!=42: raise RuntimeError(f'{subject}: trial count {len(d)}')
    bold=FMRIPREP/subject/'func'/f'{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'; frame=nib.load(bold).shape[-1]*TR
    rows=[]
    for idx,row in d.iterrows():
        onset=float(row.onset); end=trial_end(onset,float(d.iloc[idx+1].onset) if idx+1<len(d) else np.inf,frame); condition=COND[str(row.haiku_type)]
        buttons=[]
        for name in BUTTONS:
            if name in row.index and pd.notna(row[name]):
                rt=float(row[name])
                if not np.isfinite(rt) or rt<.5 or rt>end-onset+1e-9: raise RuntimeError(f'{subject}: invalid {name}')
                buttons.append((name,rt))
        if any(buttons[i][1]>=buttons[i+1][1] for i in range(len(buttons)-1)): raise RuntimeError(f'{subject}: unsorted buttons')
        if not buttons:
            rows.append(dict(onset=onset,duration=end-onset,trial_type='no_response',modulation=1.,trial_index=idx+1,button_order=0,button_time=np.nan,trial_end=end)); continue
        if buttons[0][0] != 'first_insight_rt': raise RuntimeError(f'{subject}: later button without first')
        previous=onset
        for order,(_,rt) in enumerate(buttons,1):
            button=onset+rt; pre=button-.5; duration=max(0.,pre-previous)
            if duration>1e-9: rows.append(dict(onset=previous,duration=duration,trial_type=f'search_{condition}',modulation=1.,trial_index=idx+1,button_order=order,button_time=button,trial_end=end))
            rows.append(dict(onset=pre,duration=.5,trial_type=f'pre_{condition}',modulation=1.,trial_index=idx+1,button_order=order,button_time=button,trial_end=end)); previous=button
        if end>previous+1e-9: rows.append(dict(onset=previous,duration=end-previous,trial_type='post_response',modulation=1.,trial_index=idx+1,button_order=len(buttons),button_time=previous,trial_end=end))
    return pd.DataFrame(rows).sort_values(['onset','trial_type','trial_index']).reset_index(drop=True)

def extend(source: pd.DataFrame,subject: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    pre_rows=[]; search_rows=[]; coverage=[]
    for trial,g in source[source.trial_type.str.startswith('pre_')].groupby('trial_index',sort=True):
        g=g.sort_values('button_order'); n=len(g); condition=str(g.iloc[0].trial_type).split('_',1)[1]
        if n<2: continue
        first,final=g.iloc[0].copy(),g.iloc[-1].copy()
        for row,mod in ((first,-.5),(final,.5)):
            row['trial_type']='pre_final_vs_first'; row['modulation']=mod; pre_rows.append(row)
        sg=source[source.trial_index.eq(trial)&source.trial_type.str.startswith('search_')&(source.duration>1e-9)].sort_values('button_order')
        fst=sg[sg.button_order.eq(int(first.button_order))]; fin=sg[sg.button_order.eq(int(final.button_order))]; paired=len(fst)==1 and len(fin)==1
        if paired:
            for row,mod in ((fst.iloc[0].copy(),-.5),(fin.iloc[0].copy(),.5)):
                row['trial_type']='search_final_vs_first'; row['modulation']=mod; search_rows.append(row)
        coverage.append(dict(subject=subject,trial_index=int(trial),condition=condition,n_response_buttons=n,pre_paired=True,search_paired=paired,first_search_positive=len(fst)==1,final_search_positive=len(fin)==1))
    extra=pd.concat([pd.DataFrame(pre_rows,columns=source.columns),pd.DataFrame(search_rows,columns=source.columns)],ignore_index=True)
    events=pd.concat([source,extra],ignore_index=True).sort_values(['onset','trial_type','trial_index','button_order']).reset_index(drop=True)
    return events,pd.DataFrame(coverage)

def vif(a: np.ndarray,j: int) -> float:
    y=a[:,j]; q=np.delete(a,j,axis=1); den=float(np.sum((y-y.mean())**2)); resid=y-q@np.linalg.lstsq(q,y,rcond=None)[0]
    return den/max(float(resid@resid),np.finfo(float).eps) if den>np.finfo(float).eps else float('inf')

def build_design(subject: str,events: pd.DataFrame) -> pd.DataFrame:
    d=FMRIPREP/subject/'func'; bold=d/f'{subject}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'; conf=d/f'{subject}_task-haiku_desc-confounds_timeseries.tsv'
    n=nib.load(bold).shape[-1]; motion=pd.read_csv(conf,sep='\t')[MOTION].replace([np.inf,-np.inf],np.nan).fillna(0.)
    return make_first_level_design_matrix(np.arange(n)*TR,events=events[['onset','duration','trial_type','modulation']],hrf_model='spm',drift_model='cosine',high_pass=HP,add_regs=motion.to_numpy(float),add_reg_names=MOTION,min_onset=-24,oversampling=50)

def holm(values: np.ndarray) -> np.ndarray:
    v=np.asarray(values,float); order=np.argsort(v); ordered=np.maximum.accumulate((len(v)-np.arange(len(v)))*v[order]); out=np.empty_like(v); out[order]=np.minimum(ordered,1.); return out

def image(path: Path,mask: np.ndarray) -> tuple[nib.Nifti1Image,np.ndarray]:
    im=nib.load(path); data=np.asarray(im.dataobj)
    if im.get_data_dtype()!=np.dtype('float64') or not np.isfinite(data[mask]).all() or np.unique(data[mask]).size<=256: raise RuntimeError(f'continuous float64 failure: {path}')
    return im,data

def verify_group(directory: Path,subjects: list[str],prefix: str,checks: dict[str,bool]) -> None:
    mask_img=nib.load(MASK); mask=np.asarray(mask_img.dataobj).astype(bool); threshold=float(t_distribution.isf(CDT,len(subjects)-1))
    summary=pd.read_csv(directory/'contrast_summary_holm.csv'); clusters=pd.read_csv(directory/'all_cdt_clusters_labeled.csv')
    required={'contrast_key','contrast','cluster_id','mni_x','mni_y','mni_z','peak_t','peak_z','cluster_voxels','cluster_pFWE_raw','harvard_oxford_cortical','harvard_oxford_subcortical','qualified_anatomy'}
    checks[f'{prefix}_cluster_schema']=required.issubset(clusters.columns)
    t_ok=perm_ok=geometry_ok=True
    for key in FOCAL:
        effects=[]
        for subject in subjects:
            _,dat=image(FIRST/subject/f'{subject}_{key}_effect.nii.gz',mask); effects.append(dat)
        stack=np.stack(effects); mean=stack.mean(0); sd=stack.std(0,ddof=1); manual=np.divide(mean,sd/np.sqrt(len(subjects)),out=np.zeros_like(mean),where=sd>0)
        _,saved_manual=image(directory/f'{key}_manual_t.nii.gz',mask); _,parametric=image(directory/f'{key}_parametric_t.nii.gz',mask); _,permuted=image(directory/f'{key}_t.nii.gz',mask)
        t_ok &= np.max(np.abs(manual[mask]-saved_manual[mask]))<=1e-10 and np.max(np.abs(manual[mask]-parametric[mask]))<=1e-10
        perm_ok &= np.allclose(parametric[mask],permuted[mask],rtol=1e-7,atol=1e-7)
        labels,nc=ndimage.label(parametric>threshold,ndimage.generate_binary_structure(3,1)); observed=[]
        for cid in range(1,nc+1):
            vox=np.argwhere(labels==cid); peak=vox[np.argmax(parametric[labels==cid])]; xyz=nib.affines.apply_affine(mask_img.affine,peak); observed.append((int(len(vox)),round(float(xyz[0]),5),round(float(xyz[1]),5),round(float(xyz[2]),5),round(float(parametric[tuple(peak)]),5)))
        saved=[]
        for row in clusters.loc[clusters.contrast_key.eq(key)].itertuples(index=False): saved.append((int(row.cluster_voxels),round(float(row.mni_x),5),round(float(row.mni_y),5),round(float(row.mni_z),5),round(float(row.peak_t),5)))
        geometry_ok &= sorted(observed)==sorted(saved)
    checks[f'{prefix}_manual_group_t']=t_ok; checks[f'{prefix}_permutation_t_readback']=perm_ok; checks[f'{prefix}_six_neighbor_cluster_geometry']=geometry_ok
    summary=summary.sort_values('contrast_key').reset_index(drop=True)
    h_ok=True
    for fam,g in summary.groupby('family'):
        expected=holm(g.minimum_cluster_pFWE_raw.to_numpy(float)); h_ok &= np.allclose(expected,g.minimum_cluster_pFWE_holm_family_2maps.to_numpy(float),atol=1e-15,rtol=0)
    checks[f'{prefix}_holm2']=h_ok

def main() -> None:
    checks: dict[str,bool]={}; failures=[]
    def ck(name: str,value: bool) -> None: checks[name]=bool(value); failures.append(name) if not value else None
    manifest=json.loads((MAN/'source_input_hash_manifest.json').read_text())
    ck('source_hash_manifest',all(manifest['source'].get(str(path.relative_to(ROOT)))==sha(path) for path in SOURCE_FILES))
    mask_img=nib.load(MASK); ck('common_mask_52787',int(np.asarray(mask_img.dataobj).astype(bool).sum())==52787)
    frozen=json.loads((SOURCE/'design_only/frozen_design.json').read_text())['model_results']['all_buttons']
    ck('frozen_source_counts',all(frozen[k]==v for k,v in {'total_target_events':1670,'total_first_target_events':779,'total_later_target_events':891,'total_search_segments':1663,'total_overlapping_target_window_pairs':7}.items()))
    rows=[]; exact_source=exact_successor=exact_design=True
    for subject in SUBJECTS:
        source=rebuild_source(subject); immutable=pd.read_csv(EVENTS/f'{subject}_events.tsv',sep='\t'); exact_source &= equal_df(source,immutable)
        successor,coverage=extend(immutable,subject); saved=pd.read_csv(DESIGN/f'{subject}_events.tsv',sep='\t'); exact_successor &= equal_df(successor,saved)
        saved_cov=pd.read_csv(DESIGN/f'{subject}_paired_trial_coverage.csv'); exact_successor &= equal_df(coverage,saved_cov)
        design=build_design(subject,successor); saved_x=pd.read_csv(DESIGN/f'{subject}_design_matrix.csv'); exact_design &= design.columns.tolist()==saved_x.columns.tolist() and np.max(np.abs(design.to_numpy(float)-saved_x.to_numpy(float)))<=1e-12
        a=design.to_numpy(float); cols=design.columns.tolist(); required=[f'{phase}_{condition}' for phase in ['pre','search'] for condition in CONDS]+['pre_final_vs_first','search_final_vs_first','post_response']+MOTION
        pinv=np.linalg.pinv(a.T@a); detail={}
        for name in ['pre_final_vs_first','search_final_vs_first']:
            con=np.zeros(len(cols)); con[cols.index(name)]=1.; residual=np.linalg.norm(con-a.T@np.linalg.pinv(a.T)@con); detail[name]=(vif(a,cols.index(name)),residual,float(con@pinv@con))
        passed=np.isfinite(a).all() and all(name in cols for name in required) and np.linalg.matrix_rank(a)==len(cols) and all(v<=VIF and r<=1e-10 and np.isfinite(p) and p>0 for v,r,p in detail.values())
        rows.append(dict(subject=subject,source_rows=len(source),successor_rows=len(successor),multiresponse_trials=len(coverage),paired_search_trials=int(coverage.search_paired.sum()),two_response_trials=int((coverage.n_response_buttons==2).sum()),three_response_trials=int((coverage.n_response_buttons==3).sum()),vif_pre_final_vs_first=detail['pre_final_vs_first'][0],vif_search_final_vs_first=detail['search_final_vs_first'][0],rowspace_pre_final_vs_first=detail['pre_final_vs_first'][1],rowspace_search_final_vs_first=detail['search_final_vs_first'][1],precision_pre_final_vs_first=detail['pre_final_vs_first'][2],precision_search_final_vs_first=detail['search_final_vs_first'][2],passes_gate=passed))
    q=pd.DataFrame(rows); declared=json.loads((DESIGN/'gate_summary.json').read_text()); savedq=pd.read_csv(DESIGN/'participant_gate_qc.csv')
    ck('raw_csv_reconstruction_equals_immutable_events',exact_source); ck('contrast_coded_successor_equals_saved_events',exact_successor); ck('independent_designs_equal_saved',exact_design)
    ck('gate_reproduced',bool(q.passes_gate.all())==bool(declared['all_pass']) and int(q.passes_gate.sum())==int(declared['n_pass']))
    fields=['vif_pre_final_vs_first','vif_search_final_vs_first','rowspace_pre_final_vs_first','rowspace_search_final_vs_first','precision_pre_final_vs_first','precision_search_final_vs_first']
    ck('gate_diagnostics_equal_saved',np.allclose(q[fields],savedq[fields],atol=1e-12,rtol=0))
    cov=pd.read_csv(DESIGN/'paired_trial_coverage_all.csv'); ck('declared_coverage_counts',len(cov)==599 and int((cov.n_response_buttons==2).sum())==307 and int((cov.n_response_buttons==3).sum())==292 and cov.condition.value_counts().to_dict()=={'CA':232,'JX':216,'OI':151} and int(cov.search_paired.sum())==597 and int(cov.loc[cov.subject.eq('sub-004')].shape[0])==1)
    if not declared['all_pass']:
        ck('stopped_has_no_bold_or_group_files',not any(FIRST.rglob('*.nii*')) and not any(GROUP.rglob('*.nii*')))
    else:
        mask=np.asarray(mask_img.dataobj).astype(bool); first_ok=True; algebra=True
        for subject in SUBJECTS:
            prov=json.loads((FIRST/subject/f'{subject}_provenance.json').read_text())
            for key in FOCAL:
                for suffix in ['effect','variance','t','z']:
                    path=FIRST/subject/f'{subject}_{key}_{suffix}.nii.gz'; im,dat=image(path,mask); first_ok &= sha(path)==prov['outputs'][f'{key}_{suffix}']
            for pos,neg in [('pre_final_gt_first','pre_first_gt_final'),('search_final_gt_first','search_first_gt_final')]:
                _,pdat=image(FIRST/subject/f'{subject}_{pos}_effect.nii.gz',mask); _,ndat=image(FIRST/subject/f'{subject}_{neg}_effect.nii.gz',mask); algebra &= np.allclose(pdat[mask],-ndat[mask],atol=1e-12,rtol=0)
        ck('first_level_hashes_float64_and_continuous',first_ok); ck('first_level_reverse_contrast_algebra',algebra)
        verify_group(GROUP,SUBJECTS,'n19',checks)
        n18=GROUP/'n18_excluding_sub-004'
        if n18.exists() and (n18/'contrast_summary_holm.csv').exists(): verify_group(n18,[s for s in SUBJECTS if s!='sub-004'],'n18_coverage_sensitivity',checks)
        else: ck('n18_coverage_sensitivity_present',False)
    artifact_files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hash_manifest.json']
    (MAN/'artifact_hash_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in sorted(artifact_files)},indent=2,sort_keys=True)+'\n')
    checks.update({'n_checks':len(checks),'status':'PASS' if not failures else 'FAIL','failures':failures,'verifier_imports_runner':False,'independent_n_pass':int(q.passes_gate.sum()),'max_vif_pre_final_vs_first':float(q.vif_pre_final_vs_first.max()),'max_vif_search_final_vs_first':float(q.vif_search_final_vs_first.max()),'max_rowspace_residual':float(q[['rowspace_pre_final_vs_first','rowspace_search_final_vs_first']].to_numpy().max()),'min_precision':float(q[['precision_pre_final_vs_first','precision_search_final_vs_first']].to_numpy().min())})
    (OUT/'independent_design_gate_reconstruction.csv').write_text(q.to_csv(index=False)); (OUT/'independent_verification.json').write_text(json.dumps(checks,indent=2,sort_keys=True,default=lambda value: value.item() if isinstance(value,np.generic) else str(value))+'\n')
    print(json.dumps(checks,indent=2,sort_keys=True,default=lambda value: value.item() if isinstance(value,np.generic) else str(value))); raise SystemExit(0 if not failures else 1)
if __name__=='__main__': main()
