#!/usr/bin/env python3
"""Frozen, timing-corrected matched Search/Pre sensitivity. No implicit fitting.
prepare/verify read BOLD headers ONLY; fit-subject and group require explicit stages.
Raw parser adapted from validated rawlog_phases and phase_extension runners.
"""
from pathlib import Path
import os
ROOT=Path(os.environ["HAIKU_PROJECT_ROOT"]).resolve()
OUT=ROOT/'glm_unified/section3_duration_sensitivity_20260908'
if Path(__file__).resolve().parents[3] in OUT.parents:
    raise RuntimeError('Analysis output must be outside the public checkout')
if not (OUT/'FROZEN_PLAN.md').is_file():
    raise FileNotFoundError('Protected source analysis/FROZEN_PLAN.md is required before writing outputs')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
for key, val in {'TMPDIR':OUT/'tmp','JOBLIB_TEMP_FOLDER':OUT/'tmp','XDG_CACHE_HOME':OUT/'cache','NILEARN_DATA':OUT/'cache/nilearn'}.items():
    os.environ[key]=str(val);val.mkdir(parents=True,exist_ok=True)
import argparse, hashlib, json, platform, shutil, sys, warnings
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage, stats
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
BASE=ROOT/'glm_unified'
CANON=BASE/'insight_event_timing_sensitivity/joint_search_pre_phase_contrasts_20260831'
BEHAVIOR=ROOT/'manuscript/behavioral_analysis/behavioral_data_all_subjects.csv'
MASK=OUT/'common_n19_intersection_mask.nii.gz'
SUBJECTS=[f'sub-{i:03d}' for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
CONDITIONS=['CA','JX','OI']
COND={'context-action':'CA','juxtaposition':'JX','One-image':'OI'}
MOTION=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
FOCAL=[f'{p}_{c}' for p in ['search','pre'] for c in CONDITIONS]
WIDTHS={'w05':.5,'w10':1.}
CONTRASTS={'pre_CA_gt_JX':{'pre_CA':1.,'pre_JX':-1.},'pre_CA_gt_OI':{'pre_CA':1.,'pre_OI':-1.}}
MASK_HASH='2c497f2aef874bd586b0346df22322af4564edc0afcbeb73894b26a0013b71e6'
SOURCES=[BASE/'scripts/run_joint_search_pre_phase_contrasts_20260831.py',CANON/'metadata.json',BASE/'scripts/run_ppi_precuneus_amygdala_ofc_rawlog_phases.py',BASE/'connectivity_analysis/ppi_precuneus_amygdala_ofc_phase_extension/code/run_ppi_precuneus_amygdala_ofc_phases.py']
def require(ok,message):
    if not ok: raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def versions():
    return dict(python=platform.python_version(),executable=sys.executable,nilearn=nilearn.__version__,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,nibabel=nib.__version__)
def paths(s):
    d=ROOT/'derivatives/fmriprep'/s/'func'
    bold=d/f'{s}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    return bold,bold.with_suffix('').with_suffix('.json'),d/f'{s}_task-haiku_desc-confounds_timeseries.tsv'
def raw_buttons(path):
    rows=[];trial=None;onset=None;timeout=None;buttons=[];payload=[];start=None
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if not line.startswith('MSG'):continue
            x=line.split();ts=int(x[1]);msg=x[2]
            if msg=='TRIALID':
                require(trial is None,'Unclosed trial');trial=int(x[3]);start=ts;onset=timeout=None;buttons=[];payload=[]
            elif msg=='image_onset' and trial is not None:
                require(onset is None,'Duplicate image onset');onset=ts
            elif msg=='INSIGHT' and trial is not None:
                require(onset is not None and timeout is None,'Button outside presentation');buttons.append(ts);payload.append(float(x[3]))
            elif msg=='time_out' and trial is not None:
                require(onset is not None and timeout is None,'Invalid timeout');timeout=ts
            elif msg.startswith('Scan_end_Run_') and trial is not None:
                require(timeout is not None and 0<=ts-timeout<=2,'Missing/unaligned timeout')
                times=[(b-onset)/1000. for b in buttons];duration=(timeout-onset)/1000.
                require(duration>0 and all(0<t<=duration for t in times) and all(np.diff(times)>0),'Invalid button times')
                rows.append(dict(TrialID=trial,raw_image_onset_s=onset/1000.,raw_trial_start_s=start/1000.,raw_duration_s=duration,raw_buttons_s_json=json.dumps(times),raw_payload_s_json=json.dumps(payload)))
                trial=None
    require(trial is None and len(rows)==42,'Expected exactly 42 complete raw trials')
    return pd.DataFrame(rows)
def trials_for(s,behavior,n):
    asc=list((ROOT/s/'beh').glob('*.asc'));require(len(asc)==1,'Ambiguous raw log')
    t=behavior[behavior.subject_id.eq(s)].merge(raw_buttons(asc[0]),on='TrialID',validate='one_to_one').sort_values('onset').reset_index(drop=True)
    require(len(t)==42,'Incomplete trials')
    np.testing.assert_allclose(t.image_onset,t.raw_image_onset_s,atol=1e-6,rtol=0)
    require(np.all(t.onset>=0) and np.all(t.onset+t.raw_duration_s<=n),'Scan coverage')
    require(np.all((t.onset+t.raw_duration_s).to_numpy()[:-1]<=t.onset.to_numpy()[1:]),'Presentation overlap')
    t['condition']=t.haiku_type.map(COND);require(t.condition.notna().all(),'Unknown condition')
    rts=[]
    for r in t.itertuples():
        raw=json.loads(r.raw_buttons_s_json);payload=json.loads(r.raw_payload_s_json)
        require(bool(raw)==bool(r.has_insight),'Raw/CSV response disagreement')
        if raw:np.testing.assert_allclose(float(r.first_insight_rt),payload[0],atol=.00050001,rtol=0)
        rts.append(raw[0] if raw else np.nan)
    t['image_relative_first_rt_s']=rts
    t['target']=t.image_relative_first_rt_s.gt(1.)
    t['status']=np.where(t.target,'target',np.where(t.image_relative_first_rt_s.notna(),'fast_response','no_insight'))
    t['button_scanner_s']=t.onset+t.image_relative_first_rt_s
    return t,asc[0]
def events_for(t,w):
    rows=[]
    for r in t.itertuples():
        o=float(r.onset);rt=float(r.image_relative_first_rt_s)
        if r.status=='target':
            require(rt>1 and rt-w>0,'Invalid target Search')
            parts=[(f'search_{r.condition}',o,rt-w),(f'pre_{r.condition}',o+rt-w,w)]
            require(np.isclose(parts[0][1]+parts[0][2],parts[1][1],atol=1e-10) and np.isclose(parts[1][1]+w,o+rt,atol=1e-10),'Broken partition')
        else:parts=[(r.status,o,rt if r.status=='fast_response' else float(r.raw_duration_s))]
        for name,onset,duration in parts:
            require(duration>0 and onset>=o-1e-9 and onset+duration<=o+r.raw_duration_s+1e-9,'Invalid event boundaries')
            rows.append(dict(TrialID=int(r.TrialID),condition=r.condition,status=r.status,onset=onset,duration=duration,trial_type=name,modulation=1.))
    return pd.DataFrame(rows).sort_values(['onset','trial_type']).reset_index(drop=True)
def build_design(e,conf,n):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        dm=make_first_level_design_matrix(np.arange(n,dtype=float),e[['onset','duration','trial_type','modulation']],hrf_model='spm',drift_model='cosine',high_pass=1/128.,add_regs=conf.to_numpy(float),add_reg_names=MOTION,oversampling=50)
    repaired=any('singular' in str(w.message).lower() or 'regulariz' in str(w.message).lower() for w in recorded)
    return dm,repaired,[str(w.message) for w in recorded]
def design_qc(dm):
    x=dm.to_numpy(float);finite=bool(np.isfinite(x).all());rank=int(np.linalg.matrix_rank(x));missing=[c for c in FOCAL if c not in dm]
    vifs={}
    for c in FOCAL:
        if c not in dm:continue
        y=dm[c].to_numpy(float);others=dm.drop(columns=c).to_numpy(float)
        resid=y-others@np.linalg.lstsq(others,y,rcond=None)[0]
        vif=float(np.sum((y-y.mean())**2)/max(float(resid@resid),np.finfo(float).tiny))
        vifs['vif_'+c]=vif
    maxv=max(vifs.values(),default=1e300)
    return dict(rank=rank,n_columns=len(dm.columns),finite=finite,full_rank=rank==len(dm.columns),missing=','.join(missing),max_focal_vif=maxv,passed=finite and rank==len(dm.columns) and not missing and maxv<=10,**vifs)
def motion(path,n):
    df=pd.read_csv(path,sep='\t')[MOTION].replace([np.inf,-np.inf],np.nan).fillna(0.)
    require(len(df)==n,'Motion/scan mismatch');return df

def prepare():
    require(not (OUT/'first_level').exists() and not (OUT/'group').exists(),'Cannot reprepare after fits')
    for name in ['designs','events','code']:(OUT/name).mkdir(exist_ok=True)
    plan=dict(plan_sha256=sha(OUT/'FROZEN_PLAN.md'),implementation_sha256=sha(__file__),subjects=SUBJECTS,widths=WIDTHS,software=versions(),mask_sha256=MASK_HASH,status='frozen_before_BOLD_signal_access',source_sha256={str(p):sha(p) for p in SOURCES})
    if (OUT/'analysis_plan.json').exists():require(json.loads((OUT/'analysis_plan.json').read_text())==plan,'Frozen plan changed; do not silently overwrite')
    else:dump(OUT/'analysis_plan.json',plan)
    for p in SOURCES:shutil.copy2(p,OUT/'code'/p.name)
    srcmask=CANON/'first_level_masks/common_n19_intersection_mask.nii.gz'
    require(sha(srcmask)==MASK_HASH,'Canonical mask hash');shutil.copy2(srcmask,MASK)
    mask=nib.load(MASK);require(int(np.asarray(mask.dataobj).astype(bool).sum())==48279,'Mask voxel count')
    behavior=pd.read_csv(BEHAVIOR);qcs=[];counts=[];alltrials=[];inputs={str(BEHAVIOR):sha(BEHAVIOR),str(srcmask):sha(srcmask)};headers=[];generated=[]
    for s in SUBJECTS:
        bold,meta,confpath=paths(s);img=nib.load(bold) # HEADER ONLY: never access BOLD dataobj/get_fdata here.
        n=img.shape[-1];require(np.isclose(json.loads(meta.read_text())['RepetitionTime'],1.),'TR mismatch')
        require(img.shape[:3]==mask.shape and np.allclose(img.affine,mask.affine),'BOLD/mask grid mismatch')
        headers.append(dict(subject=s,path=str(bold),shape=list(img.shape),size_bytes=bold.stat().st_size,mtime_ns=bold.stat().st_mtime_ns))
        t,asc=trials_for(s,behavior,n);alltrials.append(t)
        for p in [meta,confpath,asc]:inputs[str(p)]=sha(p)
        p=OUT/'events'/f'{s}_trials.csv';t.to_csv(p,index=False);generated.append(p)
        for c in CONDITIONS:
            ct=t[t.condition.eq(c)]
            counts.append(dict(subject=s,condition=c,n_trials=len(ct),n_responded=int(ct.image_relative_first_rt_s.notna().sum()),n_target_both_widths=int(ct.target.sum()),n_fast_pooled=int(ct.status.eq('fast_response').sum()),n_no_response_pooled=int(ct.status.eq('no_insight').sum())))
        conf=motion(confpath,n);models={};eventsets={}
        for label,w in WIDTHS.items():
            e=events_for(t,w);dm,repaired,warns=build_design(e,conf,n);q=design_qc(dm)
            q.update(subject=s,width=label,n_scans=n,numerical_rank_repair=repaired,design_warnings=' | '.join(warns));q['passed']=bool(q['passed'] and not repaired);qcs.append(q)
            ep=OUT/'events'/f'{s}_{label}.tsv';e.to_csv(ep,sep='\t',index=False)
            dp=OUT/'designs'/f'{s}_{label}.csv';dm.to_csv(dp,index=False,float_format='%.17g')
            ap=OUT/'designs'/f'{s}_{label}.npz';np.savez_compressed(ap,x=dm.to_numpy(float),columns=np.array(dm.columns,dtype=str),frame_times=dm.index.to_numpy(float))
            generated.extend([ep,dp,ap]);models[label]=dm;eventsets[label]=e
            print(s,label,'rank',q['rank'],'/',q['n_columns'],'maxVIF',round(q['max_focal_vif'],4),'PASS',q['passed'],flush=True)
        a,b=eventsets.values();pd.testing.assert_frame_equal(a[~a.status.eq('target')].reset_index(drop=True),b[~b.status.eq('target')].reset_index(drop=True))
        require(set(a[a.status.eq('target')].TrialID)==set(b[b.status.eq('target')].TrialID),'Target set mismatch')
        d0,d1=models.values();require(d0.columns.tolist()==d1.columns.tolist(),'Column mismatch')
        pd.testing.assert_frame_equal(d0.drop(columns=FOCAL),d1.drop(columns=FOCAL))
    qc=pd.DataFrame(qcs);qc.to_csv(OUT/'design_qc.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'trial_counts_by_subject_condition.csv',index=False)
    t=pd.concat(alltrials,ignore_index=True);t.to_csv(OUT/'all_trials.csv',index=False)
    gate=dict(passed=bool(len(qc)==38 and qc.passed.all()),n_designs=len(qc),full_rank_designs=int(qc.full_rank.sum()),max_vif_by_width=qc.groupby('width').max_focal_vif.max().to_dict(),failed_designs=qc[~qc.passed].to_dict('records'),matched_targets_verified=True,nuisance_identity_verified=True,partition_boundaries_verified=True,positive_search_verified=True,BOLD_signal_access=False,n_trials=len(t),n_target=int(t.target.sum()),n_fast_pooled=int(t.status.eq('fast_response').sum()),n_no_response=int(t.status.eq('no_insight').sum()))
    dump(OUT/'design_gate.json',gate);dump(OUT/'bold_header_manifest.json',headers);dump(OUT/'source_manifest.json',inputs)
    generated += [OUT/p for p in ['analysis_plan.json','FROZEN_PLAN.md','design_qc.csv','trial_counts_by_subject_condition.csv','all_trials.csv','design_gate.json','bold_header_manifest.json','source_manifest.json','common_n19_intersection_mask.nii.gz']]
    dump(OUT/'prepared_checksums.json',{str(p.relative_to(OUT)):sha(p) for p in generated})
    print(json.dumps(gate,indent=2),flush=True)
    return gate['passed']

def verify(require_pass=False,include_fits=False):
    plan=json.loads((OUT/'analysis_plan.json').read_text());require(sha(__file__)==plan['implementation_sha256'],'Runner changed since prepare')
    require(versions()==plan['software'],'Software/environment changed')
    for p,h in json.loads((OUT/'prepared_checksums.json').read_text()).items():require(sha(OUT/p)==h,'Prepared checksum mismatch '+p)
    for p,h in {**plan['source_sha256'],**json.loads((OUT/'source_manifest.json').read_text())}.items():require(sha(p)==h,'Source changed '+p)
    for h in json.loads((OUT/'bold_header_manifest.json').read_text()):
        st=Path(h['path']).stat();require(st.st_size==h['size_bytes'] and st.st_mtime_ns==h['mtime_ns'],'BOLD source changed')
    rows=[]
    for s in SUBJECTS:
        matrices=[];events=[]
        t=pd.read_csv(OUT/'events'/f'{s}_trials.csv')
        for label,w in WIDTHS.items():
            a=np.load(OUT/'designs'/f'{s}_{label}.npz',allow_pickle=False);dm=pd.DataFrame(a['x'],columns=a['columns']);qc=design_qc(dm);rows.append(qc)
            saved=pd.read_csv(OUT/'designs'/f'{s}_{label}.csv');np.testing.assert_allclose(dm,saved,rtol=1e-13,atol=1e-13)
            e=events_for(t,w);observed=pd.read_csv(OUT/'events'/f'{s}_{label}.tsv',sep='\t');pd.testing.assert_frame_equal(e,observed,check_exact=False,rtol=1e-12,atol=1e-12)
            rebuilt,repair,_=build_design(e,motion(paths(s)[2],len(dm)),len(dm));require(not repair,'Design builder rank repair')
            np.testing.assert_allclose(dm,rebuilt,atol=1e-12,rtol=1e-12);matrices.append(dm);events.append(e)
        pd.testing.assert_frame_equal(matrices[0].drop(columns=FOCAL),matrices[1].drop(columns=FOCAL))
        pd.testing.assert_frame_equal(events[0][~events[0].status.eq('target')].reset_index(drop=True),events[1][~events[1].status.eq('target')].reset_index(drop=True))
    gate=json.loads((OUT/'design_gate.json').read_text());require(gate['passed']==all(r['passed'] for r in rows),'Gate readback discrepancy')
    if require_pass:require(gate['passed'],'ALL-38 GATE FAILED: fits and group forbidden; no gate relaxation')
    if include_fits:
        for s in SUBJECTS:verify_fit(s)
    result=dict(verified=True,designs=38,gate_passed=gate['passed'],BOLD_signal_access=False,fit_checks=include_fits)
    dump(OUT/'verification.json',result);print('VERIFY',json.dumps(result),flush=True);return result

def save64(img,path):
    data=np.asarray(img.get_fdata(),dtype=np.float64);nib.Nifti1Image(data,img.affine).to_filename(path)
    read=nib.load(path);require(read.get_data_dtype()==np.dtype('float64'),'Nonfloat64 output')
    np.testing.assert_array_equal(read.get_fdata(),data);require(np.isfinite(data).all(),'Nonfinite output')
def contrast(columns,weights):
    v=np.zeros(len(columns))
    for c,w in weights.items():v[list(columns).index(c)]=w
    return v

def fit_subject(s):
    verify(require_pass=True)
    dest=OUT/'first_level'/s;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'provenance.json').exists():verify_fit(s);print('Verified complete; skipping',s);return
    bold,_,_=paths(s);files={}
    for label in WIDTHS:
        a=np.load(OUT/'designs'/f'{s}_{label}.npz',allow_pickle=False);dm=pd.DataFrame(a['x'],columns=a['columns'],index=a['frame_times'])
        model=FirstLevelModel(t_r=1.,slice_time_ref=0.,hrf_model='spm',drift_model='cosine',high_pass=1/128.,noise_model='ar1',smoothing_fwhm=6.,standardize=True,signal_scaling=0,minimize_memory=True,n_jobs=1,mask_img=str(MASK),memory=str(OUT/'cache'/s),memory_level=0)
        model.fit(str(bold),design_matrices=dm)
        np.testing.assert_array_equal(model.design_matrices_[0].to_numpy(),dm.to_numpy())
        for name,weights in {**{c:{c:1.} for c in FOCAL},**CONTRASTS}.items():
            path=dest/f'{label}_{name}_effect.nii.gz';save64(model.compute_contrast(contrast(dm.columns,weights),output_type='effect_size'),path);files[str(path.relative_to(OUT))]=sha(path)
        del model
    dump(dest/'provenance.json',dict(subject=s,implementation_sha256=sha(__file__),prepared_manifest_sha256=sha(OUT/'prepared_checksums.json'),bold_sha256=sha(bold),output_sha256=files,software=versions()))
    verify_fit(s);print('FIT COMPLETE',s,flush=True)
def verify_fit(s):
    prov=json.loads((OUT/'first_level'/s/'provenance.json').read_text());require(prov['implementation_sha256']==sha(__file__) and prov['prepared_manifest_sha256']==sha(OUT/'prepared_checksums.json'),'Stale fit')
    require(len(prov['output_sha256'])==16,'Incomplete fit outputs')
    for p,h in prov['output_sha256'].items():
        path=OUT/p;require(sha(path)==h,'Map checksum mismatch');img=nib.load(path)
        require(img.get_data_dtype()==np.dtype('float64') and img.shape==nib.load(MASK).shape,'Map dtype/grid')
        require(np.allclose(img.affine,nib.load(MASK).affine) and np.isfinite(img.get_fdata()).all(),'Invalid map')
def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);adj=np.minimum(1.,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));out=np.empty_like(p);out[order]=adj;return out
SCHEMA={'map_key':'string','width':'string','contrast':'string','cluster_id':'int64','mni_x':'float64','mni_y':'float64','mni_z':'float64','cluster_voxels':'int64','peak_t':'float64','peak_p_uncorrected_one_sided':'float64','cluster_pFWE':'float64','within_map_significant':'bool','cluster_pFWE_bonferroni4':'float64','cluster_bonferroni4_significant':'bool'}
def extract_clusters(key,width,name,timg,sizeimg,logpimg):
    td=timg.get_fdata();sz=sizeimg.get_fdata();lp=logpimg.get_fdata();labels,n=ndimage.label(td>stats.t.isf(.001,18),ndimage.generate_binary_structure(3,1));rows=[]
    for cid in range(1,n+1):
        mask=labels==cid;indices=np.argwhere(mask);ijk=indices[np.argmax(td[mask])];xyz=nib.affines.apply_affine(timg.affine,ijk);k=int(mask.sum());p=float(10**(-lp[mask].max()));peak=float(td[tuple(ijk)])
        require(np.allclose(sz[mask],k),'Permutation cluster extent mismatch')
        rows.append(dict(map_key=key,width=width,contrast=name,cluster_id=cid,mni_x=float(xyz[0]),mni_y=float(xyz[1]),mni_z=float(xyz[2]),cluster_voxels=k,peak_t=peak,peak_p_uncorrected_one_sided=float(stats.t.sf(peak,18)),cluster_pFWE=p,within_map_significant=p<.05,cluster_pFWE_bonferroni4=min(1.,4*p),cluster_bonferroni4_significant=4*p<.05))
    return pd.DataFrame(rows,columns=SCHEMA).astype(SCHEMA)
def group(jobs):
    require(1<=jobs<=8,'Group jobs must be 1..8');verify(require_pass=True,include_fits=True)
    dest=OUT/'group';dest.mkdir(exist_ok=True);design=pd.DataFrame({'intercept':np.ones(19)});tables=[];summary=[];manifest=[]
    for index,(width,name) in enumerate((w,c) for w in WIDTHS for c in CONTRASTS):
        key=width+'_'+name;maps=[str(OUT/'first_level'/s/f'{key}_effect.nii.gz') for s in SUBJECTS]
        manifest.extend(dict(map_key=key,subject=s,path=p,sha256=sha(p)) for s,p in zip(SUBJECTS,maps))
        param=SecondLevelModel(mask_img=str(MASK),smoothing_fwhm=None).fit(maps,design_matrix=design)
        timg=param.compute_contrast('intercept',output_type='stat');save64(timg,dest/f'{key}_parametric_t.nii.gz');save64(param.compute_contrast('intercept',output_type='z_score'),dest/f'{key}_parametric_z.nii.gz')
        results=non_parametric_inference(maps,design_matrix=design,second_level_contrast='intercept',mask=str(MASK),smoothing_fwhm=None,model_intercept=True,n_perm=50000,two_sided_test=False,random_state=20260908+1009*index,n_jobs=jobs,verbose=1,threshold=.001,tfce=False)
        for output,img in results.items():save64(img,dest/f'{key}_{output}.nii.gz')
        np.testing.assert_allclose(results['t'].get_fdata(),timg.get_fdata(),rtol=1e-5,atol=1e-5)
        table=extract_clusters(key,width,name,timg,results['size'],results['logp_max_size']);table.to_csv(dest/f'{key}_clusters.csv',index=False);tables.append(table)
        summary.append(dict(map_key=key,width=width,contrast=name,n_cdt_clusters=len(table),n_within_map_significant=int(table.within_map_significant.sum()),minimum_cluster_pFWE=float(table.cluster_pFWE.min()) if len(table) else 1.,seed=20260908+1009*index))
    sm=pd.DataFrame(summary);sm['map_min_pFWE_holm_global4']=holm(sm.minimum_cluster_pFWE)
    for w in WIDTHS:
        idx=sm.index[sm.width.eq(w)];sm.loc[idx,'map_min_pFWE_holm_width2']=holm(sm.loc[idx,'minimum_cluster_pFWE'])
    sm.to_csv(dest/'map_summary.csv',index=False)
    clusters=pd.concat(tables,ignore_index=True).astype(SCHEMA);clusters.to_csv(dest/'all_cdt_clusters.csv',index=False)
    clusters[clusters.within_map_significant].to_csv(dest/'significant_clusters_within_map.csv',index=False)
    clusters[['map_key','cluster_id','cluster_pFWE','cluster_pFWE_bonferroni4','cluster_bonferroni4_significant']].to_csv(dest/'cluster_specific_bonferroni4.csv',index=False)
    dump(dest/'cluster_schema.json',SCHEMA);pd.DataFrame(manifest).to_csv(dest/'input_manifest.csv',index=False)
    dump(dest/'metadata.json',dict(n_subjects=19,n_perm=50000,CDT_p=.001,one_sided=True,connectivity=6,statistic='maximum cluster extent',holm='map minima, within width2 and primary global4; NOT cluster-specific',software=versions(),prepared_manifest_sha256=sha(OUT/'prepared_checksums.json'),mask_sha256=sha(MASK)))
    dump(dest/'checksums.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='checksums.json'})
    verify_group();print('GROUP COMPLETE',flush=True)
def verify_group():
    dest=OUT/'group'
    for p,h in json.loads((dest/'checksums.json').read_text()).items():require(sha(dest/p)==h,'Group checksum mismatch')
    t=pd.read_csv(dest/'all_cdt_clusters.csv',dtype=SCHEMA);sig=pd.read_csv(dest/'significant_clusters_within_map.csv',dtype=SCHEMA)
    pd.testing.assert_frame_equal(t[t.within_map_significant].reset_index(drop=True),sig)
    sm=pd.read_csv(dest/'map_summary.csv');require(len(sm)==4,'Incomplete map family');np.testing.assert_allclose(holm(sm.minimum_cluster_pFWE),sm.map_min_pFWE_holm_global4)
    for w in WIDTHS:
        sub=sm[sm.width.eq(w)];np.testing.assert_allclose(holm(sub.minimum_cluster_pFWE),sub.map_min_pFWE_holm_width2)
    for p in dest.glob('*.nii.gz'):require(nib.load(p).get_data_dtype()==np.dtype('float64'),'Group map not float64')
    print('GROUP VERIFIED',flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit-subject','group','verify']);p.add_argument('--subject',choices=SUBJECTS);p.add_argument('--jobs',type=int,default=8);p.add_argument('--include-fits',action='store_true');p.add_argument('--include-group',action='store_true');a=p.parse_args()
    if a.stage=='prepare':sys.exit(0 if prepare() else 2)
    elif a.stage=='fit-subject':require(a.subject is not None,'--subject required');fit_subject(a.subject)
    elif a.stage=='group':group(a.jobs)
    else:
        verify(include_fits=a.include_fits)
        if a.include_group:verify_group()
if __name__=='__main__':main()
