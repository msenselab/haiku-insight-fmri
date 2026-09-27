#!/usr/bin/env python3
"""Frozen CSV-limited all-button final-versus-first Search–Pre extension."""
from __future__ import annotations
import os
import argparse, hashlib, json, platform, shutil
from pathlib import Path
from typing import Any
import nibabel as nib
import nilearn, numpy as np, pandas as pd, scipy
from joblib import Parallel, delayed
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from scipy import ndimage
from scipy.stats import t as t_distribution

ROOT=Path(os.environ["HAIKU_PROJECT_ROOT"])
OUT=ROOT/'glm_unified/preresponse_with_search_final_vs_first_20260921_r1'
PLAN=OUT/'FROZEN_PLAN.md'; DESIGN=OUT/'design_gate'; FIRST=OUT/'first_level'; GROUP=OUT/'group_inference'; MAN=OUT/'manifests'; SNAP=OUT/'code_snapshot'
SOURCE=ROOT/'glm_unified/preresponse_with_search_first_vs_all_buttons_20260903'
SOURCE_RUNNER=ROOT/'glm_unified/scripts/run_preresponse_with_search_first_vs_all_buttons_20260903.py'
SOURCE_BASE=ROOT/'glm_unified/scripts/run_adapted_preresponse_later_buttons_20260902.py'
SOURCE_PLAN=ROOT/'glm_unified/analysis_plans/PRE_RESPONSE_WITH_SEARCH_FIRST_VS_ALL_BUTTONS_PLAN_20260903.md'
SOURCE_FREEZE=SOURCE/'design_only/frozen_design.json'; SOURCE_EVENT_DIR=SOURCE/'design_only/all_buttons'
SOURCE_MASK=SOURCE/'masks/common_n19_input_mask_intersection.nii.gz'
BEHAVIOR=ROOT/'manuscript/behavioral_analysis/behavioral_data_all_subjects.csv'; FMRIPREP=ROOT/'derivatives/fmriprep'
ATLAS=ROOT/'glm_unified/atlas_cache'
SUBJECTS=[f'sub-{i:03d}' for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,23]]
CONDITIONS=['CA','JX','OI']; MOTION=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
TR=1.; FWHM=6.; HP=1/128.; HRF='spm'; NOISE='ar1'; VIF_LIMIT=10.; CDT=.001; NPERM=50000; SEED=20260921
VERSION='csv_limited_allbutton_joint_search_pre_final_vs_first_n19_20260921_r1_v1'
FOCAL={'pre_final_gt_first':{'pre_final_vs_first':1.}, 'search_final_gt_first':{'search_final_vs_first':1.}, 'pre_first_gt_final':{'pre_final_vs_first':-1.}, 'search_first_gt_final':{'search_final_vs_first':-1.}}
LABELS={'pre_final_gt_first':'Pre final > first','search_final_gt_first':'Search final > first','pre_first_gt_final':'Pre first > final','search_first_gt_final':'Search first > final'}
FAMILIES={'primary_final_gt_first':['pre_final_gt_first','search_final_gt_first'], 'secondary_first_gt_final':['pre_first_gt_final','search_first_gt_final']}
for d in [OUT,DESIGN,FIRST,GROUP,MAN,SNAP]: d.mkdir(parents=True,exist_ok=True)

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save_json(p:Path,x:Any): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def save64(img,path:Path):
 nib.Nifti1Image(np.asarray(img.get_fdata(),dtype=np.float64),img.affine).to_filename(path)
 r=nib.load(path); a=np.asarray(r.dataobj)
 if r.get_data_dtype()!=np.dtype('float64') or not np.isfinite(a).all() or np.unique(a).size<=256: raise RuntimeError(f'bad float64 map {path}')
def inputs(s):
 d=FMRIPREP/s/'func'; b=d/f'{s}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'; j=b.with_suffix('').with_suffix('.json'); c=d/f'{s}_task-haiku_desc-confounds_timeseries.tsv'; m=d/f'{s}_task-haiku_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
 for p in [b,j,c,m]:
  if not p.exists(): raise FileNotFoundError(p)
 return b,j,c,m
def motion(p,n):
 x=pd.read_csv(p,sep='\t'); missing=[z for z in MOTION if z not in x]
 if missing: raise RuntimeError(f'missing motion {missing}')
 x=x[MOTION].replace([np.inf,-np.inf],np.nan).fillna(0.)
 if len(x)!=n: raise RuntimeError('motion rows != scans')
 return x
def source_events(s):
 p=SOURCE_EVENT_DIR/f'{s}_events.tsv'
 if not p.exists(): raise FileNotFoundError(p)
 x=pd.read_csv(p,sep='\t')
 req={'onset','duration','trial_type','modulation','trial_index','button_order','button_time','trial_end'}
 if set(x.columns)!=req: raise RuntimeError(f'{s}: source columns changed {x.columns.tolist()}')
 return x
def extend_events(s):
 x=source_events(s).copy(); base=x.copy(); pre_rows=[]; search_rows=[]; coverage=[]
 prebase=base[base.trial_type.str.startswith('pre_')].copy()
 for trial,(ti,g) in enumerate(prebase.groupby('trial_index',sort=True),start=1):
  g=g.sort_values('button_order'); n=len(g); cond=str(g.iloc[0].trial_type).split('_',1)[1]
  if n<2: continue
  first=g.iloc[0].copy(); final=g.iloc[-1].copy()
  for row,mod in [(first,-.5),(final,.5)]:
   row['trial_type']='pre_final_vs_first'; row['modulation']=mod; pre_rows.append(row)
  sg=base[(base.trial_index.eq(ti))&(base.trial_type.str.startswith('search_'))&(base.duration>1e-9)].sort_values('button_order')
  first_search=sg[sg.button_order.eq(int(first.button_order))]; final_search=sg[sg.button_order.eq(int(final.button_order))]
  paired=(len(first_search)==1 and len(final_search)==1)
  if paired:
   for row,mod in [(first_search.iloc[0].copy(),-.5),(final_search.iloc[0].copy(),.5)]:
    row['trial_type']='search_final_vs_first'; row['modulation']=mod; search_rows.append(row)
  coverage.append({'subject':s,'trial_index':int(ti),'condition':cond,'n_response_buttons':n,'pre_paired':True,'search_paired':paired,'first_search_positive':len(first_search)==1,'final_search_positive':len(final_search)==1})
 pre=pd.DataFrame(pre_rows,columns=base.columns); search=pd.DataFrame(search_rows,columns=base.columns)
 y=pd.concat([base,pre,search],ignore_index=True).sort_values(['onset','trial_type','trial_index','button_order']).reset_index(drop=True)
 if len(pre)!=2*sum(r['pre_paired'] for r in coverage): raise RuntimeError('Pre coding mismatch')
 if len(search)!=2*sum(r['search_paired'] for r in coverage): raise RuntimeError('Search coding mismatch')
 cov=pd.DataFrame(coverage)
 return y,{'source_event_sha256':sha(SOURCE_EVENT_DIR/f'{s}_events.tsv'),'source_rows':len(base),'successor_rows':len(y),'pre_final_vs_first_events':len(pre),'search_final_vs_first_events':len(search),'pre_paired_trials':int(len(cov)),'search_paired_trials':int(cov.search_paired.sum()),'two_response_trials':int((cov.n_response_buttons==2).sum()),'three_response_trials':int((cov.n_response_buttons==3).sum()),'all_source_pre_events':int(base.trial_type.str.startswith('pre_').sum()),'all_source_search_events':int(base.trial_type.str.startswith('search_').sum()),'overlap_pre_pairs':int(sum(np.sum(np.diff(g.sort_values('button_order').button_time.to_numpy(float))<.5) for _,g in base[base.trial_type.str.startswith('pre_')].groupby('trial_index') if len(g)>1))},cov
def n_events(x): return x[['onset','duration','trial_type','modulation']].copy()
def vif(X:np.ndarray,j:int)->float:
 y=X[:,j]; q=np.delete(X,j,axis=1); den=float(np.sum((y-y.mean())**2)); r=y-q@np.linalg.lstsq(q,y,rcond=None)[0]
 return float(den/max(float(r@r),np.finfo(float).eps)) if den>np.finfo(float).eps else float('inf')
def cv(cols,key):
 c=np.zeros(len(cols)); c[cols.index(key)]=1.; return c
def gate_subject(s,save=True):
 b,j,c,_=inputs(s); n=nib.load(b).shape[-1]; tr=float(json.loads(j.read_text())['RepetitionTime'])
 if not np.isclose(tr,TR): raise RuntimeError(f'{s} TR')
 ev,ec,cov=extend_events(s); X=make_first_level_design_matrix(np.arange(n)*tr,events=n_events(ev),hrf_model=HRF,drift_model='cosine',high_pass=HP,add_regs=motion(c,n).to_numpy(float),add_reg_names=MOTION,min_onset=-24,oversampling=50)
 a=X.to_numpy(float); cols=X.columns.tolist(); req=[f'{q}_{z}' for q in ['pre','search'] for z in CONDITIONS]+['pre_final_vs_first','search_final_vs_first','post_response']+MOTION
 missing=[q for q in req if q not in cols]; rank=int(np.linalg.matrix_rank(a)); finite=bool(np.isfinite(a).all()); P=np.linalg.pinv(a.T@a); details={}
 for k in ['pre_final_vs_first','search_final_vs_first']:
  con=cv(cols,k); residual=con-a.T@np.linalg.pinv(a.T)@con; precision=float(con@P@con)
  details[k]={'vif':vif(a,cols.index(k)),'row_space_residual':float(np.linalg.norm(residual)),'precision_cXtXpinvc':precision,'estimable':bool(np.linalg.norm(residual)<=1e-10 and np.isfinite(precision) and precision>0)}
 passed=finite and not missing and rank==len(cols) and all(d['estimable'] and d['vif']<=VIF_LIMIT for d in details.values())
 out={'subject':s,**ec,'n_scans':n,'n_design_columns':len(cols),'design_rank':rank,'finite_design':finite,'full_rank':rank==len(cols),'missing_columns':'|'.join(missing),'required_columns_present':not missing,'passes_gate':bool(passed),'vif_pre_final_vs_first':details['pre_final_vs_first']['vif'],'vif_search_final_vs_first':details['search_final_vs_first']['vif'],'rowspace_pre_final_vs_first':details['pre_final_vs_first']['row_space_residual'],'rowspace_search_final_vs_first':details['search_final_vs_first']['row_space_residual'],'precision_pre_final_vs_first':details['pre_final_vs_first']['precision_cXtXpinvc'],'precision_search_final_vs_first':details['search_final_vs_first']['precision_cXtXpinvc'],'estimable_pre_final_vs_first':details['pre_final_vs_first']['estimable'],'estimable_search_final_vs_first':details['search_final_vs_first']['estimable']}
 if save:
  ev.to_csv(DESIGN/f'{s}_events.tsv',sep='\t',index=False); X.to_csv(DESIGN/f'{s}_design_matrix.csv',index=False); save_json(DESIGN/f'{s}_columns.json',cols); cov.to_csv(DESIGN/f'{s}_paired_trial_coverage.csv',index=False)
 return out
def manifest():
 p={'analysis_version':VERSION,'source':{str(x.relative_to(ROOT)):sha(x) for x in [SOURCE_RUNNER,SOURCE_BASE,SOURCE_PLAN,SOURCE_FREEZE,SOURCE_MASK,BEHAVIOR,PLAN]},'source_mask_voxels':int(np.asarray(nib.load(SOURCE_MASK).dataobj).astype(bool).sum()),'subjects':{}}
 if p['source_mask_voxels']!=52787: raise RuntimeError('source common mask voxel count mismatch')
 for s in SUBJECTS:
  b,j,c,m=inputs(s); p['subjects'][s]={str(x.relative_to(ROOT)):sha(x) for x in [b,j,c,m,SOURCE_EVENT_DIR/f'{s}_events.tsv']}
 save_json(MAN/'source_input_hash_manifest.json',p); return p
def prepare():
 if (DESIGN/'gate_summary.json').exists(): raise RuntimeError('gate already frozen; use a new successor directory')
 manifest(); q=pd.DataFrame([gate_subject(s) for s in SUBJECTS]).sort_values('subject'); q.to_csv(DESIGN/'participant_gate_qc.csv',index=False)
 summary={'analysis_version':VERSION,'n_subjects':len(SUBJECTS),'all_pass':bool(q.passes_gate.all()),'n_pass':int(q.passes_gate.sum()),'max_vif_pre_final_vs_first':float(q.vif_pre_final_vs_first.max()),'max_vif_search_final_vs_first':float(q.vif_search_final_vs_first.max()),'max_rowspace_residual':float(q[['rowspace_pre_final_vs_first','rowspace_search_final_vs_first']].to_numpy().max()),'min_precision':float(q[['precision_pre_final_vs_first','precision_search_final_vs_first']].to_numpy().min()),'total_source_rows':int(q.source_rows.sum()),'total_successor_rows':int(q.successor_rows.sum()),'total_pre_final_vs_first_events':int(q.pre_final_vs_first_events.sum()),'total_search_final_vs_first_events':int(q.search_final_vs_first_events.sum()),'total_multiresponse_trials':int(q.pre_paired_trials.sum()),'total_paired_search_trials':int(q.search_paired_trials.sum()),'total_two_response_trials':int(q.two_response_trials.sum()),'total_three_response_trials':int(q.three_response_trials.sum()),'total_overlap_pre_pairs':int(q.overlap_pre_pairs.sum()),'no_BOLD_fitted_or_inspected_before_gate':True,'gate':'finite + required columns + full rank + exact focal estimability (row-space residual <=1e-10) + pre_final_vs_first/search_final_vs_first VIF<=10 + finite positive cXtXpinvc'}
 save_json(DESIGN/'gate_summary.json',summary); shutil.copy2(Path(__file__),SNAP/Path(__file__).name); shutil.copy2(PLAN,SNAP/PLAN.name); print(json.dumps(summary,indent=2)); return summary
def frozen():
 p=DESIGN/'gate_summary.json'
 if not p.exists(): raise RuntimeError('run prepare first')
 x=json.loads(p.read_text()); h=json.loads((MAN/'source_input_hash_manifest.json').read_text())
 if h['source'][str(PLAN.relative_to(ROOT))]!=sha(PLAN): raise RuntimeError('plan hash changed after gate')
 if h['source'][str(SOURCE_RUNNER.relative_to(ROOT))]!=sha(SOURCE_RUNNER): raise RuntimeError('source runner hash changed after gate')
 return x
def first(s):
 f=frozen()
 if not f['all_pass']: raise RuntimeError('BOLD fit prohibited: gate failed')
 b,j,c,m=inputs(s); n=nib.load(b).shape[-1]; tr=float(json.loads(j.read_text())['RepetitionTime'])
 ev,ec,cov=extend_events(s); model=FirstLevelModel(t_r=tr,slice_time_ref=0.,smoothing_fwhm=FWHM,hrf_model=HRF,drift_model='cosine',high_pass=HP,noise_model=NOISE,mask_img=str(m),standardize=True,signal_scaling=0,minimize_memory=True,n_jobs=1,memory=str(OUT/'_nilearn_cache'),memory_level=1).fit(str(b),events=n_events(ev),confounds=motion(c,n))
 X=model.design_matrices_[0]; a=X.to_numpy(float); cols=X.columns.tolist(); checks=gate_subject(s,save=False)
 if cols!=json.loads((DESIGN/f'{s}_columns.json').read_text()) or not checks['passes_gate']: raise RuntimeError(f'{s}: fitted design mismatch/gate violation')
 sd=FIRST/s; sd.mkdir(exist_ok=True); ev.to_csv(sd/f'{s}_events.tsv',sep='\t',index=False); X.to_csv(sd/f'{s}_design_matrix.csv',index=False); save_json(sd/f'{s}_columns.json',cols)
 np.savez_compressed(sd/f'{s}_design_checkpoint.npz',X=a.astype(np.float64),columns=np.asarray(cols,dtype=str),contrast_pre_final_vs_first=cv(cols,'pre_final_vs_first'),contrast_search_final_vs_first=cv(cols,'search_final_vs_first'))
 outputs={}
 for key,w in FOCAL.items():
  v=sum(value*cv(cols,name) for name,value in w.items()); r=model.compute_contrast(v,output_type='all')
  for source,suffix in [('effect_size','effect'),('effect_variance','variance'),('stat','t'),('z_score','z')]:
   p=sd/f'{s}_{key}_{suffix}.nii.gz'; save64(r[source],p); outputs[f'{key}_{suffix}']=sha(p)
 prov={'analysis_version':VERSION,'subject':s,'gate_sha256':sha(DESIGN/'gate_summary.json'),'runner_sha256':sha(Path(__file__)),'source_hash_manifest_sha256':sha(MAN/'source_input_hash_manifest.json'),'events_sha256':sha(sd/f'{s}_events.tsv'),'design_sha256':sha(sd/f'{s}_design_matrix.csv'),'checkpoint_sha256':sha(sd/f'{s}_design_checkpoint.npz'),'outputs':outputs}
 save_json(sd/f'{s}_provenance.json',prov); return {'subject':s,**ec,'design_rank':int(np.linalg.matrix_rank(a)),'n_design_columns':len(cols),'max_vif':max(checks['vif_pre_final_vs_first'],checks['vif_search_final_vs_first']),'passes_recheck':checks['passes_gate'],'n_maps':len(outputs)}
def fit_all(njobs):
 f=frozen()
 if not f['all_pass']:
  save_json(OUT/'execution_status.json',{'status':'STOPPED_GATE','gate_sha256':sha(DESIGN/'gate_summary.json'),'BOLD_fit':'not_run'}); return
 q=pd.DataFrame(Parallel(n_jobs=njobs)(delayed(first)(s) for s in SUBJECTS)).sort_values('subject'); q.to_csv(FIRST/'participant_fit_qc.csv',index=False); save_json(OUT/'execution_status.json',{'status':'FIRST_LEVEL_COMPLETE','n_subjects':len(q),'gate_sha256':sha(DESIGN/'gate_summary.json')})
def holm(v):
 v=np.asarray(v,float); o=np.argsort(v); z=np.maximum.accumulate((len(v)-np.arange(len(v)))*v[o]); y=np.empty_like(v); y[o]=np.minimum(z,1); return y
def clusters(key,t,z,size,logp,thr,nperm,lookup):
 td=np.asarray(t.dataobj); zd=np.asarray(z.dataobj); sd=np.asarray(size.dataobj); lp=np.asarray(logp.dataobj); labs,nc=ndimage.label(td>thr,ndimage.generate_binary_structure(3,1)); vv=float(abs(np.linalg.det(t.affine[:3,:3]))); rows=[]
 for cid in range(1,nc+1):
  mask=labs==cid; ind=np.argwhere(mask); peak=ind[np.argmax(td[mask])]; xyz=nib.affines.apply_affine(t.affine,peak); n=int(mask.sum()); reported=sd[mask]; reported=reported[reported>0]
  if len(reported) and int(round(reported.max()))!=n: raise RuntimeError('cluster extent mismatch')
  cort,sub,qual=lookup(*xyz); rows.append({'contrast_key':key,'contrast':LABELS[key],'cluster_id':cid,'mni_x':float(xyz[0]),'mni_y':float(xyz[1]),'mni_z':float(xyz[2]),'peak_t':float(td[tuple(peak)]),'peak_z':float(zd[tuple(peak)]),'cluster_voxels':n,'cluster_mm3':n*vv,'cluster_pFWE_raw':float(10**(-np.max(lp[mask]))),'survives_within_map_cluster_FWE_05':bool(10**(-np.max(lp[mask]))<.05),'permutation_resolution':1/(nperm+1),'harvard_oxford_cortical':cort,'harvard_oxford_subcortical':sub,'qualified_anatomy':qual})
 return pd.DataFrame(rows)
def atlas_lookup():
 a=[]
 for name in ['cort-maxprob-thr0-2mm','sub-maxprob-thr0-2mm']:
  x=fetch_atlas_harvard_oxford(name,data_dir=str(ATLAS)); img=x.maps if hasattr(x.maps,'get_fdata') else nib.load(x.maps); a.append((img,np.asarray(img.dataobj),list(x.labels)))
 generic={'Background','Left Cerebral Cortex','Right Cerebral Cortex','Left Cerebral White Matter','Right Cerebral White Matter','Outside atlas'}
 def f(x,y,z):
  vals=[]
  for im,d,labs in a:
   ij=np.rint(np.linalg.inv(im.affine)@np.array([x,y,z,1.])).astype(int)[:3]; k=int(d[tuple(ij)]) if all(0<=ij[q]<d.shape[q] for q in range(3)) else 0; vals.append(labs[k] if 0<=k<len(labs) else 'Outside atlas')
  c,s=vals; anatomy=s if s not in generic else (c if c not in generic else 'Unclassified at peak')
  qual=anatomy if not ('White Matter' in s and c not in generic) else f'{c}; subcortical peak label {s} (qualified)'
  return c,s,qual
 return f
def group(nperm,njobs,seed):
 f=frozen()
 if not f['all_pass']: raise RuntimeError('group prohibited: gate failed')
 for s in SUBJECTS:
  p=FIRST/s/f'{s}_provenance.json'
  if not p.exists(): raise FileNotFoundError(p)
  v=json.loads(p.read_text())
  for key in FOCAL:
   for suf in ['effect','variance','t','z']:
    z=FIRST/s/f'{s}_{key}_{suf}.nii.gz'
    if sha(z)!=v['outputs'][f'{key}_{suf}']: raise RuntimeError(f'modified map {z}')
 mask=nib.load(SOURCE_MASK); mb=np.asarray(mask.dataobj).astype(bool); design=pd.DataFrame({'intercept':np.ones(len(SUBJECTS))}); thr=float(t_distribution.isf(CDT,len(SUBJECTS)-1)); lookup=atlas_lookup(); sums=[]; tabs=[]; checks=[]; inrows=[]
 for ix,key in enumerate(FOCAL):
  ps=[FIRST/s/f'{s}_{key}_effect.nii.gz' for s in SUBJECTS]
  for s,p in zip(SUBJECTS,ps): inrows.append({'subject':s,'contrast_key':key,'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size})
  sl=SecondLevelModel(mask_img=str(SOURCE_MASK),smoothing_fwhm=None).fit([str(p) for p in ps],design_matrix=design); t=sl.compute_contrast('intercept',output_type='stat'); z=sl.compute_contrast('intercept',output_type='z_score'); save64(t,GROUP/f'{key}_parametric_t.nii.gz'); save64(z,GROUP/f'{key}_parametric_z.nii.gz')
  stack=np.stack([np.asarray(nib.load(p).dataobj,dtype=float) for p in ps]); mean=stack.mean(0); sd=stack.std(0,ddof=1); man=nib.Nifti1Image(np.divide(mean,sd/np.sqrt(len(SUBJECTS)),out=np.zeros_like(mean),where=sd>0),mask.affine); save64(man,GROUP/f'{key}_manual_t.nii.gz'); d=float(np.max(np.abs(np.asarray(t.dataobj)[mb]-np.asarray(man.dataobj)[mb]))); checks.append({'contrast_key':key,'maximum_absolute_t_difference':d,'tolerance':1e-10,'passes':d<=1e-10})
  if d>1e-10: raise RuntimeError('manual t mismatch')
  thisseed=seed+ix*1009; out=non_parametric_inference([str(p) for p in ps],design_matrix=design,second_level_contrast='intercept',mask=str(SOURCE_MASK),smoothing_fwhm=None,model_intercept=True,n_perm=nperm,two_sided_test=False,random_state=thisseed,n_jobs=njobs,verbose=1,threshold=CDT,tfce=False)
  for n,img in out.items(): save64(img,GROUP/f'{key}_{n}.nii.gz')
  if not np.allclose(np.asarray(out['t'].dataobj)[mb],np.asarray(t.dataobj)[mb],rtol=1e-7,atol=1e-7): raise RuntimeError('permutation t mismatch')
  cl=clusters(key,t,z,out['size'],out['logp_max_size'],thr,nperm,lookup); tabs.append(cl); sig=cl[cl.survives_within_map_cluster_FWE_05] if not cl.empty else cl; sums.append({'family':next(a for a,b in FAMILIES.items() if key in b),'contrast_key':key,'contrast':LABELS[key],'n_cdt_clusters':len(cl),'n_within_map_significant_clusters':len(sig),'minimum_cluster_pFWE_raw':float(cl.cluster_pFWE_raw.min()) if not cl.empty else 1.,'random_seed':thisseed,'n_sign_flips':nperm,'one_sided':True,'voxel_CDT_p':CDT,'cluster_forming_t':thr})
 summary=pd.DataFrame(sums); summary['minimum_cluster_pFWE_holm_family_2maps']=np.nan
 for fam,ks in FAMILIES.items():
  ii=summary.index[summary.family.eq(fam)]; summary.loc[ii,'minimum_cluster_pFWE_holm_family_2maps']=holm(summary.loc[ii,'minimum_cluster_pFWE_raw'])
 summary['map_survives_holm_family_05']=summary.minimum_cluster_pFWE_holm_family_2maps<.05; summary.to_csv(GROUP/'contrast_summary_holm.csv',index=False)
 allc=pd.concat(tabs,ignore_index=True) if tabs else pd.DataFrame()
 if not allc.empty:
  q=summary.set_index('contrast_key'); allc['map_minimum_cluster_pFWE_holm_family_2maps']=allc.contrast_key.map(q.minimum_cluster_pFWE_holm_family_2maps); allc['map_survives_holm_family_05']=allc.contrast_key.map(q.map_survives_holm_family_05)
 allc.to_csv(GROUP/'all_cdt_clusters_labeled.csv',index=False); allc[allc.survives_within_map_cluster_FWE_05].to_csv(GROUP/'significant_clusters_within_map_labeled.csv',index=False); allc[(allc.survives_within_map_cluster_FWE_05)&(allc.map_survives_holm_family_05)].to_csv(GROUP/'clusters_in_family_holm_significant_maps.csv',index=False); pd.DataFrame(checks).to_csv(GROUP/'manual_t_reconstruction_qc.csv',index=False); pd.DataFrame(inrows).to_csv(GROUP/'input_manifest.csv',index=False)
 meta={'analysis_version':VERSION,'n_subjects':19,'common_mask_path':str(SOURCE_MASK),'common_mask_sha256':sha(SOURCE_MASK),'common_mask_voxels':int(mb.sum()),'group_inference':{'n_sign_flips_per_map':nperm,'one_sided':True,'voxel_CDT_p':CDT,'cluster_forming_t':thr,'connectivity':'six-neighbor','cluster_FWE':'maximum cluster extent within map','families':FAMILIES,'seeds':summary[['contrast_key','random_seed']].to_dict('records')},'software':{'python':platform.python_version(),'nilearn':nilearn.__version__,'nibabel':nib.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
 save_json(OUT/'metadata.json',meta); save_json(OUT/'execution_status.json',{'status':'COMPLETE_PENDING_INDEPENDENT_VERIFICATION','gate_sha256':sha(DESIGN/'gate_summary.json'),'group_summary_sha256':sha(GROUP/'contrast_summary_holm.csv')})
def report():
 g=json.loads((DESIGN/'gate_summary.json').read_text()); status=json.loads((OUT/'execution_status.json').read_text()) if (OUT/'execution_status.json').exists() else {}; lines=['# Results — CSV-limited all-button final-versus-first extension','',f"**Gate:** {'PASS' if g['all_pass'] else 'STOPPED'} ({g['n_pass']}/19).",'', '## Scope','', 'This N=19 analysis is CSV-limited: 1,670 retained presses from 779 responded presentations. It tests pooled final-versus-first associations adjusted for condition, not causal response-order effects and not condition×order effects. Overlapping 0.5-s Pre windows were retained additively.','']
 lines+=['## Design diagnostics','',f"- `pre_final_vs_first` events: {g['total_pre_final_vs_first_events']}; `search_final_vs_first` events: {g['total_search_final_vs_first_events']}; multiresponse trials: {g['total_multiresponse_trials']}; paired positive-Search trials: {g['total_paired_search_trials']}; retained overlapping Pre pairs: {g['total_overlap_pre_pairs']}.",f"- Maximum VIFs: Pre={g['max_vif_pre_final_vs_first']:.6g}, Search={g['max_vif_search_final_vs_first']:.6g}; maximum row-space residual={g['max_rowspace_residual']:.3g}; minimum c'(X'X)^+c={g['min_precision']:.6g}.",'']
 if not g['all_pass']: lines += ['## Stopped-gate status','', 'No BOLD fit, contrast maps, group test, or clusters were produced because the frozen gate failed.'];
 else:
  q=pd.read_csv(GROUP/'contrast_summary_holm.csv'); c=pd.read_csv(GROUP/'all_cdt_clusters_labeled.csv'); lines += ['## Directional map minima','', '| family | map | min raw within-map cluster pFWE | Holm2 | CDT clusters | within-map FWE clusters |', '|---|---|---:|---:|---:|---:|']
  for r in q.itertuples(index=False): lines.append(f'| {r.family} | {r.contrast} | {r.minimum_cluster_pFWE_raw:.6g} | {r.minimum_cluster_pFWE_holm_family_2maps:.6g} | {r.n_cdt_clusters} | {r.n_within_map_significant_clusters} |')
  lines+=['','## Within-map corrected clusters','']
  sig=c[c.survives_within_map_cluster_FWE_05]
  if sig.empty: lines+=['None.']
  else:
   for r in sig.itertuples(index=False): lines.append(f"- {r.contrast}: k={r.cluster_voxels}, pFWE={r.cluster_pFWE_raw:.6g}, peak=({r.mni_x:.1f}, {r.mni_y:.1f}, {r.mni_z:.1f}), t={r.peak_t:.3f}; {r.qualified_anatomy} [cortical: {r.harvard_oxford_cortical}; subcortical: {r.harvard_oxford_subcortical}].")
  lines+=['',f"Execution status: `{status.get('status','unknown')}`. Independent verification: `{json.loads((OUT/'independent_verification.json').read_text()).get('status','NOT_RUN') if (OUT/'independent_verification.json').exists() else 'NOT_RUN'}`."]
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--step',choices=['prepare','fit','group','report','all'],default='all'); p.add_argument('--n-jobs-first',type=int,default=2); p.add_argument('--n-jobs-group',type=int,default=8); p.add_argument('--n-perm',type=int,default=NPERM); p.add_argument('--seed',type=int,default=SEED); a=p.parse_args()
 if a.step in ['prepare','all']: prepare()
 if a.step in ['fit','all']: fit_all(a.n_jobs_first)
 if a.step in ['group','all']: group(a.n_perm,a.n_jobs_group,a.seed)
 if a.step in ['report','all']: report()
if __name__=='__main__': main()
