#!/usr/bin/env python3
"""Both models at matched nominal seven-voxel 3mm, with five-voxel physical sensitivity."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,norm
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent;GLM=HERE.parents[1];M2=GLM/'roi_behavior_analysis/joint_m2_three_peaks_20260915'
CONDS=('CA','JX','OI');ROIS=('angular','precuneus');PHASES=('pre','post');OUTCOMES=('rt','reports')
SEED=20260924;B=50000
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def corr(x,y):
 x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
 return np.sum(x*y,axis=-1)/np.sqrt(np.sum(x*x,axis=-1)*np.sum(y*y,axis=-1))
PRE=HERE/'pre_extractions.csv';POST=HERE/'post_extractions.csv';BEH=M2/'participant_behavior.csv'
pre={(r['subject'],r['condition'],r['roi']):r for r in read(PRE)};post={(r['subject'],r['condition'],r['roi']):r for r in read(POST)};beh={(r['subject'],r['condition']):r for r in read(BEH)}
ids=sorted({s for s,c in beh});assert len(ids)==19 and len(beh)==57 and 'sub-005' not in ids
assert set(pre)==set(post)=={(s,c,r) for s in ids for c in CONDS for r in ROIS}
values=[]
for s in ids:
 for c in CONDS:
  values.append(dict(subject=s,condition=c,rt_raw_s=beh[s,c]['mean_first_insight_rt_s'],mean_reports_per_presentation=beh[s,c]['mean_reports_per_presentation'],**{f'{phase}_{roi}_{rule}_beta':(pre if phase=='pre' else post)[s,c,roi][f'beta_{rule}'] for phase in PHASES for roi in ROIS for rule in ('grid7','physical5')},**{f'pre_{roi}_old6_beta':pre[s,c,roi]['beta_old6mm'] for roi in ROIS}))
save(HERE/'participant_values.csv',values)
results={}
for rule in ('grid7','physical5'):
 rows=[];raw={}
 for phase in PHASES:
  for roi in ROIS:
   for c in CONDS:
    sub=[x for x in values if x['condition']==c]
    x=np.asarray([float(z[f'{phase}_{roi}_{rule}_beta']) for z in sub]);assert np.isfinite(x).all()
    for outcome in OUTCOMES:
     y=np.asarray([float(z['rt_raw_s' if outcome=='rt' else 'mean_reports_per_presentation']) for z in sub]);assert np.isfinite(y).all()
     r,p=pearsonr(x,y);lo,hi=np.tanh(np.arctanh(r)+norm.ppf([.025,.975])/np.sqrt(16))
     rows.append(dict(rule=rule,phase=phase,roi=roi,condition=c,outcome=outcome,n=19,r=float(r),ci95_low=float(lo),ci95_high=float(hi),p_raw=float(p)))
     raw[phase,roi,c,outcome]=(x,y)
 assert len(rows)==24
 for row,p in zip(rows,multipletests([z['p_raw'] for z in rows],method='holm')[1]):row['p_holm24']=float(p)
 fam=defaultdict(list)
 for j,z in enumerate(rows):fam[z['phase'],z['roi'],z['outcome']].append(j)
 assert len(fam)==8 and all(len(v)==3 for v in fam.values())
 for (phase,roi,outcome),indices in fam.items():
  assert {rows[j]['condition'] for j in indices}==set(CONDS)
  for j,p in zip(indices,multipletests([rows[k]['p_raw'] for k in indices],method='holm')[1]):
   rows[j]['family_id']=f'{phase}:{roi}:{outcome}';rows[j]['family_size']=3;rows[j]['p_holm3']=float(p)
 save(HERE/f'correlations_{rule}.csv',rows)
 idx=np.random.default_rng(SEED).integers(0,len(ids),size=(B,len(ids)),dtype=np.int16)
 pairs=[]
 for phase in PHASES:
  estimates=[]
  for c in CONDS:
   x,y=raw[phase,'angular',c,'rt'];estimates.append((float(corr(x,y)),corr(x[idx],y[idx])))
  for k,name in ((1,'CA-minus-JX'),(2,'CA-minus-OI')):
   d=estimates[0][0]-estimates[k][0];boot=estimates[0][1]-estimates[k][1];se=float(boot.std(ddof=1));assert np.isfinite(boot).all() and se>0
   pairs.append(dict(rule=rule,phase=phase,contrast=name,n=19,r_CA=estimates[0][0],r_comparator=estimates[k][0],delta_r=d,bootstrap_se=se,boot=boot))
 null=[np.abs((z['boot']-z['delta_r'])/z['bootstrap_se']) for z in pairs];maxT=np.maximum.reduce(null);cut=np.quantile(maxT,.95,method='higher')
 for k,z in enumerate(pairs):
  t=abs(z['delta_r'])/z['bootstrap_se']
  z['p_raw_centered_bootstrap']=(1+int(np.count_nonzero(null[k]>=t)))/(B+1)
  z['p_maxT4']=(1+int(np.count_nonzero(maxT>=t)))/(B+1)
  z['simultaneous_ci_low']=z['delta_r']-cut*z['bootstrap_se'];z['simultaneous_ci_high']=z['delta_r']+cut*z['bootstrap_se'];z.pop('boot')
 save(HERE/f'angular_rt_differences_{rule}.csv',pairs)
 results[rule]=rows
 for z in rows:
  if z['outcome']=='rt':print(rule,z['phase'],z['roi'],z['condition'],f"r={z['r']:.3f}",f"Holm3={z['p_holm3']:.4f}",f"Holm24={z['p_holm24']:.4f}")
 for z in pairs:print(rule,'DIRECT',z['phase'],z['contrast'],f"delta={z['delta_r']:.3f}",f"maxT4={z['p_maxT4']:.4f}")
manifest=dict(plan_sha256=sha(HERE/'PLAN.md'),extraction_sha256=sha(HERE/'EXTRACTION_PROVENANCE.json'),script_sha256=sha(Path(__file__)),input_hashes={str(p):sha(p) for p in (PRE,POST,BEH)},subjects=ids,n=19,B=B,seed=SEED,families='Eight post-hoc Holm3 within phase*ROI*outcome; Holm24 within rule; maxT4 angular RT within rule. Grid7 primary; physical5 descriptive sensitivity.')
(HERE/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
