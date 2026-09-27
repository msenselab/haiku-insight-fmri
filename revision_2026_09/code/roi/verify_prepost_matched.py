#!/usr/bin/env python3
"""Independent NIfTI/checkpoint readback for both 3-mm rules, all 24 tests, four contrasts."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np,nibabel as nib
from scipy.stats import pearsonr,norm
from statsmodels.stats.multitest import multipletests
HERE=Path(os.environ['HAIKU_PROJECT_ROOT']).resolve()/'glm_unified/roi_behavior_analysis/prepost_matched3mm_20260924';GLM=HERE.parents[1];M2=GLM/'roi_behavior_analysis/joint_m2_three_peaks_20260915'
if Path(__file__).resolve().parents[3] in HERE.parents:
 raise RuntimeError('Analysis output must be outside the public checkout')
read=lambda path:list(csv.DictReader(path.open(newline='')))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
e=json.loads((HERE/'EXTRACTION_PROVENANCE.json').read_text());m=json.loads((HERE/'PROVENANCE.json').read_text())
assert e['plan_sha256']==m['plan_sha256']==sha(HERE/'PLAN.md') and e['script_sha256']==sha(HERE/'extract.py') and m['script_sha256']==sha(HERE/'analyze.py')
assert m['extraction_sha256']==sha(HERE/'EXTRACTION_PROVENANCE.json')
for path,h in e['source_hashes'].items():assert sha(Path(path))==h,path
for path,h in m['input_hashes'].items():assert sha(Path(path))==h,path
pre=read(HERE/'pre_extractions.csv');post=read(HERE/'post_extractions.csv');v=read(HERE/'participant_values.csv');beh={(z['subject'],z['condition']):z for z in read(M2/'participant_behavior.csv')};old={(z['subject'],z['roi'],z['condition']):z for z in read(M2/'participant_roi_betas.csv') if z['phase']=='pre'}
assert len(pre)==len(post)==114 and len(v)==57 and sorted({z['subject'] for z in v})==m['subjects']
mask=nib.load(e['pre_mask']);aff=mask.affine;common=mask.get_fdata()>0
pmap={(z['subject'],z['condition'],z['roi']):z for z in pre};qmap={(z['subject'],z['condition'],z['roi']):z for z in post}
for z in pre:
 s,c,roi=z['subject'],z['condition'],z['roi'];cp=Path(z['checkpoint_path']);assert sha(cp)==z['checkpoint_sha256']
 with np.load(cp) as t:
  indices={tuple(x):i for i,x in enumerate(t['union_ijk'])};theta=t['theta'];col=list(t['columns']).index('pre_'+c);center=e['centers']['pre'][roi]
  ci=np.rint(nib.affines.apply_affine(np.linalg.inv(aff),center)).astype(int)
  near=np.array([ci+[dx,dy,dz] for dx in range(-2,3) for dy in range(-2,3) for dz in range(-2,3) if tuple(ci+[dx,dy,dz]) in indices and common[tuple(ci+[dx,dy,dz])]])
  masks={'grid7':np.linalg.norm((near-ci)*3.,axis=1)<=3.+1e-9,'physical5':np.linalg.norm(nib.affines.apply_affine(aff,near)-center,axis=1)<=3.+1e-5,'old6':np.linalg.norm(nib.affines.apply_affine(aff,near)-center,axis=1)<=6.0}
  assert sum(masks['grid7'])==7 and sum(masks['physical5'])==5 and sum(masks['old6'])==27
  for field,kind in [('beta_grid7','grid7'),('beta_physical5','physical5'),('beta_old6mm','old6')]:
   val=np.mean([theta[col,indices[tuple(x)]] for x in near[masks[kind]]]);assert abs(val-float(z[field]))<1e-10,(s,c,roi,field)
  lab={'angular':'Angular Gyrus','precuneus':'Precuneus Cortex'}[roi]
  assert abs(float(z['beta_old6mm'])-float(old[s,lab,c]['beta']))<1e-10
for z in post:
 s,c,roi=z['subject'],z['condition'],z['roi'];p=Path(z['map_path']);assert sha(p)==z['map_sha256'];img=nib.load(p);center=e['centers']['post'][roi];ci=np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine),center)).astype(int)
 near=np.array([ci+[dx,dy,dz] for dx in range(-1,2) for dy in range(-1,2) for dz in range(-1,2)])
 masks={'grid7':np.linalg.norm((near-ci)*3.,axis=1)<=3.+1e-9,'physical5':np.linalg.norm((near-ci)@img.affine[:3,:3].T,axis=1)<=3.+1e-9}
 assert sum(masks['grid7'])==7 and sum(masks['physical5'])==5
 for field,kind in [('beta_grid7','grid7'),('beta_physical5','physical5')]:
  val=np.mean([img.get_fdata()[tuple(x)] for x in near[masks[kind]]]);assert abs(val-float(z[field]))<1e-10
for z in v:
 k=(z['subject'],z['condition']);assert abs(float(z['rt_raw_s'])-float(beh[k]['mean_first_insight_rt_s']))<1e-12
 assert abs(float(z['mean_reports_per_presentation'])-float(beh[k]['mean_reports_per_presentation']))<1e-12
 for phase,source in [('pre',pmap),('post',qmap)]:
  for roi in ('angular','precuneus'):
   for kind in ('grid7','physical5'):
    assert abs(float(z[f'{phase}_{roi}_{kind}_beta'])-float(source[*k,roi][f'beta_{kind}']))<1e-12
rsmap={}
for kind in ('grid7','physical5'):
 rows=read(HERE/f'correlations_{kind}.csv');assert len(rows)==24
 pvals=[];families={};raw={}
 for row in rows:
  sub=sorted((x for x in v if x['condition']==row['condition']),key=lambda z:z['subject']);assert len(sub)==19
  x=np.array([float(z[f"{row['phase']}_{row['roi']}_{kind}_beta"]) for z in sub]);y=np.array([float(z['rt_raw_s' if row['outcome']=='rt' else 'mean_reports_per_presentation']) for z in sub]);r,p=pearsonr(x,y);lo,hi=np.tanh(np.arctanh(r)+norm.ppf([.025,.975])/np.sqrt(16))
  assert row['n']=='19' and row['rule']==kind
  for key,val in [('r',r),('p_raw',p),('ci95_low',lo),('ci95_high',hi)]:assert abs(float(row[key])-val)<1e-10,(row,key)
  pvals.append(p);families.setdefault((row['phase'],row['roi'],row['outcome']),[]).append(row);raw[row['phase'],row['roi'],row['condition'],row['outcome']]=(x,y)
 for row,p in zip(rows,multipletests(pvals,method='holm')[1]):assert abs(float(row['p_holm24'])-p)<1e-10
 assert len(families)==8
 for key,group in families.items():
  assert len(group)==3 and {z['condition'] for z in group}=={'CA','JX','OI'}
  for row,p in zip(group,multipletests([float(z['p_raw']) for z in group],method='holm')[1]):
   assert abs(float(row['p_holm3'])-p)<1e-10 and row['family_id']==':'.join(key)
 idx=np.random.default_rng(m['seed']).integers(0,19,size=(m['B'],19),dtype=np.int16)
 def corr(x,y):
  x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
  return (x*y).sum(axis=-1)/np.sqrt((x*x).sum(axis=-1)*(y*y).sum(axis=-1))
 contrasts=[]
 for phase in ('pre','post'):
  res=[]
  for c in ('CA','JX','OI'):
   x,y=raw[phase,'angular',c,'rt'];res.append((pearsonr(x,y).statistic,corr(x[idx],y[idx])))
  for j,label in ((1,'CA-minus-JX'),(2,'CA-minus-OI')):
   d=res[0][0]-res[j][0];boot=res[0][1]-res[j][1];se=boot.std(ddof=1);contrasts.append((phase,label,d,se,np.abs((boot-d)/se)))
 maximum=np.maximum.reduce([z[4] for z in contrasts]);cut=np.quantile(maximum,.95,method='higher')
 for row,(phase,label,d,se,null) in zip(read(HERE/f'angular_rt_differences_{kind}.csv'),contrasts):
  assert (row['phase'],row['contrast'])==(phase,label)
  for key,val in [('delta_r',d),('bootstrap_se',se),('p_raw_centered_bootstrap',(1+np.count_nonzero(null>=abs(d)/se))/(m['B']+1)),('p_maxT4',(1+np.count_nonzero(maximum>=abs(d)/se))/(m['B']+1)),('simultaneous_ci_low',d-cut*se),('simultaneous_ci_high',d+cut*se)]:assert abs(float(row[key])-val)<1e-10,(row,key)
 rsmap[kind]=rows
oldpost=read(GLM/'roi_behavior_analysis/prepost_peak_correlation_reporting_postprecuneus_20260924/correlations_all24.csv');oldref={(r['phase'],r['roi'],r['condition'],r['outcome']):r for r in oldpost}
for row in rsmap['grid7']:
 if row['phase']=='post':assert abs(float(row['r'])-float(oldref[row['phase'],row['roi'],row['condition'],row['outcome']]['r']))<1e-10
print('PASS independent re-extraction 114 Pre/114 Post values, checkpoint/map hashes, 6-mm baseline, 48 correlations/CI/raw/Holm3/Holm24, 8 paired maxT contrasts; Post grid7 coefficients unchanged')
