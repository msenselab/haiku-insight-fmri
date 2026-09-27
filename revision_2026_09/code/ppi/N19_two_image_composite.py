#!/usr/bin/env python3
"""Source-locked N19 positive composite exact sign-flip analysis."""
from pathlib import Path
import os
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import t as student_t
O=Path(os.environ['HAIKU_PROJECT_ROOT']).resolve()/'glm_unified/connectivity_analysis/ppi_precuneus_angular_historicalpeak_cajxavg_gt_oi_n19_20260923'; R=O.parent
if Path(__file__).resolve().parents[3] in O.parents:
 raise RuntimeError('Analysis output must be outside the public checkout')
A=R/'ppi_precuneus_angular_historicalpeak_firstfinal_n20n18_20260923/participant_coefficients.csv'
B=R/'ppi_precuneus_angular_ofc_condition_firstfinal_n20_20260922/participant_coefficients.csv'
HASHES={'angular_historical':'258a0b81252ca162c36f549d78ff3705a626384c4f0998c7423955971a8f1e66','ofc_left':'f9cbc137416fff8c3c267c8be4ec001afc4dde7034b6fb0a869085647ac459ff'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(v):
 n=len(v);observed=float(v.sum());pos=two=0
 for start in range(0,1<<n,16384):
  x=np.arange(start,min(start+16384,1<<n),dtype=np.uint32)
  signs=2*((x[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(float)-1.;y=signs@v
  pos+=int((y>=observed-1e-12*max(1,abs(observed))).sum());two+=int((abs(y)>=abs(observed)-1e-12*max(1,abs(observed))).sum())
 return pos,two
def holm(p):
 p=np.asarray(p,float);out=np.empty(len(p));last=0
 for rank,idx in enumerate(np.argsort(p,kind='stable')):last=max(last,(len(p)-rank)*p[idx]);out[idx]=min(1,last)
 return out
assert sha(A)==HASHES['angular_historical'] and sha(B)==HASHES['ofc_left']
rows=[];results=[];sources=[('angular_historical',A),('ofc_left',B)]
for target,path in sources:
 d=pd.read_csv(path);assert set(d.columns)=={'subject','target','phase','condition','beta'} and not d.duplicated(['subject','target','phase','condition']).any()
 d=d[d.target==target];assert len(d)==120
 for phase in ['first','final']:
  sub=d[d.phase==phase];assert len(sub)==60
  w=sub.pivot(index='subject',columns='condition',values='beta').sort_index();assert w.shape==(20,3) and set(w.columns)=={'CA','JX','OI'} and 'sub-005' in w.index
  w=w.drop(index='sub-005');assert w.shape==(19,3) and w.notna().all().all(); v=(.5*w.CA+.5*w.JX-w.OI).to_numpy(float);n=len(v);mean=v.mean();sd=v.std(ddof=1);sem=sd/np.sqrt(n);t=mean/sem;half=student_t.ppf(.975,n-1)*sem;positive,two=exact(v)
  for sid,row in w.iterrows():rows.append(dict(target=target,phase=phase,subject=sid,CA=row.CA,JX=row.JX,OI=row.OI,contrast=.5*row.CA+.5*row.JX-row.OI))
  results.append(dict(target=target,phase=phase,n=n,df=n-1,mean=float(mean),sd=float(sd),sem=float(sem),t=float(t),ci_low=float(mean-half),ci_high=float(mean+half),dz=float(mean/sd),positive_exceedances=positive,two_sided_exceedances=two,patterns=1<<n,p_directional=positive/(1<<n),p_two_sided=two/(1<<n)))
res=pd.DataFrame(results);res['holm4_directional']=holm(res.p_directional);res['holm4_two_sided']=holm(res.p_two_sided)
pd.DataFrame(rows).to_csv(O/'participant_vectors.csv',index=False,float_format='%.17g');res.to_csv(O/'group_results.csv',index=False,float_format='%.17g');(O/'source_hashes.json').write_text(json.dumps(dict(source_hashes=HASHES,exclusion='sub-005 only at group inference',target_source_paths=[str(A),str(B)],code_sha256=sha(Path(__file__))),indent=2)+'\n')
print(res[['target','phase','n','mean','ci_low','ci_high','t','p_directional','holm4_directional','p_two_sided','holm4_two_sided']].to_string(index=False))
