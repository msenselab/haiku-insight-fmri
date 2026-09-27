#!/usr/bin/env python3
"""N19 group-only condition-specific gPPI readback, two targets/two phases."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import t as student_t
O=Path(__file__).resolve().parent;R=O.parent
S={'angular_historical':R/'ppi_precuneus_angular_historicalpeak_firstfinal_n20n18_20260923/participant_coefficients.csv','ofc_left':R/'ppi_precuneus_angular_ofc_condition_firstfinal_n20_20260922/participant_coefficients.csv'}
H={'angular_historical':'258a0b81252ca162c36f549d78ff3705a626384c4f0998c7423955971a8f1e66','ofc_left':'f9cbc137416fff8c3c267c8be4ec001afc4dde7034b6fb0a869085647ac459ff'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signs(n):
 for start in range(0,1<<n,8192):
  bits=np.arange(start,min(start+8192,1<<n),dtype=np.uint32)
  yield 2*((bits[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(float)-1.
def exact(v):
 observed=abs(v.sum());count=0
 for sg in signs(len(v)):count+=int((np.abs(sg@v)>=observed-1e-12*max(1.,observed)).sum())
 return count/(1<<len(v)),count
def hotelling(v):
 n=v.shape[0];cross=v.T@v;obsmean=v.mean(axis=0)
 def t2(m):
  aa=(cross[0,0]-n*m[:,0]**2)/(n-1);bb=(cross[0,1]-n*m[:,0]*m[:,1])/(n-1);cc=(cross[1,1]-n*m[:,1]**2)/(n-1)
  return n*(cc*m[:,0]**2-2*bb*m[:,0]*m[:,1]+aa*m[:,1]**2)/(aa*cc-bb**2)
 observed=t2(obsmean[None,:])[0];count=0
 for sg in signs(n):count+=int((t2(sg@v/n)>=observed-1e-12*max(1.,abs(observed))).sum())
 return float(observed),count/(1<<n),count
def summary(v):
 n=len(v);m=float(v.mean());sd=float(v.std(ddof=1));se=sd/np.sqrt(n);t=m/se;h=float(student_t.ppf(.975,n-1)*se);p,e=exact(v)
 return dict(n=n,df=n-1,mean_beta=m,sd=sd,sem=float(se),t=float(t),ci95_low=m-h,ci95_high=m+h,dz=m/sd,p_raw_exact_two_sided=p,exact_exceedances=e,exact_patterns=1<<n)
def holm(p):
 x=np.asarray(p,float);out=np.empty(len(x));last=0.
 for k,j in enumerate(np.argsort(x,kind='stable')):last=max(last,(len(x)-k)*x[j]);out[j]=min(1.,last)
 return out
assert (O/'FROZEN_PLAN.md').exists() and not (O/'group_coefficients.csv').exists()
coef=[];pair=[];omni=[];vectors=[]
for target,path in S.items():
 assert sha(path)==H[target];d=pd.read_csv(path); assert set(d.columns)=={'subject','target','phase','condition','beta'} and not d.duplicated(['subject','target','phase','condition']).any();d=d[d.target==target];assert len(d)==120
 for phase in ['first','final']:
  w=d[d.phase==phase].pivot(index='subject',columns='condition',values='beta').sort_index();assert w.shape==(20,3) and set(w.columns)=={'CA','JX','OI'};w=w.drop(index='sub-005');assert w.shape==(19,3) and w.notna().all().all()
  for sid,r in w.iterrows():vectors.append(dict(target=target,phase=phase,subject=sid,CA=r.CA,JX=r.JX,OI=r.OI))
  for cond in ['CA','JX','OI']:coef.append(dict(target=target,phase=phase,condition=cond,**summary(w[cond].to_numpy(float))))
  dif={name:w[a].to_numpy(float)-w[b].to_numpy(float) for name,a,b in [('CA_minus_JX','CA','JX'),('CA_minus_OI','CA','OI'),('JX_minus_OI','JX','OI')]}
  for name,v in dif.items():pair.append(dict(target=target,phase=phase,contrast=name,**summary(v)))
  t,p,e=hotelling(np.column_stack([dif['CA_minus_JX'],dif['CA_minus_OI']]));omni.append(dict(target=target,phase=phase,n=19,T2=t,p_raw_exact=p,exact_exceedances=e,exact_patterns=2**19))
coef=pd.DataFrame(coef);pair=pd.DataFrame(pair);omni=pd.DataFrame(omni);coef['p_holm12']=holm(coef.p_raw_exact_two_sided);pair['p_holm12']=holm(pair.p_raw_exact_two_sided);pair['p_holm3_within_target_phase']=np.nan
for _,idx in pair.groupby(['target','phase']).groups.items():pair.loc[list(idx),'p_holm3_within_target_phase']=holm(pair.loc[list(idx),'p_raw_exact_two_sided'])
omni['p_holm4']=holm(omni.p_raw_exact)
pd.DataFrame(vectors).to_csv(O/'participant_vectors.csv',index=False,float_format='%.17g');coef.to_csv(O/'group_coefficients.csv',index=False,float_format='%.17g');pair.to_csv(O/'group_pairwise.csv',index=False,float_format='%.17g');omni.to_csv(O/'group_omnibus.csv',index=False,float_format='%.17g');(O/'source_hashes.json').write_text(json.dumps(dict(source_sha256=H,source_paths={k:str(v) for k,v in S.items()},exclusion='sub-005 only at group inference'),indent=2)+'\n')
print('OMNIBUS\n'+omni[['target','phase','T2','p_raw_exact','p_holm4']].to_string(index=False));print('CONDITIONS\n'+coef[['target','phase','condition','mean_beta','ci95_low','ci95_high','p_raw_exact_two_sided','p_holm12']].to_string(index=False));print('PAIRWISE\n'+pair[['target','phase','contrast','mean_beta','p_raw_exact_two_sided','p_holm3_within_target_phase','p_holm12']].to_string(index=False))
