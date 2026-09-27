#!/usr/bin/env python3
"""Independent group-only readback: source rows, QR-free vectors and exact sign tests."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy import stats
P=Path(__file__).resolve().parent;R=P.parent
sources={'angular_historical':R/'ppi_precuneus_angular_historicalpeak_firstfinal_n20n18_20260923/participant_coefficients.csv','ofc_left':R/'ppi_precuneus_angular_ofc_condition_firstfinal_n20_20260922/participant_coefficients.csv'}
hashes={'angular_historical':'258a0b81252ca162c36f549d78ff3705a626384c4f0998c7423955971a8f1e66','ofc_left':'f9cbc137416fff8c3c267c8be4ec001afc4dde7034b6fb0a869085647ac459ff'}
subject=pd.read_csv(P/'participant_vectors.csv');co=pd.read_csv(P/'group_coefficients.csv');pa=pd.read_csv(P/'group_pairwise.csv');om=pd.read_csv(P/'group_omnibus.csv');assert len(subject)==76 and len(co)==12 and len(pa)==12 and len(om)==4 and 'sub-005' not in set(subject.subject)
def sg(n):
 for start in range(0,2**n,8192):
  integers=np.arange(start,min(start+8192,2**n),dtype=np.uint32)
  yield 1.-2*((integers[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(float)
def scalar(x):
 threshold=abs(x.mean());cnt=0
 for signs in sg(len(x)):cnt+=int(np.count_nonzero(abs(signs@x/len(x))>=threshold-1e-12*max(1,threshold)))
 return cnt/(2**len(x))
def hotelling(x):
 n=len(x);m=x.mean(axis=0);s=np.cov(x,rowvar=False);observed=float(n*m@np.linalg.solve(s,m));cross=x.T@x;cnt=0
 for signs in sg(n):
  y=signs@x/n;a=(cross[0,0]-n*y[:,0]**2)/(n-1);b=(cross[0,1]-n*y[:,0]*y[:,1])/(n-1);c=(cross[1,1]-n*y[:,1]**2)/(n-1)
  statistic=n*(c*y[:,0]**2-2*b*y[:,0]*y[:,1]+a*y[:,1]**2)/(a*c-b*b);cnt+=int(np.count_nonzero(statistic>=observed-1e-12*max(1,observed)))
 return observed,cnt/(2**n)
def holm(x):
 x=np.asarray(x,float);result=np.empty(len(x));last=0
 for i,j in enumerate(np.argsort(x)):last=max(last,(len(x)-i)*x[j]);result[j]=min(1,last)
 return result
checks=0
for target,path in sources.items():
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[target]
 original=pd.read_csv(path);assert len(original.query('target == @target'))==120
 for phase in ['first','final']:
  x=original.query('target == @target and phase == @phase').pivot(index='subject',columns='condition',values='beta').drop(index='sub-005').sort_index();assert x.shape==(19,3)
  r=subject.query('target == @target and phase == @phase').set_index('subject').loc[x.index];np.testing.assert_allclose(r[['CA','JX','OI']].to_numpy(),x[['CA','JX','OI']].to_numpy(),atol=1e-15,rtol=0);checks+=19
  for c in ['CA','JX','OI']:
   v=x[c].to_numpy();rec=co.query('target == @target and phase == @phase and condition == @c').iloc[0];se=stats.sem(v);ci=stats.t.ppf(.975,18)*se;np.testing.assert_allclose([rec.mean_beta,rec.ci95_low,rec.ci95_high,rec.p_raw_exact_two_sided],[v.mean(),v.mean()-ci,v.mean()+ci,scalar(v)],atol=1e-11,rtol=0);checks+=4
  d={name:x[a].to_numpy()-x[b].to_numpy() for name,a,b in [('CA_minus_JX','CA','JX'),('CA_minus_OI','CA','OI'),('JX_minus_OI','JX','OI')]}
  for name,v in d.items():
   rec=pa.query('target == @target and phase == @phase and contrast == @name').iloc[0];np.testing.assert_allclose([rec.mean_beta,rec.p_raw_exact_two_sided],[v.mean(),scalar(v)],atol=1e-11,rtol=0);checks+=2
  t,p=hotelling(np.column_stack([d['CA_minus_JX'],d['CA_minus_OI']]));rec=om.query('target == @target and phase == @phase').iloc[0];np.testing.assert_allclose([rec.T2,rec.p_raw_exact],[t,p],atol=1e-11,rtol=0);checks+=2
for f,raw,adj in [(co,'p_raw_exact_two_sided','p_holm12'),(pa,'p_raw_exact_two_sided','p_holm12'),(om,'p_raw_exact','p_holm4')]:np.testing.assert_allclose(holm(f[raw]),f[adj],atol=1e-12,rtol=0);checks+=len(f)
for _,idx in pa.groupby(['target','phase']).groups.items():np.testing.assert_allclose(holm(pa.loc[list(idx),'p_raw_exact_two_sided']),pa.loc[list(idx),'p_holm3_within_target_phase'],atol=1e-12,rtol=0);checks+=len(idx)
# The prior N19 composite is exactly reconstructed from the condition rows.
composite=pd.read_csv(R/'ppi_precuneus_angular_historicalpeak_cajxavg_gt_oi_n19_20260923/group_results.csv')
for row in composite.itertuples():
 v=subject.query('target == @row.target and phase == @row.phase');val=.5*v.CA.to_numpy()+.5*v.JX.to_numpy()-v.OI.to_numpy();np.testing.assert_allclose(val.mean(),row.mean,atol=1e-13,rtol=0);checks+=1
result=dict(status='PASS',checks=checks,subjects=19,condition_rows=12,pairwise_rows=12,omnibus_rows=4,source_locked=True);(P/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
