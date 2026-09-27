#!/usr/bin/env python3
"""Re-index four N19 target×phase families without refitting."""
from pathlib import Path
import hashlib,json
import pandas as pd
from statsmodels.stats.multitest import multipletests
O=Path(__file__).resolve().parent; R=O.parent
C=R/'ppi_precuneus_angular_historicalpeak_condition_specific_n19_20260923'; E=R/'ppi_precuneus_angular_historicalpeak_cajxavg_gt_oi_n19_20260923'
P={'coefficients':C/'group_coefficients.csv','pairwise':C/'group_pairwise.csv','omnibus':C/'group_omnibus.csv','composite':E/'group_results.csv'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'FROZEN_PLAN.md').exists() and not (O/'ledger.csv').exists()
d={k:pd.read_csv(v) for k,v in P.items()}; records=[]
for target in ['angular_historical','ofc_left']:
 for phase in ['first','final']:
  x=d['coefficients'].query('target == @target and phase == @phase').set_index('condition').loc[['CA','JX','OI']];assert len(x)==3 and (x.n==19).all()
  p=d['pairwise'].query('target == @target and phase == @phase').set_index('contrast').loc[['CA_minus_JX','CA_minus_OI','JX_minus_OI']];assert len(p)==3 and (p.n==19).all()
  o=d['omnibus'].query('target == @target and phase == @phase');e=d['composite'].query('target == @target and phase == @phase');assert len(o)==len(e)==1 and int(o.iloc[0]['n'])==int(e.iloc[0]['n'])==19
  for table,kind in [(x,'coefficient_vs_zero'),(p,'pairwise_condition')]:
   raw=table.p_raw_exact_two_sided.to_numpy();adjusted=multipletests(raw,alpha=.05,method='holm')[1]
   for (name,row),adj in zip(table.iterrows(),adjusted):records.append(dict(target=target,phase=phase,test_kind=kind,contrast=name,n=19,raw_p=float(row.p_raw_exact_two_sided),within_cell_family_size=3,holm_within_cell=float(adj),prior_broad_family_adjusted=float(row.p_holm12)))
  row=o.iloc[0];records.append(dict(target=target,phase=phase,test_kind='two_dimensional_condition_omnibus',contrast='CA-JX and CA-OI',n=19,raw_p=float(row.p_raw_exact),within_cell_family_size=1,holm_within_cell=float(row.p_raw_exact),prior_broad_family_adjusted=float(row.p_holm4)))
  row=e.iloc[0];records.append(dict(target=target,phase=phase,test_kind='posthoc_one_sided_composite',contrast='(CA+JX)/2-OI',n=19,raw_p=float(row.p_directional),within_cell_family_size=1,holm_within_cell=float(row.p_directional),prior_broad_family_adjusted=float(row.holm4_directional)))
res=pd.DataFrame(records);assert len(res)==32
res.to_csv(O/'ledger.csv',index=False,float_format='%.17g')
(O/'source_hashes.json').write_text(json.dumps({'sources':{str(p):sha(p) for p in P.values()},'script_sha256':sha(Path(__file__)),'family_definition':'four independent reporting families by target × interval, coefficient Holm3 and pairwise Holm3 separately; omnibus and posthoc singleton per cell; broad original adjustments retained for sensitivity','posthoc_change':True},indent=2)+'\n')
print(res.to_string(index=False));print('PASS ledger rows',len(res))
