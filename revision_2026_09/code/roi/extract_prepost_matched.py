#!/usr/bin/env python3
"""Source-locked M2 Pre 3-mm grid/physical spheres and Post physical sensitivity."""
import csv, hashlib, json
from pathlib import Path
import nibabel as nib
import numpy as np
HERE=Path(__file__).resolve().parent
GLM=HERE.parents[1]
M2=GLM/'roi_behavior_analysis/joint_m2_three_peaks_20260915'
SOURCE=GLM/'insight_event_timing_sensitivity/joint_search_pre_phase_contrasts_20260831'
POST=GLM/'roi_behavior_analysis'
POSTFILES={
 'angular':POST/'historical_post0p5_angular_3mm_rt_n19_20260923/participant_values.csv',
 'precuneus':POST/'prepost_peak_correlation_reporting_postprecuneus_20260924/post_precuneus_participant_values.csv',
}
CENTERS={'pre':{'angular':[-57,-57,17.7],'precuneus':[-6,-51,40.8]},'post':{'angular':[-48,-51,24.3],'precuneus':[0,-60,44.1]}}
ROIS=('angular','precuneus');CONDS=('CA','JX','OI')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
 assert rows
 with (HERE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
beh={(r['subject'],r['condition']):r for r in read(M2/'participant_behavior.csv')}
ids=sorted({s for s,c in beh});assert len(ids)==19 and len(beh)==57 and 'sub-005' not in ids
assert set(beh)=={(s,c) for s in ids for c in CONDS}
legacy={(r['subject'],r['roi'],r['condition']):r for r in read(M2/'participant_roi_betas.csv') if r['phase']=='pre' and r['roi'] in ('Angular Gyrus','Precuneus Cortex')}
maskpath=SOURCE/'first_level_masks/common_n19_intersection_mask.nii.gz';im=nib.load(maskpath);common=im.get_fdata()>0;ijk=np.argwhere(common);xyz=nib.affines.apply_affine(im.affine,ijk)
assert len(ijk)==48279 and np.allclose(nib.affines.voxel_sizes(im.affine),[3,3,3.3],atol=1e-6)
# Apply actual coordinate geometry before looking at any behavioral outcome.
membership={}
for roi,center in CENTERS['pre'].items():
 idx=np.rint(nib.affines.apply_affine(np.linalg.inv(im.affine),center)).astype(int)
 actual=nib.affines.apply_affine(im.affine,idx)
 assert np.linalg.norm(actual-center)<.001
 grid=np.linalg.norm((ijk-idx)*3.,axis=1)<=3.+1e-9
 physical=np.linalg.norm(xyz-np.array(center),axis=1)<=3.+1e-5
 old=np.linalg.norm(xyz-np.array(center),axis=1)<=6.0
 assert grid.sum()==7 and physical.sum()==5 and old.sum()==27
 assert not np.any(grid & ~old)
 membership[roi]={'grid':grid,'physical':physical,'old':old,'actual':actual.tolist(),'index':idx.tolist(),'grid_ijk':ijk[grid].tolist(),'physical_ijk':ijk[physical].tolist()}
pre=[];sourcehashes={str(maskpath):sha(maskpath),str(M2/'participant_behavior.csv'):sha(M2/'participant_behavior.csv'),str(M2/'participant_roi_betas.csv'):sha(M2/'participant_roi_betas.csv')}
for s in ids:
 cp=M2/'checkpoints'/f'{s}.npz';vpath=M2/'checkpoints'/f'{s}_verification.json';v=json.loads(vpath.read_text());assert sha(cp)==v['checkpoint_sha256'];sourcehashes[str(cp)]=sha(cp);sourcehashes[str(vpath)]=sha(vpath)
 with np.load(cp) as ck:
  union=ck['union_ijk'];theta=ck['theta'];cols=ck['columns'];a=ck['affine']
  np.testing.assert_allclose(a,im.affine,atol=1e-6)
  assert len(union)==theta.shape[1]==81 and len(cols)==theta.shape[0]
  assert len(set(map(tuple,union)))==81 and all(common[tuple(k)] for k in union)
  loc={tuple(v):k for k,v in enumerate(union)}
  for roi,label in [('angular','Angular Gyrus'),('precuneus','Precuneus Cortex')]:
   selected={typ:np.array([loc[tuple(v)] for v in membership[roi][f'{typ}_ijk']],dtype=int) for typ in ('grid','physical')}
   old_idx=np.array([loc[tuple(v)] for v in ijk[membership[roi]['old']]],dtype=int)
   for c in CONDS:
    j=list(cols).index('pre_'+c)
    old=float(theta[j,old_idx].mean());assert abs(old-float(legacy[s,label,c]['beta']))<1e-10
    pre.append(dict(subject=s,condition=c,roi=roi,model='M2_pre',peak_mni=json.dumps(CENTERS['pre'][roi]),actual_center_mni=json.dumps(membership[roi]['actual']),beta_grid7=float(theta[j,selected['grid']].mean()),beta_physical5=float(theta[j,selected['physical']].mean()),beta_old6mm=old,n_grid7=7,n_physical5=5,checkpoint_path=str(cp),checkpoint_sha256=sha(cp)))
   # Validate saved M2 contrast coefficients at all newly included ROI voxels.
   for aa,bb in [('CA','JX'),('CA','OI'),('JX','OI')]:
    con=SOURCE/'first_level'/s/f'{s}_pre_{aa}_gt_{bb}_effect.nii.gz';cm=nib.load(con)
    np.testing.assert_allclose(cm.affine,a,atol=1e-6)
    expected=theta[list(cols).index('pre_'+aa),selected['grid']]-theta[list(cols).index('pre_'+bb),selected['grid']]
    observed=np.array([cm.dataobj[tuple(v)] for v in membership[roi]['grid_ijk']]);np.testing.assert_allclose(observed,expected,atol=1e-6,rtol=1e-7)
    sourcehashes[str(con)]=sha(con)
assert len(pre)==114;save('pre_extractions.csv',pre)
post=[]
for roi,p in POSTFILES.items():
 sourcehashes[str(p)]=sha(p)
 archived={(r['subject'],r['condition']):r for r in read(p)};assert set(archived)==set(beh)
 for s in ids:
  for c in CONDS:
   x=archived[s,c];mp=Path(x['map_path']);assert sha(mp)==x['map_sha256'];sourcehashes[str(mp)]=sha(mp)
   img=nib.load(mp);aff=img.affine;data=img.get_fdata();ctr=CENTERS['post'][roi]
   idx=np.rint(nib.affines.apply_affine(np.linalg.inv(aff),ctr)).astype(int)
   pts=np.array([idx+[i,j,k] for i in range(-1,2) for j in range(-1,2) for k in range(-1,2)])
   grid=pts[np.linalg.norm((pts-idx)*3.,axis=1)<=3.+1e-9]
   phys=pts[np.linalg.norm((pts-idx)@aff[:3,:3].T,axis=1)<=3.+1e-9]
   assert len(grid)==7 and len(phys)==5 and len(grid)==int(x['n_voxels'])
   seven=float(np.mean([data[tuple(q)] for q in grid]));five=float(np.mean([data[tuple(q)] for q in phys]))
   assert abs(seven-float(x['beta_3mm_post']))<1e-10
   if 'beta_strict_physical_3mm_post' in x:assert abs(five-float(x['beta_strict_physical_3mm_post']))<1e-10
   post.append(dict(subject=s,condition=c,roi=roi,model='historical_post0p5',peak_mni=json.dumps(ctr),actual_center_mni=json.dumps(nib.affines.apply_affine(aff,idx).tolist()),beta_grid7=seven,beta_physical5=five,n_grid7=7,n_physical5=5,map_path=str(mp),map_sha256=sha(mp)))
assert len(post)==114;save('post_extractions.csv',post)
prov={'plan_sha256':sha(HERE/'PLAN.md'),'script_sha256':sha(Path(__file__)),'source_hashes':sourcehashes,'subjects':ids,'pre_mask':str(maskpath),'centers':CENTERS,'pre_roi_geometry':{r:{k:v for k,v in d.items() if k not in ('grid','physical','old')} for r,d in membership.items()},'n_pre_rows':len(pre),'n_post_rows':len(post)}
(HERE/'EXTRACTION_PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n')
print('Extracted',len(pre),'Pre 7/5 rows from hashed M2 checkpoint theta and',len(post),'Post 7/5 rows from hashed NIfTI maps; 6-mm M2 means and three contrasts validated per ROI and participant')
