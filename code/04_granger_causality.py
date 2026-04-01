#!/usr/bin/env python3
"""
Granger Causality Analysis - Full Sample (Efficient Version)

NOTE: This script requires fMRIPrep preprocessed BOLD timeseries
(derivatives/fmriprep/) which are not included in this Tier 1 release.
Pre-computed outputs reproducing Table 4 are in:
  data/granger/granger_group_final.csv   (group summary - matches Table 4)
  data/granger/granger_all_subjects.csv  (per-subject F-statistics)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from nilearn.maskers import NiftiSpheresMasker
from scipy import stats

# Check statsmodels
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except:
    HAS_STATSMODELS = False
    print("statsmodels not available, using simple F-test approach")

# Paths
fmriprep_dir = Path("/dss/studies/fmri-haiku/derivatives/fmriprep")
output_dir = Path("/dss/studies/fmri-haiku/glm_unified/granger_analysis_full")
output_dir.mkdir(exist_ok=True)

# ROIs
ROIS = {
    'PCC': (0, -52, 26),
    'vmPFC': (0, 46, -10),
    'L_Angular': (-48, -64, 30),
    'ATL': (-54, -6, -22),
}

EXCLUDE = ['sub-005', 'sub-015', 'sub-017', 'sub-022']

def simple_granger(x, y, max_lag=4):
    """Simple Granger-like causality using lagged regression"""
    n = len(x)
    if n < max_lag + 10:
        return np.nan, np.nan
    
    # Build lagged matrices
    y_target = y[max_lag:]
    
    # Restricted model: y ~ past y only
    X_r = np.column_stack([y[max_lag-i-1:n-i-1] for i in range(max_lag)])
    
    # Unrestricted model: y ~ past y + past x
    X_ur = np.column_stack([X_r] + [x[max_lag-i-1:n-i-1] for i in range(max_lag)])
    
    # Add intercept
    X_r = np.column_stack([np.ones(len(y_target)), X_r])
    X_ur = np.column_stack([np.ones(len(y_target)), X_ur])
    
    # Fit models
    try:
        beta_r = np.linalg.lstsq(X_r, y_target, rcond=None)[0]
        beta_ur = np.linalg.lstsq(X_ur, y_target, rcond=None)[0]
        
        # Compute RSS
        rss_r = np.sum((y_target - X_r @ beta_r)**2)
        rss_ur = np.sum((y_target - X_ur @ beta_ur)**2)
        
        # F-test
        df1 = max_lag  # added parameters
        df2 = len(y_target) - X_ur.shape[1]
        
        if rss_ur == 0:
            return np.nan, np.nan
            
        f_stat = ((rss_r - rss_ur) / df1) / (rss_ur / df2)
        p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return f_stat, p_val
    except:
        return np.nan, np.nan

print("=" * 60)
print("Granger Causality Analysis - Full Sample")
print("=" * 60)

# Get subjects
subjects = sorted([d.name for d in fmriprep_dir.glob("sub-*") 
                   if d.is_dir() and d.name not in EXCLUDE])
print(f"\nProcessing {len(subjects)} subjects...")

# Paths to test
paths = [
    ('PCC', 'vmPFC'),
    ('vmPFC', 'PCC'),
    ('L_Angular', 'PCC'),
    ('ATL', 'vmPFC'),
]

results = []

# Process subjects
for sub in subjects:
    bold_file = fmriprep_dir / sub / "func" / f"{sub}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    
    if not bold_file.exists():
        print(f"  {sub}: No BOLD data")
        continue
    
    print(f"  {sub}...", end=" ")
    
    try:
        # Extract timeseries
        coords = list(ROIS.values())
        masker = NiftiSpheresMasker(
            seeds=coords,
            radius=6,
            standardize=True,
            detrend=True,
            t_r=2.0,
            low_pass=0.1,
            high_pass=0.01
        )
        
        ts = masker.fit_transform(str(bold_file))
        ts_df = pd.DataFrame(ts, columns=list(ROIS.keys()))
        
        # Compute Granger for each path
        for source, target in paths:
            if HAS_STATSMODELS:
                try:
                    data = np.column_stack([ts_df[target].values, ts_df[source].values])
                    gc_result = grangercausalitytests(data, maxlag=4, verbose=False)
                    f_stat = gc_result[2][0]['ssr_ftest'][0]  # lag 2
                    p_val = gc_result[2][0]['ssr_ftest'][1]
                except:
                    f_stat, p_val = simple_granger(ts_df[source].values, ts_df[target].values)
            else:
                f_stat, p_val = simple_granger(ts_df[source].values, ts_df[target].values)
            
            results.append({
                'subject': sub,
                'source': source,
                'target': target,
                'path': f'{source}→{target}',
                'f_stat': f_stat,
                'p_val': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            })
        
        print("OK")
        
    except Exception as e:
        print(f"Error: {e}")
        continue

# Convert to DataFrame
df = pd.DataFrame(results)
df.to_csv(output_dir / "granger_all_subjects.csv", index=False)

# Group statistics
print("\n" + "=" * 60)
print("GROUP RESULTS")
print("=" * 60)

group_stats = []
for path in df['path'].unique():
    path_data = df[df['path'] == path]
    n = len(path_data)
    f_vals = path_data['f_stat'].dropna()
    
    if len(f_vals) >= 3:
        t_stat, t_pval = stats.ttest_1samp(f_vals, 0)
    else:
        t_stat, t_pval = np.nan, np.nan
    
    group_stats.append({
        'path': path,
        'n': n,
        'mean_F': f_vals.mean(),
        'std_F': f_vals.std(),
        'n_sig': path_data['significant'].sum(),
        'pct_sig': (path_data['significant'].sum() / n * 100) if n > 0 else 0,
        't_stat': t_stat,
        'p_value': t_pval
    })
    
    sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else ""
    print(f"\n{path}: N={n}, Mean F={f_vals.mean():.2f}±{f_vals.std():.2f}, t={t_stat:.2f}, p={t_pval:.4f} {sig}")

group_df = pd.DataFrame(group_stats)
group_df.to_csv(output_dir / "granger_group_stats.csv", index=False)

# Net Granger (PCC→vmPFC vs vmPFC→PCC)
print("\n" + "=" * 60)
print("NET GRANGER CAUSALITY")
print("=" * 60)

pcc_vmpfc = df[df['path'] == 'PCC→vmPFC'].set_index('subject')['f_stat']
vmpfc_pcc = df[df['path'] == 'vmPFC→PCC'].set_index('subject')['f_stat']

common = pcc_vmpfc.index.intersection(vmpfc_pcc.index)
net_gc = pcc_vmpfc[common] - vmpfc_pcc[common]

if len(net_gc.dropna()) >= 3:
    t_net, p_net = stats.ttest_1samp(net_gc.dropna(), 0)
    direction = "PCC → vmPFC (DMN drives Reward)" if net_gc.mean() > 0 else "vmPFC → PCC (Reward drives DMN)"
    print(f"\nNet GC (PCC→vmPFC minus vmPFC→PCC):")
    print(f"  Mean = {net_gc.mean():.2f} ± {net_gc.std():.2f}")
    print(f"  t({len(net_gc.dropna())-1}) = {t_net:.2f}, p = {p_net:.4f}")
    print(f"  Dominant direction: {direction}")

# Save report
report = f"""# Granger Causality Analysis - Full Sample

**Date:** 2026-01-31  
**Subjects:** N = {len(subjects)}  
**ROIs:** PCC, vmPFC, L_Angular, ATL  
**Max Lag:** 4 TRs

## Group Results

| Path | N | Mean F | Std F | % Sig | t-stat | p-value |
|------|---|--------|-------|-------|--------|---------|
"""

for _, row in group_df.iterrows():
    sig = "**" if row['p_value'] < 0.05 else ""
    report += f"| {row['path']} | {row['n']} | {row['mean_F']:.2f} | {row['std_F']:.2f} | {row['pct_sig']:.1f}% | {row['t_stat']:.2f} | {row['p_value']:.4f}{sig} |\n"

report += f"""
## Net Granger Causality

**PCC ↔ vmPFC:**
- Net GC = {net_gc.mean():.2f} ± {net_gc.std():.2f}
- t({len(net_gc.dropna())-1}) = {t_net:.2f}, p = {p_net:.4f}
- **Direction:** {direction}

## Interpretation

"""

# Add interpretation based on results
sig_paths = group_df[group_df['p_value'] < 0.05]
if len(sig_paths) > 0:
    report += "### Significant Paths:\n"
    for _, row in sig_paths.iterrows():
        report += f"- **{row['path']}**: F = {row['mean_F']:.2f}, p = {row['p_value']:.4f}\n"
else:
    report += "No individual paths reached group-level significance.\n"

with open(output_dir / "GRANGER_FULL_REPORT.md", 'w') as f:
    f.write(report)

print(f"\n\nResults saved to: {output_dir}")
