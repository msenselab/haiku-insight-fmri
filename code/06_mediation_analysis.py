#!/usr/bin/env python3
"""
Full-sample Mediation Analysis for Haiku fMRI Study
=====================================================
NOTE: This script requires outputs from 03_roi_analysis.py and
05_connectivity_analysis.py, which in turn require fMRIPrep derivatives.
Pre-computed mediation results are in:
  data/mediation/path_comparison.csv    (connectivity path coefficients, Table 5)
  data/mediation/mediation_results.csv  (bootstrap mediation, Mediation section)

Model: Condition → DMN (PCC) → Reward (vmPFC)
Question: Does DMN mediate the condition effect on reward signaling?

Using bootstrapped mediation for proper inference.
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
CONN_DIR = BASE_DIR / 'data' / 'connectivity'
OUT_DIR = BASE_DIR / 'data' / 'mediation' / 'recomputed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("HAIKU fMRI: Full-Sample Mediation Analysis")
print("="*70)

# Load condition-specific connectivity data
df = pd.read_csv(CONN_DIR / 'condition_specific_connectivity_legacy_subject_values.csv')
print(f"\nLoaded connectivity data: {len(df)} rows")

# Get unique subjects
subjects = df['subject'].unique()
print(f"Number of subjects: {len(subjects)}")
print(f"Subjects: {', '.join(sorted(subjects))}")

# Conditions
conditions = ['CA', 'JX', 'OI']

print("\n" + "="*70)
print("1. Extracting connectivity data for mediation paths")
print("="*70)

# Build subject-level dataframe with connectivity measures per condition
data_list = []
for sub in subjects:
    for cond in conditions:
        sub_df = df[(df['subject'] == sub) & (df['condition'] == cond)]
        
        # Extract key connectivity values
        pcc_vmpfc = sub_df[((sub_df['roi1'] == 'PCC') & (sub_df['roi2'] == 'vmPFC')) | 
                          ((sub_df['roi1'] == 'vmPFC') & (sub_df['roi2'] == 'PCC'))]['r'].values
        atl_angular = sub_df[((sub_df['roi1'] == 'ATL') & (sub_df['roi2'] == 'L_Angular')) | 
                            ((sub_df['roi1'] == 'L_Angular') & (sub_df['roi2'] == 'ATL'))]['r'].values
        pcc_angular = sub_df[((sub_df['roi1'] == 'PCC') & (sub_df['roi2'] == 'L_Angular')) | 
                            ((sub_df['roi1'] == 'L_Angular') & (sub_df['roi2'] == 'PCC'))]['r'].values
        atl_pcc = sub_df[((sub_df['roi1'] == 'ATL') & (sub_df['roi2'] == 'PCC')) | 
                        ((sub_df['roi1'] == 'PCC') & (sub_df['roi2'] == 'ATL'))]['r'].values
        atl_vmpfc = sub_df[((sub_df['roi1'] == 'ATL') & (sub_df['roi2'] == 'vmPFC')) | 
                          ((sub_df['roi1'] == 'vmPFC') & (sub_df['roi2'] == 'ATL'))]['r'].values
        angular_vmpfc = sub_df[((sub_df['roi1'] == 'L_Angular') & (sub_df['roi2'] == 'vmPFC')) | 
                              ((sub_df['roi1'] == 'vmPFC') & (sub_df['roi2'] == 'L_Angular'))]['r'].values
        
        data_list.append({
            'subject': sub,
            'condition': cond,
            'PCC_vmPFC': pcc_vmpfc[0] if len(pcc_vmpfc) > 0 else np.nan,
            'ATL_Angular': atl_angular[0] if len(atl_angular) > 0 else np.nan,
            'PCC_Angular': pcc_angular[0] if len(pcc_angular) > 0 else np.nan,
            'ATL_PCC': atl_pcc[0] if len(atl_pcc) > 0 else np.nan,
            'ATL_vmPFC': atl_vmpfc[0] if len(atl_vmpfc) > 0 else np.nan,
            'Angular_vmPFC': angular_vmpfc[0] if len(angular_vmpfc) > 0 else np.nan,
        })

conn_df = pd.DataFrame(data_list)
print(f"\nConnectivity dataframe: {conn_df.shape}")
print(f"Missing values:\n{conn_df.isnull().sum()}")

# Drop any subjects with missing data
conn_df = conn_df.dropna()
final_subjects = conn_df['subject'].unique()
n_subjects = len(final_subjects)
print(f"\nFinal sample size: n={n_subjects}")

print("\n" + "="*70)
print("2. Path Analysis: Condition Differences in Connectivity")
print("="*70)

# Key paths to analyze
paths = ['PCC_vmPFC', 'ATL_Angular', 'PCC_Angular', 'ATL_PCC', 'ATL_vmPFC', 'Angular_vmPFC']
path_labels = {
    'PCC_vmPFC': 'Self-Ref → Reward',
    'ATL_Angular': 'Meaning → Integration',
    'PCC_Angular': 'Self-Ref → Integration',
    'ATL_PCC': 'Meaning → Self-Ref',
    'ATL_vmPFC': 'Meaning → Reward',
    'Angular_vmPFC': 'Integration → Reward'
}

# Compute condition means and test differences
results = []
print(f"\nPath coefficients by condition (n={n_subjects}):")
print("-" * 70)

for path in paths:
    ca_vals = conn_df[conn_df['condition'] == 'CA'][path].values
    jx_vals = conn_df[conn_df['condition'] == 'JX'][path].values
    oi_vals = conn_df[conn_df['condition'] == 'OI'][path].values
    
    # Repeated-measures ANOVA (within-subject)
    f_stat, p_anova = stats.f_oneway(ca_vals, jx_vals, oi_vals)
    
    # Paired t-tests for specific comparisons
    t_ca_jx, p_ca_jx = stats.ttest_rel(ca_vals, jx_vals)
    t_ca_oi, p_ca_oi = stats.ttest_rel(ca_vals, oi_vals)
    t_jx_oi, p_jx_oi = stats.ttest_rel(jx_vals, oi_vals)
    
    # Effect sizes (Cohen's d for paired samples)
    d_ca_jx = (np.mean(ca_vals) - np.mean(jx_vals)) / np.std(ca_vals - jx_vals)
    d_ca_oi = (np.mean(ca_vals) - np.mean(oi_vals)) / np.std(ca_vals - oi_vals)
    
    results.append({
        'path': path,
        'label': path_labels[path],
        'CA_mean': np.mean(ca_vals),
        'CA_se': stats.sem(ca_vals),
        'JX_mean': np.mean(jx_vals),
        'JX_se': stats.sem(jx_vals),
        'OI_mean': np.mean(oi_vals),
        'OI_se': stats.sem(oi_vals),
        'F': f_stat,
        'p_anova': p_anova,
        't_CA_JX': t_ca_jx,
        'p_CA_JX': p_ca_jx,
        'd_CA_JX': d_ca_jx,
        't_CA_OI': t_ca_oi,
        'p_CA_OI': p_ca_oi,
        'd_CA_OI': d_ca_oi,
        't_JX_OI': t_jx_oi,
        'p_JX_OI': p_jx_oi
    })
    
    sig_marker = '*' if p_anova < 0.05 else ''
    print(f"{path_labels[path]:25} | CA: {np.mean(ca_vals):.3f}±{stats.sem(ca_vals):.3f} | "
          f"JX: {np.mean(jx_vals):.3f}±{stats.sem(jx_vals):.3f} | "
          f"OI: {np.mean(oi_vals):.3f}±{stats.sem(oi_vals):.3f} | F={f_stat:.2f}, p={p_anova:.4f}{sig_marker}")

results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUT_DIR}/path_comparison.csv', index=False)
print(f"\nSaved path comparison to: {OUT_DIR}/path_comparison.csv")

print("\n" + "="*70)
print("3. Bootstrapped Mediation Analysis")
print("="*70)
print("Model: Condition → DMN (PCC) → Reward (vmPFC)")
print("Testing if PCC mediates condition effect on vmPFC coupling")

def bootstrap_mediation(X, M, Y, n_boot=5000, alpha=0.05):
    """Bootstrap mediation analysis"""
    n = len(X)
    
    # Calculate observed effects
    a = np.polyfit(X, M, 1)[0]
    X_mat = np.column_stack([np.ones(n), X, M])
    betas = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    c_prime = betas[1]
    b = betas[2]
    c = np.polyfit(X, Y, 1)[0]
    indirect = a * b
    
    # Bootstrap for confidence intervals
    boot_indirect = []
    boot_a = []
    boot_b = []
    boot_c = []
    boot_c_prime = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_b, M_b, Y_b = X[idx], M[idx], Y[idx]
        
        a_b = np.polyfit(X_b, M_b, 1)[0]
        boot_a.append(a_b)
        
        X_mat_b = np.column_stack([np.ones(n), X_b, M_b])
        try:
            betas_b = np.linalg.lstsq(X_mat_b, Y_b, rcond=None)[0]
            c_prime_b = betas_b[1]
            b_b = betas_b[2]
        except:
            continue
        boot_b.append(b_b)
        boot_c_prime.append(c_prime_b)
        
        c_b = np.polyfit(X_b, Y_b, 1)[0]
        boot_c.append(c_b)
        boot_indirect.append(a_b * b_b)
    
    ci_low = alpha / 2
    ci_high = 1 - alpha / 2
    
    return {
        'a': a,
        'a_ci': (np.percentile(boot_a, ci_low*100), np.percentile(boot_a, ci_high*100)),
        'b': b,
        'b_ci': (np.percentile(boot_b, ci_low*100), np.percentile(boot_b, ci_high*100)),
        'c': c,
        'c_ci': (np.percentile(boot_c, ci_low*100), np.percentile(boot_c, ci_high*100)),
        'c_prime': c_prime,
        'c_prime_ci': (np.percentile(boot_c_prime, ci_low*100), np.percentile(boot_c_prime, ci_high*100)),
        'indirect': indirect,
        'indirect_ci': (np.percentile(boot_indirect, ci_low*100), np.percentile(boot_indirect, ci_high*100)),
        'prop_mediated': indirect / c if abs(c) > 0.001 else 0,
        'n_boot': n_boot
    }

mediation_results = []

# Test 1: Does semantic-DMN connectivity mediate condition effect on reward?
print("\n--- Mediation: ATL-PCC → vmPFC coupling ---")
print("Testing if semantic→DMN pathway explains condition differences in reward")

for cond1, cond2 in [('CA', 'JX'), ('CA', 'OI'), ('JX', 'OI')]:
    sub_df = conn_df[conn_df['condition'].isin([cond1, cond2])].copy()
    sub_df['X'] = (sub_df['condition'] == cond1).astype(int)
    
    X = sub_df['X'].values
    M = sub_df['ATL_PCC'].values
    Y = sub_df['PCC_vmPFC'].values
    
    result = bootstrap_mediation(X, M, Y, n_boot=5000)
    indirect_sig = not (result['indirect_ci'][0] <= 0 <= result['indirect_ci'][1])
    
    print(f"\n{cond1} vs {cond2}:")
    print(f"  Path a (Condition→ATL-PCC):    {result['a']:.4f} [{result['a_ci'][0]:.4f}, {result['a_ci'][1]:.4f}]")
    print(f"  Path b (ATL-PCC→PCC-vmPFC):    {result['b']:.4f} [{result['b_ci'][0]:.4f}, {result['b_ci'][1]:.4f}]")
    print(f"  Path c (Total effect):         {result['c']:.4f} [{result['c_ci'][0]:.4f}, {result['c_ci'][1]:.4f}]")
    print(f"  Path c' (Direct effect):       {result['c_prime']:.4f} [{result['c_prime_ci'][0]:.4f}, {result['c_prime_ci'][1]:.4f}]")
    print(f"  Indirect effect (a×b):         {result['indirect']:.4f} [{result['indirect_ci'][0]:.4f}, {result['indirect_ci'][1]:.4f}]")
    print(f"  Proportion mediated:           {result['prop_mediated']:.2%}")
    print(f"  Mediation significant (95% CI excludes 0): {'YES' if indirect_sig else 'NO'}")
    
    mediation_results.append({
        'comparison': f'{cond1}_vs_{cond2}',
        'mediator': 'ATL_PCC',
        'outcome': 'PCC_vmPFC',
        'n': n_subjects,
        'a_path': result['a'],
        'a_ci_low': result['a_ci'][0],
        'a_ci_high': result['a_ci'][1],
        'b_path': result['b'],
        'b_ci_low': result['b_ci'][0],
        'b_ci_high': result['b_ci'][1],
        'c_total': result['c'],
        'c_ci_low': result['c_ci'][0],
        'c_ci_high': result['c_ci'][1],
        'c_prime_direct': result['c_prime'],
        'c_prime_ci_low': result['c_prime_ci'][0],
        'c_prime_ci_high': result['c_prime_ci'][1],
        'indirect': result['indirect'],
        'indirect_ci_low': result['indirect_ci'][0],
        'indirect_ci_high': result['indirect_ci'][1],
        'prop_mediated': result['prop_mediated'],
        'significant': indirect_sig
    })

# Test 2: Does PCC-Angular connectivity mediate condition effect on reward?
print("\n--- Mediation: PCC-Angular → vmPFC coupling ---")
print("Testing if DMN integration explains condition differences in reward")

for cond1, cond2 in [('CA', 'JX'), ('CA', 'OI'), ('JX', 'OI')]:
    sub_df = conn_df[conn_df['condition'].isin([cond1, cond2])].copy()
    sub_df['X'] = (sub_df['condition'] == cond1).astype(int)
    
    X = sub_df['X'].values
    M = sub_df['PCC_Angular'].values
    Y = sub_df['PCC_vmPFC'].values
    
    result = bootstrap_mediation(X, M, Y, n_boot=5000)
    indirect_sig = not (result['indirect_ci'][0] <= 0 <= result['indirect_ci'][1])
    
    print(f"\n{cond1} vs {cond2}:")
    print(f"  Indirect effect:               {result['indirect']:.4f} [{result['indirect_ci'][0]:.4f}, {result['indirect_ci'][1]:.4f}]")
    print(f"  Mediation significant: {'YES' if indirect_sig else 'NO'}")
    
    mediation_results.append({
        'comparison': f'{cond1}_vs_{cond2}',
        'mediator': 'PCC_Angular',
        'outcome': 'PCC_vmPFC',
        'n': n_subjects,
        'a_path': result['a'],
        'a_ci_low': result['a_ci'][0],
        'a_ci_high': result['a_ci'][1],
        'b_path': result['b'],
        'b_ci_low': result['b_ci'][0],
        'b_ci_high': result['b_ci'][1],
        'c_total': result['c'],
        'c_ci_low': result['c_ci'][0],
        'c_ci_high': result['c_ci'][1],
        'c_prime_direct': result['c_prime'],
        'c_prime_ci_low': result['c_prime_ci'][0],
        'c_prime_ci_high': result['c_prime_ci'][1],
        'indirect': result['indirect'],
        'indirect_ci_low': result['indirect_ci'][0],
        'indirect_ci_high': result['indirect_ci'][1],
        'prop_mediated': result['prop_mediated'],
        'significant': indirect_sig
    })

# Save mediation results
med_df = pd.DataFrame(mediation_results)
med_df.to_csv(f'{OUT_DIR}/mediation_results.csv', index=False)
print(f"\nSaved mediation results to: {OUT_DIR}/mediation_results.csv")

print("\n" + "="*70)
print("4. Summary of Key Findings")
print("="*70)

# Summarize path comparison
print("\n--- Path Differences by Condition ---")
sig_paths = results_df[results_df['p_anova'] < 0.05]
if len(sig_paths) > 0:
    print("Significant paths (p < 0.05):")
    for _, row in sig_paths.iterrows():
        print(f"  - {row['label']}: F={row['F']:.2f}, p={row['p_anova']:.4f}")
else:
    print("No paths showed significant condition differences (p < 0.05)")

# Look at near-significant and effect sizes for PCC-vmPFC
print("\n--- Critical Path: PCC->vmPFC (Self-Ref->Reward) ---")
pcc_vmpfc = results_df[results_df['path'] == 'PCC_vmPFC'].iloc[0]
print(f"  CA: {pcc_vmpfc['CA_mean']:.3f} +/- {pcc_vmpfc['CA_se']:.3f}")
print(f"  JX: {pcc_vmpfc['JX_mean']:.3f} +/- {pcc_vmpfc['JX_se']:.3f}")
print(f"  OI: {pcc_vmpfc['OI_mean']:.3f} +/- {pcc_vmpfc['OI_se']:.3f}")
print(f"  F = {pcc_vmpfc['F']:.2f}, p = {pcc_vmpfc['p_anova']:.4f}")
print(f"  CA vs JX: t={pcc_vmpfc['t_CA_JX']:.2f}, p={pcc_vmpfc['p_CA_JX']:.4f}, d={pcc_vmpfc['d_CA_JX']:.2f}")
print(f"  CA vs OI: t={pcc_vmpfc['t_CA_OI']:.2f}, p={pcc_vmpfc['p_CA_OI']:.4f}, d={pcc_vmpfc['d_CA_OI']:.2f}")

# Summarize mediation
print("\n--- Mediation Results ---")
sig_med = med_df[med_df['significant'] == True]
if len(sig_med) > 0:
    print("Significant mediation effects:")
    for _, row in sig_med.iterrows():
        print(f"  - {row['comparison']}, {row['mediator']}->{row['outcome']}: "
              f"indirect={row['indirect']:.4f} [{row['indirect_ci_low']:.4f}, {row['indirect_ci_high']:.4f}]")
else:
    print("No significant mediation effects detected with 95% bootstrap CI")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
