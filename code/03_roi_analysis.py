#!/usr/bin/env python3
"""
ROI Analysis for Haiku Insight Study

NOTE: This script requires access to first-level GLM beta maps
(glm_unified/first_level/) which are not included in this Tier 1 release
due to file size. Pre-computed outputs are available in:
  data/roi/roi_betas_individual.csv  (per-subject beta estimates)
  data/roi/roi_statistics_full.csv   (group statistics, reproduces Table 2)

Extracts signal from key regions across networks:
- Reward: NAcc, vmPFC
- DMN: PCC, Angular Gyrus
- Semantic: Temporal regions, IFG
- Visual: Occipital cortex

Compares CA vs JX vs OI during Search and Insight
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image, masking
from nilearn.maskers import NiftiSpheresMasker
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('/dss/studies/fmri-haiku')
UNIFIED_DIR = BASE_DIR / 'glm_unified'
FIRST_LEVEL_DIR = UNIFIED_DIR / 'first_level'
OUTPUT_DIR = UNIFIED_DIR / 'roi_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

# Exclude outlier
SUBJECTS = [f'sub-{i:03d}' for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23]]

# Define ROIs with MNI coordinates (x, y, z) and sphere radius
ROIS = {
    # Reward Network
    'L_NAcc': {'coords': (-10, 12, -6), 'radius': 6, 'network': 'Reward'},
    'R_NAcc': {'coords': (10, 12, -6), 'radius': 6, 'network': 'Reward'},
    'vmPFC': {'coords': (0, 44, -8), 'radius': 8, 'network': 'Reward'},
    
    # DMN
    'PCC': {'coords': (0, -52, 26), 'radius': 8, 'network': 'DMN'},
    'Precuneus': {'coords': (0, -60, 40), 'radius': 8, 'network': 'DMN'},
    'L_Angular': {'coords': (-46, -66, 30), 'radius': 8, 'network': 'DMN'},
    'R_Angular': {'coords': (48, -64, 30), 'radius': 8, 'network': 'DMN'},
    
    # Semantic/Control
    'L_IFG': {'coords': (-48, 24, 12), 'radius': 8, 'network': 'Semantic'},
    'R_IFG': {'coords': (48, 24, 12), 'radius': 8, 'network': 'Semantic'},
    'L_Temporal': {'coords': (-56, -42, -8), 'radius': 8, 'network': 'Semantic'},
    'R_Temporal': {'coords': (56, -42, -8), 'radius': 8, 'network': 'Semantic'},
    
    # Visual
    'L_Visual': {'coords': (-24, -90, -4), 'radius': 10, 'network': 'Visual'},
    'R_Visual': {'coords': (24, -90, -4), 'radius': 10, 'network': 'Visual'},
    'Calcarine': {'coords': (0, -84, 4), 'radius': 10, 'network': 'Visual'},
}

# Contrasts to extract
CONTRASTS = {
    'search': ['search_CA', 'search_JX', 'search_OI'],
    'insight': ['insight_CA', 'insight_JX', 'insight_OI'],
}

print("="*70)
print("ROI ANALYSIS: Haiku Types × Networks × Phase")
print("="*70)
print(f"Subjects: N = {len(SUBJECTS)}")
print(f"ROIs: {len(ROIS)}")

# Extract ROI values
all_data = []

for roi_name, roi_info in ROIS.items():
    print(f"\nExtracting: {roi_name} ({roi_info['network']})")
    
    masker = NiftiSpheresMasker(
        seeds=[roi_info['coords']],
        radius=roi_info['radius'],
        standardize=False
    )
    
    for phase, contrast_list in CONTRASTS.items():
        for contrast in contrast_list:
            condition = contrast.split('_')[1]  # CA, JX, or OI
            
            for sub in SUBJECTS:
                effect_file = FIRST_LEVEL_DIR / sub / f'{sub}_{contrast}_effect.nii.gz'
                
                if effect_file.exists():
                    try:
                        effect_img = image.load_img(str(effect_file))
                        value = masker.fit_transform(effect_img)[0, 0]
                        
                        all_data.append({
                            'subject': sub,
                            'roi': roi_name,
                            'network': roi_info['network'],
                            'phase': phase,
                            'condition': condition,
                            'beta': value
                        })
                    except Exception as e:
                        pass

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_DIR / 'roi_data.csv', index=False)
print(f"\nExtracted {len(df)} data points")

# Statistical Analysis
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

results = []

for network in ['Reward', 'DMN', 'Semantic', 'Visual']:
    print(f"\n{'='*50}")
    print(f"Network: {network}")
    print('='*50)
    
    network_rois = [r for r, info in ROIS.items() if info['network'] == network]
    
    for phase in ['search', 'insight']:
        print(f"\n--- {phase.upper()} ---")
        
        # Average across ROIs in network
        net_df = df[(df['network'] == network) & (df['phase'] == phase)]
        
        # Pivot for analysis
        pivot = net_df.groupby(['subject', 'condition'])['beta'].mean().unstack()
        
        if len(pivot.columns) < 3:
            continue
        
        ca = pivot['CA'].dropna().values
        jx = pivot['JX'].dropna().values
        oi = pivot['OI'].dropna().values
        
        # Get common subjects
        n = min(len(ca), len(jx), len(oi))
        ca, jx, oi = ca[:n], jx[:n], oi[:n]
        
        # One-way ANOVA
        f_stat, p_anova = stats.f_oneway(ca, jx, oi)
        print(f"  ANOVA (CA vs JX vs OI): F={f_stat:.2f}, p={p_anova:.4f}")
        
        # Pairwise t-tests
        t_ca_jx, p_ca_jx = stats.ttest_rel(ca, jx)
        t_ca_oi, p_ca_oi = stats.ttest_rel(ca, oi)
        t_jx_oi, p_jx_oi = stats.ttest_rel(jx, oi)
        
        # Two-image vs One-image
        two_img = (ca + jx) / 2
        t_cut, p_cut = stats.ttest_rel(two_img, oi)
        
        print(f"  CA vs JX: t={t_ca_jx:.2f}, p={p_ca_jx:.4f} {'*' if p_ca_jx < 0.05 else ''}")
        print(f"  CA vs OI: t={t_ca_oi:.2f}, p={p_ca_oi:.4f} {'*' if p_ca_oi < 0.05 else ''}")
        print(f"  JX vs OI: t={t_jx_oi:.2f}, p={p_jx_oi:.4f} {'*' if p_jx_oi < 0.05 else ''}")
        print(f"  TwoImg vs OneImg (Cut): t={t_cut:.2f}, p={p_cut:.4f} {'*' if p_cut < 0.05 else ''}")
        
        results.append({
            'network': network,
            'phase': phase,
            'mean_CA': ca.mean(),
            'mean_JX': jx.mean(),
            'mean_OI': oi.mean(),
            'F_anova': f_stat,
            'p_anova': p_anova,
            't_CA_JX': t_ca_jx,
            'p_CA_JX': p_ca_jx,
            't_cut': t_cut,
            'p_cut': p_cut
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'roi_statistics.csv', index=False)

# Create Figures
print("\n" + "="*70)
print("CREATING FIGURES")
print("="*70)

# Figure 1: Bar plots by network and phase
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, network in enumerate(['Reward', 'DMN', 'Semantic', 'Visual']):
    for j, phase in enumerate(['search', 'insight']):
        ax = axes[j, i]
        
        net_df = df[(df['network'] == network) & (df['phase'] == phase)]
        means = net_df.groupby('condition')['beta'].mean()
        sems = net_df.groupby('condition')['beta'].sem()
        
        conditions = ['CA', 'JX', 'OI']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        x = np.arange(len(conditions))
        bars = ax.bar(x, [means.get(c, 0) for c in conditions], 
                      yerr=[sems.get(c, 0) for c in conditions],
                      color=colors, capsize=5, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel('Beta (effect size)')
        ax.set_title(f'{network}\n({phase})')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add significance markers
        r = results_df[(results_df['network'] == network) & (results_df['phase'] == phase)]
        if len(r) > 0:
            if r['p_CA_JX'].values[0] < 0.05:
                ax.annotate('*', xy=(0.5, max(means.values()) * 1.1), fontsize=16, ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roi_barplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roi_barplots.png")

# Figure 2: Network comparison (Search vs Insight)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, network in enumerate(['Reward', 'DMN', 'Semantic', 'Visual']):
    ax = axes[i]
    
    for phase, color, offset in [('search', '#3498db', -0.15), ('insight', '#e74c3c', 0.15)]:
        net_df = df[(df['network'] == network) & (df['phase'] == phase)]
        means = net_df.groupby('condition')['beta'].mean()
        sems = net_df.groupby('condition')['beta'].sem()
        
        conditions = ['CA', 'JX', 'OI']
        x = np.arange(len(conditions)) + offset
        
        ax.bar(x, [means.get(c, 0) for c in conditions], 
               yerr=[sems.get(c, 0) for c in conditions],
               width=0.3, color=color, capsize=3, alpha=0.8, label=phase)
    
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Beta')
    ax.set_title(network)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roi_search_vs_insight.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roi_search_vs_insight.png")

# Figure 3: Cut Effect (TwoImage - OneImage) by network
fig, ax = plt.subplots(figsize=(10, 5))

cut_effects = []
for network in ['Reward', 'DMN', 'Semantic', 'Visual']:
    for phase in ['search', 'insight']:
        net_df = df[(df['network'] == network) & (df['phase'] == phase)]
        pivot = net_df.groupby(['subject', 'condition'])['beta'].mean().unstack()
        
        if 'CA' in pivot.columns and 'JX' in pivot.columns and 'OI' in pivot.columns:
            two_img = (pivot['CA'] + pivot['JX']) / 2
            cut = two_img - pivot['OI']
            cut_effects.append({
                'network': network,
                'phase': phase,
                'cut_effect': cut.mean(),
                'cut_sem': cut.sem()
            })

cut_df = pd.DataFrame(cut_effects)

x = np.arange(4)
width = 0.35

search_vals = cut_df[cut_df['phase'] == 'search']['cut_effect'].values
search_sems = cut_df[cut_df['phase'] == 'search']['cut_sem'].values
insight_vals = cut_df[cut_df['phase'] == 'insight']['cut_effect'].values
insight_sems = cut_df[cut_df['phase'] == 'insight']['cut_sem'].values

ax.bar(x - width/2, search_vals, width, yerr=search_sems, label='Search', color='#3498db', capsize=5)
ax.bar(x + width/2, insight_vals, width, yerr=insight_sems, label='Insight', color='#e74c3c', capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(['Reward', 'DMN', 'Semantic', 'Visual'])
ax.set_ylabel('Cut Effect (TwoImg - OneImg)')
ax.set_title('Cut Effect by Network and Phase')
ax.axhline(y=0, color='gray', linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roi_cut_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roi_cut_effect.png")

# Summary
print("\n" + "="*70)
print("SUMMARY OF SIGNIFICANT EFFECTS (p < 0.05)")
print("="*70)

sig_results = results_df[
    (results_df['p_anova'] < 0.05) | 
    (results_df['p_CA_JX'] < 0.05) | 
    (results_df['p_cut'] < 0.05)
]

for _, row in sig_results.iterrows():
    print(f"\n{row['network']} - {row['phase']}:")
    if row['p_anova'] < 0.05:
        print(f"  ANOVA: F={row['F_anova']:.2f}, p={row['p_anova']:.4f} *")
    if row['p_CA_JX'] < 0.05:
        print(f"  CA vs JX: t={row['t_CA_JX']:.2f}, p={row['p_CA_JX']:.4f} *")
    if row['p_cut'] < 0.05:
        print(f"  Cut Effect: t={row['t_cut']:.2f}, p={row['p_cut']:.4f} *")

print("\n" + "="*70)
print(f"All results saved to: {OUTPUT_DIR}")
print("="*70)
