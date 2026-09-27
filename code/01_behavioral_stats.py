#!/usr/bin/env python3
"""
Compute behavioral statistics for N=19 sample (excluding sub-005 and sub-022).
Uses data from glm_unified/behavioral_rt.csv (already filtered to N=19).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load RT data (N=19)
# ============================================================
rt_df = pd.read_csv(Path(__file__).parent.parent / 'data/behavioral/behavioral_rt.csv')
rt_df = rt_df.dropna(subset=['subject'])  # drop empty rows
print(f"N = {len(rt_df)} subjects")
print(f"Subjects: {sorted(rt_df['subject'].tolist())}")
print()

# ============================================================
# 2. Descriptive statistics
# ============================================================
conditions = ['RT_CA', 'RT_JX', 'RT_OI']
cond_labels = ['CA', 'JX', 'OI']

print("=" * 60)
print("DESCRIPTIVE STATISTICS (RT in seconds)")
print("=" * 60)
desc_results = []
for cond, label in zip(conditions, cond_labels):
    m = rt_df[cond].mean()
    sd = rt_df[cond].std(ddof=1)
    sem = sd / np.sqrt(len(rt_df))
    median = rt_df[cond].median()
    print(f"  {label}: M = {m:.2f} s, SD = {sd:.2f}, SEM = {sem:.2f}, Median = {median:.2f}")
    desc_results.append({
        'condition': label,
        'mean': round(m, 3),
        'sd': round(sd, 3),
        'sem': round(sem, 3),
        'median': round(median, 3),
        'n': len(rt_df)
    })

print()

# ============================================================
# 3. Repeated-measures ANOVA
# ============================================================
print("=" * 60)
print("REPEATED-MEASURES ANOVA (one-way, 3 conditions)")
print("=" * 60)

# Reshape to long format for ANOVA
rt_long = rt_df.melt(id_vars='subject', value_vars=conditions,
                      var_name='condition', value_name='RT')
rt_long['condition'] = rt_long['condition'].map({
    'RT_CA': 'CA', 'RT_JX': 'JX', 'RT_OI': 'OI'
})

# Try pingouin first, fall back to manual computation
try:
    import pingouin as pg
    aov = pg.rm_anova(data=rt_long, dv='RT', within='condition', subject='subject',
                       correction=True, detailed=True)
    print(aov.to_string())
    print()

    # Extract key values
    F_val = aov['F'].values[0]
    df_num = int(aov['DF'].values[0])
    df_den = int(aov.loc[aov['Source'] == 'Error', 'DF'].values[0]) if 'Error' in aov['Source'].values else int((len(rt_df) - 1) * (len(conditions) - 1))
    p_val = aov['p-unc'].values[0]
    # pingouin rm_anova with detailed=True provides ng2 (generalized eta^2)
    # Compute partial eta^2 manually: SS_condition / (SS_condition + SS_error)
    ss_cond = aov.loc[aov['Source'] == 'condition', 'SS'].values[0]
    ss_error = aov.loc[aov['Source'] == 'Error', 'SS'].values[0]
    eta_sq = ss_cond / (ss_cond + ss_error)  # partial eta squared
    ng2 = aov['ng2'].values[0] if 'ng2' in aov.columns else None

    # Extract sphericity from the ANOVA table itself
    mauchly_w = aov['W-spher'].values[0] if 'W-spher' in aov.columns else None
    mauchly_p = aov['p-spher'].values[0] if 'p-spher' in aov.columns else None
    spher_met = aov['sphericity'].values[0] if 'sphericity' in aov.columns else None
    eps = aov['eps'].values[0] if 'eps' in aov.columns else None
    p_gg = aov['p-GG-corr'].values[0] if 'p-GG-corr' in aov.columns else None

    if mauchly_w is not None:
        print(f"\nMauchly's test of sphericity: W = {mauchly_w:.4f}, p = {mauchly_p:.4f}")
        print(f"  Sphericity assumption met: {spher_met}")
    if eps is not None:
        print(f"  Greenhouse-Geisser epsilon = {eps:.4f}")
    if p_gg is not None:
        print(f"  GG-corrected p = {p_gg:.4f}")
    print(f"  Partial eta-squared = {eta_sq:.4f}")
    if ng2 is not None:
        print(f"  Generalized eta-squared = {ng2:.6f}")

    has_pingouin = True

except ImportError:
    print("pingouin not available, using manual computation")
    has_pingouin = False

    # Manual repeated-measures ANOVA (one-way)
    k = len(conditions)  # number of conditions
    n = len(rt_df)  # number of subjects

    # Grand mean
    grand_mean = rt_long['RT'].mean()

    # Condition means
    cond_means = {c: rt_df[c].mean() for c in conditions}

    # Subject means
    subj_means = rt_df[conditions].mean(axis=1)

    # SS_condition
    ss_cond = n * sum((m - grand_mean)**2 for m in cond_means.values())

    # SS_subject
    ss_subj = k * sum((sm - grand_mean)**2 for sm in subj_means)

    # SS_total
    ss_total = sum((rt_df[c] - grand_mean).pow(2).sum() for c in conditions)

    # SS_error = SS_total - SS_condition - SS_subject
    ss_error = ss_total - ss_cond - ss_subj

    df_cond = k - 1
    df_subj = n - 1
    df_error = (k - 1) * (n - 1)

    ms_cond = ss_cond / df_cond
    ms_error = ss_error / df_error

    F_val = ms_cond / ms_error
    p_val = 1 - stats.f.cdf(F_val, df_cond, df_error)
    eta_sq = ss_cond / (ss_cond + ss_error)  # partial eta squared

    df_num = df_cond
    df_den = df_error

    print(f"  SS_condition = {ss_cond:.4f}")
    print(f"  SS_error = {ss_error:.4f}")
    print(f"  df_condition = {df_cond}, df_error = {df_error}")
    print(f"  F({df_cond}, {df_error}) = {F_val:.3f}, p = {p_val:.4f}")
    print(f"  Partial eta-squared = {eta_sq:.4f}")

    mauchly_w = None
    mauchly_p = None
    eps = None

print()
print(f"ANOVA SUMMARY: F({df_num}, {df_den}) = {F_val:.3f}, p = {p_val:.4f}, partial eta^2 = {eta_sq:.4f}")
print()

# ============================================================
# 4. Pairwise t-tests (paired, two-tailed, Bonferroni-corrected)
# ============================================================
print("=" * 60)
print("PAIRWISE COMPARISONS (paired t-tests, Bonferroni-corrected)")
print("=" * 60)

pairs = [('RT_CA', 'RT_JX', 'CA vs JX'),
         ('RT_CA', 'RT_OI', 'CA vs OI'),
         ('RT_JX', 'RT_OI', 'JX vs OI')]

n_comparisons = len(pairs)
pairwise_results = []

for c1, c2, label in pairs:
    d = rt_df[c1] - rt_df[c2]
    t_stat, p_uncorr = stats.ttest_rel(rt_df[c1], rt_df[c2])
    p_bonf = min(p_uncorr * n_comparisons, 1.0)

    # Cohen's d for paired samples
    cohens_d = d.mean() / d.std(ddof=1)

    # Mean difference
    mean_diff = d.mean()
    se_diff = d.std(ddof=1) / np.sqrt(len(d))

    print(f"\n  {label}:")
    print(f"    Mean diff = {mean_diff:.3f} s (SE = {se_diff:.3f})")
    print(f"    t({len(rt_df)-1}) = {t_stat:.3f}, p_uncorrected = {p_uncorr:.4f}, p_Bonferroni = {p_bonf:.4f}")
    print(f"    Cohen's d = {cohens_d:.3f}")

    sig_str = ""
    if p_bonf < 0.001:
        sig_str = "***"
    elif p_bonf < 0.01:
        sig_str = "**"
    elif p_bonf < 0.05:
        sig_str = "*"
    elif p_bonf < 0.10:
        sig_str = "+"
    else:
        sig_str = "ns"

    print(f"    Significance: {sig_str}")

    pairwise_results.append({
        'comparison': label,
        'mean_diff': round(mean_diff, 4),
        'se_diff': round(se_diff, 4),
        't': round(t_stat, 3),
        'df': len(rt_df) - 1,
        'p_uncorrected': round(p_uncorr, 4),
        'p_bonferroni': round(p_bonf, 4),
        'cohens_d': round(cohens_d, 3),
        'significance': sig_str
    })

print()

# ============================================================
# 5. Save Figure-3 RT results
# ============================================================
# Descriptive stats
desc_df = pd.DataFrame(desc_results)

# ANOVA summary
anova_summary = pd.DataFrame([{
    'test': 'rm_ANOVA',
    'F': round(F_val, 3),
    'df_num': df_num,
    'df_den': df_den,
    'p': round(p_val, 4),
    'partial_eta_sq': round(eta_sq, 4),
    'mauchly_W': round(mauchly_w, 4) if mauchly_w is not None else 'NA',
    'mauchly_p': round(mauchly_p, 4) if mauchly_p is not None else 'NA',
    'epsilon_GG': round(eps, 4) if eps is not None else 'NA'
}])

# Pairwise results
pairwise_df = pd.DataFrame(pairwise_results)

# Combine all
output_path = Path(__file__).parent.parent / 'data/behavioral/behavioral_stats_n19.csv'
with open(output_path, 'w') as f:
    f.write("# Behavioral Statistics for N=19 (excluding sub-005, sub-022)\n")
    f.write("# Generated by recompute_behavioral_n19.py\n\n")
    f.write("## Descriptive Statistics (RT in seconds)\n")
    desc_df.to_csv(f, index=False)
    f.write("\n## ANOVA Summary\n")
    anova_summary.to_csv(f, index=False)
    f.write("\n## Pairwise Comparisons\n")
    pairwise_df.to_csv(f, index=False)

print(f"Results saved to {output_path}")
print()
print("=" * 60)
print("SUMMARY OF KEY RESULTS")
print("=" * 60)
print(f"\nN = {len(rt_df)} subjects (sub-005 and sub-022 excluded)")
print(f"\nDescriptive stats:")
for r in desc_results:
    print(f"  {r['condition']}: M = {r['mean']:.2f}, SD = {r['sd']:.2f}")
print(f"\nANOVA: F({df_num}, {df_den}) = {F_val:.3f}, p = {p_val:.4f}, eta_p^2 = {eta_sq:.4f}")
print(f"\nPairwise (Bonferroni-corrected):")
for r in pairwise_results:
    print(f"  {r['comparison']}: t({r['df']}) = {r['t']:.3f}, p_bonf = {r['p_bonferroni']:.4f}, d = {r['cohens_d']:.3f} [{r['significance']}]")
