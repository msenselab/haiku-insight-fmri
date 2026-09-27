#!/usr/bin/env python3
"""Plot behavioral RT-to-first-insight results for fmri-haiku.

Creates a manuscript-style within-subject condition plot using the saved
subject-level behavioral RT table and annotates the reported repeated-measures
ANOVA / Bonferroni-corrected pairwise comparisons.
"""

from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RELEASE = Path(__file__).resolve().parents[1]
DATA = RELEASE / 'data' / 'behavioral' / 'behavioral_rt.csv'
OUTDIR = Path(os.environ['HAIKU_FIGURE3_OUTPUT_DIR']).expanduser().resolve() if os.environ.get('HAIKU_FIGURE3_OUTPUT_DIR') else Path(tempfile.mkdtemp(prefix='haiku-figure3-', dir=os.environ.get('TMPDIR')))
if RELEASE == OUTDIR or RELEASE in OUTDIR.parents:
    raise RuntimeError('Regeneration output must be outside the public checkout')
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG = OUTDIR / 'fig_behavioral_rt_first_insight_conditions.png'
PDF = OUTDIR / 'fig_behavioral_rt_first_insight_conditions.pdf'
SUMMARY_CSV = OUTDIR / 'fig_behavioral_rt_first_insight_conditions_summary.csv'
LONG_CSV = OUTDIR / 'fig_behavioral_rt_first_insight_conditions_subject_values.csv'

CONDITIONS = ['CA', 'JX', 'OI']
COLS = {c: f'RT_{c}' for c in CONDITIONS}
COLORS = {'CA': '#7B5EA7', 'JX': '#D8904F', 'OI': '#6C8EBF'}
LABELS = {'CA': 'CA', 'JX': 'JX', 'OI': 'OI'}


def paired_dz(x, y):
    d = np.asarray(x) - np.asarray(y)
    return d.mean() / d.std(ddof=1)


def add_bracket(ax, x1, x2, y, h, text, fs=15):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.8, c='0.2', clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.10, text, ha='center', va='bottom', fontsize=fs)


def main():
    df = pd.read_csv(DATA)
    long = df.melt(id_vars='subject', value_vars=list(COLS.values()),
                   var_name='condition', value_name='rt_s')
    long['condition'] = long['condition'].str.replace('RT_', '', regex=False)
    long['condition'] = pd.Categorical(long['condition'], categories=CONDITIONS, ordered=True)
    long = long.sort_values(['subject', 'condition'])

    # Save exact figure data.
    long.to_csv(LONG_CSV, index=False)

    summary_rows = []
    for cond in CONDITIONS:
        vals = df[COLS[cond]].dropna().to_numpy()
        summary_rows.append({
            'condition': cond,
            'n': len(vals),
            'mean_rt_s': vals.mean(),
            'sd_rt_s': vals.std(ddof=1),
            'sem_rt_s': stats.sem(vals, nan_policy='omit'),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    # Manuscript-style paired condition figure.
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 16,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    x = np.arange(len(CONDITIONS))

    # light participant trajectories
    wide = df[['subject'] + list(COLS.values())].copy()
    for _, row in wide.iterrows():
        ys = [row[COLS[c]] for c in CONDITIONS]
        ax.plot(x, ys, color='0.78', lw=1.0, alpha=0.75, zorder=1)

    # participant dots with deterministic jitter
    rng = np.random.default_rng(20260430)
    for i, cond in enumerate(CONDITIONS):
        vals = df[COLS[cond]].to_numpy()
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=38, facecolor=COLORS[cond], edgecolor='white', linewidth=0.7,
                   alpha=0.78, zorder=3)

    means = summary['mean_rt_s'].to_numpy()
    sems = summary['sem_rt_s'].to_numpy()
    ax.errorbar(x, means, yerr=sems, fmt='o', ms=13, lw=2.5,
                color='black', ecolor='black', capsize=7, capthick=2.2, zorder=5)

    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0, 14.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS])
    ax.set_ylabel('RT to first insight (s)')
    ax.set_xlabel('Haiku type')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis='both', length=5, width=1.5)

    fig.tight_layout()
    fig.savefig(PNG, dpi=450, bbox_inches='tight')
    fig.savefig(PDF, bbox_inches='tight')
    plt.close(fig)

    # Also print a compact verification summary.
    print(f'Saved: {PNG}')
    print(f'Saved: {PDF}')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
