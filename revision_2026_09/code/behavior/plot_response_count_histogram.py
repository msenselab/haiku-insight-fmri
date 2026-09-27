import os
#!/usr/bin/env python3
"""Plot the raw-complete number of button responses per presentation."""
from pathlib import Path
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(os.environ["HAIKU_PROJECT_ROOT"])
SOURCE_DIR = ROOT / 'glm_unified/connectivity_analysis/ppi_precuneus_amygdala_ofc_rawlog_phases/events'
OUT = ROOT / 'glm_unified/reports/behavioral_response_count_histogram_20260921'
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(SOURCE_DIR / 'sub-*_trials.csv')))
assert len(files) == 19, f'Expected 19 participant trial files, found {len(files)}'
frames = []
for path in files:
    df = pd.read_csv(path)
    needed = {'subject_id', 'TrialID', 'condition', 'raw_buttons_s_json'}
    assert needed.issubset(df.columns), f'Missing columns in {path}'
    frames.append(df[list(needed)].copy())
trials = pd.concat(frames, ignore_index=True)
assert len(trials) == 798, f'Expected 798 presentations, found {len(trials)}'
assert not trials.duplicated(['subject_id', 'TrialID']).any()

def count_buttons(value):
    if pd.isna(value) or str(value).strip() == '':
        return 0
    parsed = json.loads(str(value))
    return len(parsed)

trials['n_responses'] = trials['raw_buttons_s_json'].map(count_buttons).astype(int)
assert trials['n_responses'].sum() == 1891
assert (trials['n_responses'] > 0).sum() == 779

counts = trials['n_responses'].value_counts().sort_index()
full_index = pd.RangeIndex(0, int(trials['n_responses'].max()) + 1, name='n_responses')
counts = counts.reindex(full_index, fill_value=0)
summary = pd.DataFrame({
    'n_responses': counts.index,
    'n_presentations': counts.values,
    'percent_presentations': counts.values / len(trials) * 100,
})

trials[['subject_id', 'TrialID', 'condition', 'n_responses']].to_csv(
    OUT / 'trial_response_counts.csv', index=False
)
summary.to_csv(OUT / 'response_count_distribution.csv', index=False)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
fig, ax = plt.subplots(figsize=(8.2, 5.2))
bar_color = '#5B4B8A'
bars = ax.bar(summary['n_responses'], summary['n_presentations'],
              width=0.78, color=bar_color, edgecolor='white', linewidth=0.8)
for bar, n, pct in zip(bars, summary['n_presentations'], summary['percent_presentations']):
    if n:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(n)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
ax.set_title('Number of button responses per presentation')
ax.set_xlabel('Number of responses')
ax.set_ylabel('Number of presentations')
ax.set_xticks(summary['n_responses'])
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, summary['n_presentations'].max() * 1.18)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#D9D9D9', linewidth=0.7, alpha=0.75)
ax.set_axisbelow(True)
ax.text(0.99, 0.98,
        'Raw-complete N = 19; 798 presentations; 1,891 responses',
        transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color='#444444')
fig.tight_layout()
fig.savefig(OUT / 'response_count_histogram.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'response_count_histogram.pdf', bbox_inches='tight')
plt.close(fig)

print(summary.to_string(index=False, formatters={'percent_presentations': '{:.2f}'.format}))
print(f'Wrote outputs to {OUT}')
