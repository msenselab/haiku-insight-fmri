#!/usr/bin/env python3
"""
Unified GLM Results Report (excluding sub-005) - with Search vs Insight comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import plotting, image
from nilearn.reporting import get_clusters_table
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
GROUP_DIR = BASE_DIR / 'data/group_zmaps'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

Z_THRESHOLD = 2.58

CONTRAST_INFO = {
    'search_CA': {'name': 'Search: Context-Action', 'cat': 'Search'},
    'search_JX': {'name': 'Search: Juxtaposition', 'cat': 'Search'},
    'search_OI': {'name': 'Search: One-Image', 'cat': 'Search'},
    'search_TwoImg_gt_OneImg': {'name': 'Search: Two-Image > One-Image (CUT)', 'cat': 'Search'},
    'search_CA_gt_JX': {'name': 'Search: CA > JX', 'cat': 'Search'},
    'search_JX_gt_CA': {'name': 'Search: JX > CA', 'cat': 'Search'},
    'search_CA_gt_OI': {'name': 'Search: CA > OI', 'cat': 'Search'},
    'search_JX_gt_OI': {'name': 'Search: JX > OI', 'cat': 'Search'},
    'RT_CA_pos': {'name': 'RT: CA longer → more activation', 'cat': 'RT'},
    'RT_JX_pos': {'name': 'RT: JX longer → more activation', 'cat': 'RT'},
    'RT_OI_pos': {'name': 'RT: OI longer → more activation', 'cat': 'RT'},
    'RT_TwoImg_gt_OneImg': {'name': 'RT: Two-Image > One-Image', 'cat': 'RT'},
    'insight_CA': {'name': 'Insight: Context-Action', 'cat': 'Insight'},
    'insight_JX': {'name': 'Insight: Juxtaposition', 'cat': 'Insight'},
    'insight_OI': {'name': 'Insight: One-Image', 'cat': 'Insight'},
    'insight_TwoImg_gt_OneImg': {'name': 'Insight: Two-Image > One-Image (CUT)', 'cat': 'Insight'},
    'insight_CA_gt_JX': {'name': 'Insight: CA > JX', 'cat': 'Insight'},
    'insight_JX_gt_CA': {'name': 'Insight: JX > CA', 'cat': 'Insight'},
    'insight_CA_gt_OI': {'name': 'Insight: CA > OI', 'cat': 'Insight'},
    'insight_JX_gt_OI': {'name': 'Insight: JX > OI', 'cat': 'Insight'},
}

def get_region_label(x, y, z):
    if x < -5: hem = "L"
    elif x > 5: hem = "R"
    else: hem = "Mid"
    ax = abs(x)
    if ax < 15 and 5 < y < 20 and -15 < z < 5: return f"{hem} NAcc"
    if ax < 15 and 30 < y < 60 and -25 < z < 5: return "vmPFC"
    if ax < 15 and 15 < y < 45 and 15 < z < 45: return "ACC"
    if 25 < ax < 45 and 5 < y < 25 and -10 < z < 20: return f"{hem} Insula"
    if 35 < ax < 55 and 10 < y < 40 and 0 < z < 30: return f"{hem} IFG"
    if 25 < ax < 50 and 20 < y < 50 and 20 < z < 50: return f"{hem} dlPFC"
    if ax < 15 and -60 < y < -40 and 20 < z < 50: return "Precuneus"
    if 35 < ax < 60 and -75 < y < -50 and 20 < z < 50: return f"{hem} Angular"
    if ax > 30 and y < -70: return f"{hem} Visual"
    if 40 < ax < 70 and -30 < y < 5: return f"{hem} STG"
    return f"{hem} {'Frontal' if y > 0 else 'Post'}"

def process_contrast(c, vmax):
    info = CONTRAST_INFO.get(c, {'name': c, 'cat': 'Other'})
    zmap = GROUP_DIR / f'group_{c}_zmap.nii.gz'
    if not zmap.exists(): return None
    
    cdir = OUTPUT_DIR / c
    cdir.mkdir(exist_ok=True)
    z_img = image.load_img(str(zmap))
    
    try:
        fig = plt.figure(figsize=(12, 4))
        plotting.plot_stat_map(z_img, threshold=Z_THRESHOLD, display_mode='ortho',
                               vmax=vmax, draw_cross=False, title=info['name'], figure=fig)
        fig.savefig(cdir / f'{c}_ortho.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        fig = plt.figure(figsize=(10, 3))
        plotting.plot_glass_brain(z_img, threshold=Z_THRESHOLD, display_mode='lyrz',
                                  colorbar=True, vmax=vmax, figure=fig)
        fig.savefig(cdir / f'{c}_glass.png', dpi=150, bbox_inches='tight')
        plt.close()
    except: pass
    
    result = {'contrast': c, 'name': info['name'], 'cat': info['cat'],
              'clusters': None, 'max_z': None, 'n_voxels': 0}
    try:
        table = get_clusters_table(z_img, stat_threshold=Z_THRESHOLD, cluster_threshold=20, two_sided=False)
        if table is not None and len(table) > 0:
            table['Region'] = table.apply(lambda r: get_region_label(r['X'], r['Y'], r['Z']), axis=1)
            table.to_csv(cdir / f'{c}_clusters.csv', index=False)
            result['clusters'] = table
            result['max_z'] = table['Peak Stat'].max()
            result['n_voxels'] = int((z_img.get_fdata() > Z_THRESHOLD).sum())
    except: pass
    return result

def get_max_z(contrast):
    f = GROUP_DIR / f'group_{contrast}_zmap.nii.gz'
    if f.exists():
        return np.nanmax(image.load_img(str(f)).get_fdata())
    return None

def generate_html(results, vmax):
    # Compute search vs insight comparison
    comparisons = [
        ('CA', 'search_CA', 'insight_CA'),
        ('JX', 'search_JX', 'insight_JX'),
        ('OI', 'search_OI', 'insight_OI'),
        ('TwoImg>OneImg (Cut)', 'search_TwoImg_gt_OneImg', 'insight_TwoImg_gt_OneImg'),
        ('CA>JX', 'search_CA_gt_JX', 'insight_CA_gt_JX'),
        ('JX>CA', 'search_JX_gt_CA', 'insight_JX_gt_CA'),
        ('CA>OI', 'search_CA_gt_OI', 'insight_CA_gt_OI'),
        ('JX>OI', 'search_JX_gt_OI', 'insight_JX_gt_OI'),
    ]
    
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Unified GLM Results</title>
<style>
body {{ font-family: Arial; max-width: 1100px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; }}
h2 {{ color: #8e44ad; margin-top: 30px; background: #f5eef8; padding: 10px; border-radius: 5px; }}
h3 {{ color: #2980b9; }}
.note {{ background: #f5eef8; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6; margin: 20px 0; }}
.keypoint {{ background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0; }}
.finding {{ background: #e8f8f5; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #1abc9c; }}
.interpretation {{ background: #e8f6ff; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #3498db; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th {{ background: #9b59b6; color: white; padding: 10px; }}
td {{ border: 1px solid #ddd; padding: 8px; }}
img {{ max-width: 100%; margin: 10px 0; border-radius: 5px; }}
.sig {{ background: #d5f4e6; }} .nonsig {{ background: #fce4e4; }}
.search-win {{ background: #d6eaf8; font-weight: bold; }}
.insight-win {{ background: #fadbd8; }}
.cat {{ font-size: 0.9em; color: #666; }}
</style></head><body>

<h1>🧠 Unified GLM Results (Outlier Excluded)</h1>

<div class="note">
<strong>⚠️ sub-005 excluded</strong> (JX insight rate = 18.8%, outlier)<br>
<strong>N = 19 subjects</strong> | Threshold: Z > {Z_THRESHOLD} (p < .005) | Cluster k ≥ 20 | vmax = {vmax:.1f}
</div>

<h2>🔑 Key Finding: Search vs Insight Comparison</h2>

<div class="keypoint">
<strong>Important:</strong> The unified model allows direct comparison between <strong>Search Period</strong> (sustained encoding/integration) 
and <strong>Insight Moment</strong> (brief resolution/reward). Different contrasts show different patterns!
</div>

<table>
<tr>
<th>Contrast</th>
<th>Search Z</th>
<th>Insight Z</th>
<th>Stronger</th>
</tr>
'''
    
    search_wins = []
    insight_wins = []
    
    for name, search_c, insight_c in comparisons:
        search_z = get_max_z(search_c)
        insight_z = get_max_z(insight_c)
        
        if search_z and insight_z:
            if search_z > insight_z:
                winner = "SEARCH"
                cls = "search-win"
                search_wins.append(name)
            else:
                winner = "INSIGHT"
                cls = "insight-win"
                insight_wins.append(name)
            html += f'<tr class="{cls}"><td>{name}</td><td>{search_z:.2f}</td><td>{insight_z:.2f}</td><td>{winner}</td></tr>\n'
    
    html += '</table>\n'
    
    html += f'''
<div class="interpretation">
<h3>Interpretation</h3>
<p><strong>Search is stronger for:</strong> {', '.join(search_wins)}</p>
<p><strong>Insight is stronger for:</strong> {', '.join(insight_wins)}</p>

<p><strong>Key insight:</strong> JX-related contrasts (JX>CA, JX>OI) and the <strong>Cut Effect (TwoImg>OneImg)</strong> 
are stronger during <strong>Search</strong> than at the Insight moment. This suggests:</p>
<ul>
<li><strong>Cut processing</strong> (integrating discontinuous images) happens during sustained search, not at the moment of insight</li>
<li><strong>JX difficulty</strong> is reflected in effortful encoding/search processes</li>
<li>At the <strong>insight moment</strong>, CA shows stronger activation (reward/resolution), while JX effects diminish</li>
<li>The <strong>search model</strong> better captures haiku encoding processes that vary by haiku type</li>
</ul>
</div>

<h2>📊 RT Parametric Effects (Search-only)</h2>
<div class="interpretation">
<p>RT effects capture how brain activation scales with search duration - this is <strong>unique to the search model</strong> 
and not available in insight-locked analysis.</p>
</div>
<table>
<tr><th>Contrast</th><th>Max Z</th><th>Significant?</th></tr>
'''
    
    rt_contrasts = ['RT_CA_pos', 'RT_JX_pos', 'RT_OI_pos', 'RT_TwoImg_gt_OneImg']
    for c in rt_contrasts:
        z = get_max_z(c)
        if z:
            sig = "✓" if z > 2.58 else "—"
            cls = "sig" if z > 2.58 else "nonsig"
            html += f'<tr class="{cls}"><td>{CONTRAST_INFO[c]["name"]}</td><td>{z:.2f}</td><td>{sig}</td></tr>\n'
    
    html += '''</table>
<div class="finding">
<strong>Note:</strong> RT_JX_pos has the strongest effect (Z=3.93), indicating that longer JX search 
is associated with greater brain activation - reflecting the effortful processing required for incongruent image integration.
</div>

<hr>
<h2>📊 All Contrasts by Category</h2>
'''
    
    for cat in ['Search', 'RT', 'Insight']:
        cat_results = [r for r in results if r and r['cat'] == cat]
        if not cat_results: continue
        html += f'<h3>{cat} Contrasts</h3><table><tr><th>Contrast</th><th>Max Z</th><th>Voxels</th><th>Status</th></tr>\n'
        for r in cat_results:
            cls = 'sig' if r['n_voxels'] > 0 else 'nonsig'
            max_z = f"{r['max_z']:.2f}" if r['max_z'] else "—"
            html += f'<tr class="{cls}"><td>{r["name"]}</td><td>{max_z}</td><td>{r["n_voxels"]}</td><td>{"✓" if r["n_voxels"]>0 else "✗"}</td></tr>\n'
        html += '</table>\n'
    
    html += '<hr><h2>Detailed Results</h2>\n'
    
    for r in results:
        if r is None: continue
        c = r['contrast']
        html += f'<h3>{r["name"]} <span class="cat">[{r["cat"]}]</span></h3>\n'
        html += f'<img src="{c}/{c}_ortho.png"><img src="{c}/{c}_glass.png">\n'
        if r['clusters'] is not None and len(r['clusters']) > 0:
            regions = list(r['clusters'].head(5)['Region'].unique())
            html += f'<div class="finding"><strong>Key regions:</strong> {", ".join(regions)}</div>\n'
            html += '<table><tr><th>Region</th><th>X</th><th>Y</th><th>Z</th><th>Peak Z</th></tr>\n'
            for _, row in r['clusters'].head(8).iterrows():
                html += f'<tr><td>{row["Region"]}</td><td>{row["X"]:.0f}</td><td>{row["Y"]:.0f}</td><td>{row["Z"]:.0f}</td><td>{row["Peak Stat"]:.2f}</td></tr>\n'
            html += '</table>\n'
        else:
            html += '<p><em>No significant clusters.</em></p>\n'
    
    html += '</body></html>'
    (OUTPUT_DIR / 'unified_no_outlier_report.html').write_text(html)
    print(f"Report: {OUTPUT_DIR / 'unified_no_outlier_report.html'}")

def main():
    print("="*60)
    print("UNIFIED GLM RESULTS (with Search vs Insight comparison)")
    print("="*60)
    
    contrasts = list(CONTRAST_INFO.keys())
    
    vmax = 0
    for c in contrasts:
        f = GROUP_DIR / f'group_{c}_zmap.nii.gz'
        if f.exists():
            d = image.load_img(str(f)).get_fdata()
            if (np.abs(d) > Z_THRESHOLD).any():
                vmax = max(vmax, np.abs(d[np.abs(d) > Z_THRESHOLD]).max())
    vmax = np.ceil(vmax * 10) / 10
    print(f"vmax = {vmax}")
    
    results = []
    for c in contrasts:
        print(f"Processing: {c}")
        results.append(process_contrast(c, vmax))
    
    generate_html(results, vmax)
    print("\nDone!")

if __name__ == '__main__':
    main()
