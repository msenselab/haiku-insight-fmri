"""Verify the public aggregate-only revised-manuscript tier (no private inputs)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
MAPS = ROOT / 'unthresholded_tmaps'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    provenance = json.loads((REV/'provenance.json').read_text())
    figure_meta = json.loads((REV/'figures/captions.json').read_text())
    files = provenance['files']
    assert len(files) == 53 and len({x['path'] for x in files}) == 53
    assert figure_meta['tab_id'] == 't.0' and provenance['tab_id'] == 't.0'
    assert not any(key in figure_meta or key in provenance for key in ('document_id','document_revision_id','revision_id'))
    assert sorted(x['figure'] for x in figure_meta['figures']) == list(range(1, 8))
    # These are the only inherited participant-level release inputs, expressly
    # retained for unchanged Figure 3; do not mistake the new tier for raw data.
    inherited = ROOT/'data/behavioral'
    assert {p.name for p in inherited.iterdir() if p.is_file()} == {
        'behavioral_rt.csv','behavioral_stats_n19.csv',
        'figure1_rt_subject_values.csv','figure1_rt_summary.csv',
    }
    assert len(read_rows(inherited/'behavioral_rt.csv')) == 19
    assert len(read_rows(inherited/'figure1_rt_subject_values.csv')) == 57
    assert {p.name for p in (ROOT/'code').iterdir() if p.is_file()} == {
        '01_behavioral_stats.py','01_plot_behavioral_rt_figure.py',
    }
    for name in ('01_behavioral_stats.py','01_plot_behavioral_rt_figure.py'):
        script=ROOT/'code'/name
        compile(script.read_text(), str(script), 'exec')
    assert not any(any(p.is_file() for p in (ROOT/'data'/name).rglob('*')) for name in
                   ('granger','ppi','group_zmaps','roi','wholebrain','connectivity','mediation','brain_behavior'))
    assert not (ROOT/'reports').exists()
    for row in files:
        p = ROOT / row['path']
        assert p.is_file() and sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], row['path']
        if '/data/' in row['path'] and p.suffix == '.csv':
            text = p.read_text(errors='replace')
            assert not re.search(r'sub-\d{3}|/dss/|/home/|Slide\d+', text, re.I), row['path']
        if '/code/' in row['path'] and p.suffix == '.py':
            text = p.read_text()
            compile(text, str(p), 'exec')
            assert '/dss/' not in text and '/home/' not in text, row['path']
            if any(row['path'].endswith('/'+name) for name in (
                'run_duration_sensitivity.py','run_w1_group_only_two_sided.py','N19_group_readback.py',
                'N19_two_image_composite.py','N19_within_cell_holm3.py',
                'extract_prepost_matched.py','analyze_prepost_matched.py',
                'verify_prepost_matched.py')):
                assert 'Analysis output must be outside the public checkout' in text
    for row in figure_meta['figures']:
        p = REV/'figures'/row['filename']
        assert 'inline_object_id' not in row
        assert sha(p) == row['sha256'] and row['caption'].startswith(f"Figure {row['figure']}."), row
        assert p.read_bytes().startswith((b'\x89PNG', b'\xff\xd8')), p
    maps = json.loads((MAPS/'manifest.json').read_text())
    assert len(maps) == 12 and len({x['filename'] for x in maps}) == 12
    assert not any(x['model']=='DurationSensitivityW10' and x['contrast']=='PreMinusSearch_CAgtJX' for x in maps)
    assert not (REV/'data/one_second_pre/interaction_summary_two_sided.csv').exists()
    w1_script=REV/'code/one_second_pre/run_w1_group_only_two_sided.py'
    assert w1_script.is_file() and "key=f'interaction_{a}_minus_{b}'" in w1_script.read_text()
    assert len(read_rows(MAPS/'manifest.csv')) == len(maps)
    for row in maps:
        p = MAPS/row['filename']
        assert p.is_file() and sha(p) == row['source_sha256'] and p.stat().st_size == row['source_bytes'], row['filename']
        im = nib.load(p)
        a = im.get_fdata(dtype=np.float64)
        assert a.ndim == 3 and np.isfinite(a).all() and a.min() < 0 < a.max(), row['filename']
        assert row['shape'] == 'x'.join(map(str, im.shape)), row['filename']
        assert '/dss/' not in json.dumps(row) and '/home/' not in json.dumps(row), row['filename']
    r = read_rows(REV/'data/roi/correlations_grid7.csv')
    key = [x for x in r if x['phase'].lower().startswith('post') and x['roi'].lower().startswith('angular') and x['condition']=='CA' and x['outcome'].lower() == 'rt']
    assert len(key) == 1 and abs(float(key[0]['r']) + 0.668) < 0.002, key
    ppi = read_rows(REV/'data/ppi/N19_retrospective_Holm3_ledger.csv')
    key = [x for x in ppi if x['target']=='angular_historical' and x['phase']=='first' and x['contrast']=='CA' and x['test_kind']=='coefficient_vs_zero']
    assert len(key)==1 and abs(float(key[0]['holm_within_cell']) - 0.0468902587890625)<1e-7
    summary = read_rows(REV/'data/behavior/response_count_distribution.csv')
    assert sum(int(x['n_presentations']) for x in summary) == 798
    assert sum(int(x['n_responses'])*int(x['n_presentations']) for x in summary) == 1891
    assert not (REV/'data/response_order/first_to_third_descriptive_summary.csv').exists(), 'N20/18/15 table cannot represent manuscript N19/19/15'
    assert not (REV/'data/response_order/N19_18_15_two_event_display_summary.csv').exists(), 'Undisplayed variant is not a current-figure source'
    rows=read_rows(REV/'data/response_order/figure6C_N19_19_15_all_available_summary.csv')
    assert len(rows)==12 and len({r['roi_key'] for r in rows})==4
    counts={int(r['button_order']):int(r['n']) for r in rows}
    assert [counts[k] for k in (1,2,3)]==[19,19,15]
    assert all(int(r['n'])==[19,19,15][int(r['button_order'])-1] for r in rows)
    interaction = [x for x in read_rows(REV/'data/first_response/contrast_summary.csv') if x['map_key']=='phase_pre_gt_search_CA_gt_JX']
    assert len(interaction)==1 and int(interaction[0]['n_significant_clusters'])>=1
    assert abs(float(interaction[0]['minimum_cluster_pFWE']) - 0.03592) < 1e-6
    print(f"PASS: {len(files)} source-linked assets; 7 live-Doc figures; 12 signed t-maps; within-map FWE; aggregate privacy/code checks; headline RT/PPI/behavior values")

if __name__ == '__main__':
    main()
