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
    assert len(files) == 52 and len({x['path'] for x in files}) == 52
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
                'run_duration_sensitivity.py','N19_group_readback.py',
                'N19_two_image_composite.py','N19_within_cell_holm3.py',
                'extract_prepost_matched.py','analyze_prepost_matched.py',
                'verify_prepost_matched.py')):
                assert 'Analysis output must be outside the public checkout' in text
            if row['path'].endswith('/run_w10_phase_interaction_one_sided_retrospective.py'):
                assert all(token in text for token in (
                    'two_sided_test=False','n_perm=50000','20266972',
                    '20260918','participant_contrast_manifest.csv',
                    'Analysis output must be outside the public checkout'))
    for row in figure_meta['figures']:
        p = REV/'figures'/row['filename']
        assert 'inline_object_id' not in row
        assert sha(p) == row['sha256'] and row['caption'].startswith(f"Figure {row['figure']}."), row
        assert p.read_bytes().startswith((b'\x89PNG', b'\xff\xd8')), p
    maps = json.loads((MAPS/'manifest.json').read_text())
    assert len(maps) == 13 and len({x['filename'] for x in maps}) == 13
    w10 = [x for x in maps if x['model']=='DurationSensitivityW10' and x['contrast']=='PreMinusSearch_CAgtJX']
    assert len(w10)==1 and w10[0]['source_sha256']=='29fd8f876b339b53d94ced512b796dec7036cbd3594fca2b193a7fcf04d99ed3'
    assert w10[0]['filename']=='model-DurationSensitivityW10_contrast-PreMinusSearch_CAgtJX_stat-t_desc-unthresholded.nii.gz'
    assert not (REV/'data/one_second_pre/interaction_summary_two_sided.csv').exists()
    assert not (REV/'code/one_second_pre/run_w1_group_only_two_sided.py').exists()
    assert (REV/'code/one_second_pre/run_duration_sensitivity.py').is_file()
    assert len(read_rows(MAPS/'manifest.csv')) == len(maps)
    for row in maps:
        p = MAPS/row['filename']
        assert p.is_file() and sha(p) == row['source_sha256'] and p.stat().st_size == row['source_bytes'], row['filename']
        im = nib.load(p)
        a = im.get_fdata(dtype=np.float64)
        assert a.ndim == 3 and np.isfinite(a).all() and a.min() < 0 < a.max(), row['filename']
        assert row['shape'] == 'x'.join(map(str, im.shape)), row['filename']
        assert '/dss/' not in json.dumps(row) and '/home/' not in json.dumps(row), row['filename']
        header_text=' '.join(str(im.header[key]) for key in ('descrip','aux_file','intent_name','db_name'))
        header_text+=' '+' '.join(str(ext.get_content()) for ext in im.header.extensions)
        assert not re.search(r'/dss/|/home/|sub-\d{3}|Slide\d+',header_text,re.I),row['filename']
    directional=read_rows(REV/'data/one_second_phase_interaction/retrospective_one_sided_summary.csv')
    assert len(directional)==2 and {r['seed'] for r in directional}=={'20266972','20260918'}
    for r in directional:
        assert r['model']=='DurationSensitivityW10' and r['contrast']=='PreMinusSearch_CAgtJX'
        assert r['tail_status'].startswith('retrospective_positive_one_sided')
        assert (int(r['n']),int(r['n_perm']),int(r['mask_voxels']),int(r['cluster_k']))==(19,50000,48279,24)
        assert [float(r[k]) for k in ('peak_mni_x','peak_mni_y','peak_mni_z')]==[-9.,-60.,-8.7]
        assert abs(float(r['comparison_two_sided_within_map_pFWE'])-.08844)<1e-8
        target={'20266972':.0318,'20260918':.0314}[r['seed']]
        assert abs(float(r['minimum_within_map_cluster_pFWE'])-target)<1e-8
    assert (REV/'code/one_second_phase_interaction/run_w10_phase_interaction_one_sided_retrospective.py').is_file()
    # Current live supplement: S1 is example poems (not the earlier simulation
    # grid); S2 contains only the Early/Middle/Pre-response GLM and its results.
    supp = REV/'data/supplement'
    assert {p.name for p in supp.iterdir() if p.is_file()} == {
        'README.md','S2_N19_map_ledger.csv','S2_N19_clusters.csv',
    }
    assert 'not a table displayed in the live supplement' in (supp/'README.md').read_text()
    s2_maps = read_rows(supp/'S2_N19_map_ledger.csv')
    s2_clusters = read_rows(supp/'S2_N19_clusters.csv')
    assert len(s2_maps) == 21 and all(int(r['n']) == 19 for r in s2_maps)
    assert len(s2_clusters) == 41 and all(int(r['sample_n']) == 19 for r in s2_clusters)
    assert {r['contrast'] for r in s2_clusters} == {
        'phase_middle_minus_early_average','phase_pre_minus_middle_average',
        'phase_pre_minus_early_average',
    }
    assert all(r['significant'] == 'True' and float(r['pFWE']) < .05 for r in s2_clusters)
    assert not any('S1_' in r['path'] for r in files)
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
    print(f"PASS: {len(files)} source-linked assets; S2 single-GLM 21-map/41-cluster supplement; 7 live-Doc figures; 13 signed t-maps; within-map FWE; exploratory W10 one-sided ledger; aggregate privacy/code checks; headline RT/PPI/behavior values")

if __name__ == '__main__':
    main()
