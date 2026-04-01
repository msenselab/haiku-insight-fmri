#!/usr/bin/env python3
"""
Connectivity Extraction Script for Haiku fMRI Study
====================================================
NOTE: This script requires fMRIPrep preprocessed BOLD timeseries
(derivatives/fmriprep/) which are not included in this Tier 1 release.
Pre-computed outputs reproducing Table 5 are in:
  data/connectivity/condition_specific_connectivity.csv  (per-subject)
  data/connectivity/condition_connectivity_comparison.csv (group stats)

Extracts ROI timeseries from fMRIPrep preprocessed data and computes
condition-specific pairwise correlations.

Usage:
    python extract_connectivity.py

Output:
    ../connectivity_analysis/condition_specific_connectivity.csv

Author: Reconstructed for reproducibility
Date: 2026-01-30
"""

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiSpheresMasker
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path('/dss/studies/fmri-haiku')
FMRIPREP_DIR = BASE_DIR / 'derivatives' / 'fmriprep'
GLM_DIR = BASE_DIR / 'glm_unified'
FIRST_LEVEL_DIR = GLM_DIR / 'first_level'
OUTPUT_DIR = GLM_DIR / 'connectivity_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

# ROI definitions (MNI coordinates)
# Based on literature and group-level activation peaks
ROIS = {
    'dlPFC': (-44, 36, 26),       # Left dorsolateral prefrontal cortex
    'PCC': (0, -52, 26),          # Posterior cingulate cortex (midline)
    'L_Angular': (-48, -64, 30),  # Left angular gyrus
    'ATL': (-54, -6, -22),        # Left anterior temporal lobe
    'vmPFC': (0, 46, -10),        # Ventromedial prefrontal cortex (midline)
    'Motor': (-38, -22, 56),      # Left motor cortex (control region)
}

# Sphere radius for ROI extraction (mm)
ROI_RADIUS = 6

# TR in seconds (for event timing)
TR = 2.0

# Minimum TRs required per condition for reliable correlation
MIN_TRS_PER_CONDITION = 50

# Conditions to analyze
CONDITIONS = ['CA', 'JX', 'OI']

# =============================================================================
# Helper Functions
# =============================================================================

def get_subjects():
    """Get list of subjects with complete data."""
    subjects = []
    for sub_dir in sorted(FIRST_LEVEL_DIR.glob('sub-*')):
        sub_id = sub_dir.name
        # Check for preprocessed BOLD
        bold_file = FMRIPREP_DIR / sub_id / 'func' / f'{sub_id}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        # Check for events file
        events_file = sub_dir / f'{sub_id}_events.tsv'
        
        if bold_file.exists() and events_file.exists():
            # Check number of volumes (exclude truncated scans)
            img = nib.load(bold_file)
            n_vols = img.shape[-1]
            if n_vols >= 1400:  # Minimum expected volumes
                subjects.append(sub_id)
            else:
                print(f"  Excluding {sub_id}: only {n_vols} volumes (expected >= 1400)")
        else:
            if not bold_file.exists():
                print(f"  Excluding {sub_id}: missing preprocessed BOLD")
            if not events_file.exists():
                print(f"  Excluding {sub_id}: missing events file")
    
    return subjects


def load_events(sub_id):
    """Load and parse events file."""
    events_file = FIRST_LEVEL_DIR / sub_id / f'{sub_id}_events.tsv'
    events = pd.read_csv(events_file, sep='\t')
    return events


def get_condition_trs(events, n_vols, condition):
    """
    Get TR indices for a specific condition.
    
    Parameters
    ----------
    events : DataFrame
        Events with 'onset', 'duration', and condition columns
    n_vols : int
        Total number of volumes
    condition : str
        Condition name (CA, JX, OI)
    
    Returns
    -------
    np.array
        Boolean mask of TRs belonging to this condition
    """
    # Find condition column - could be 'trial_type', 'condition', or 'haiku_type'
    cond_col = None
    for col in ['haiku_type', 'condition', 'trial_type']:
        if col in events.columns:
            cond_col = col
            break
    
    if cond_col is None:
        raise ValueError(f"Could not find condition column in events")
    
    # Filter events for this condition
    cond_events = events[events[cond_col] == condition]
    
    # Create TR mask
    tr_mask = np.zeros(n_vols, dtype=bool)
    
    for _, event in cond_events.iterrows():
        onset_tr = int(event['onset'] / TR)
        # Include TRs during the trial (onset to onset + duration)
        # Add a few TRs for HRF delay
        duration_trs = max(1, int(event['duration'] / TR))
        hrf_delay = 3  # ~6 seconds for HRF peak
        
        start_tr = onset_tr + hrf_delay
        end_tr = min(start_tr + duration_trs, n_vols)
        
        if start_tr < n_vols:
            tr_mask[start_tr:end_tr] = True
    
    return tr_mask


def extract_roi_timeseries(sub_id):
    """
    Extract timeseries from all ROIs for a subject.
    
    Returns
    -------
    np.array
        Shape (n_timepoints, n_rois)
    """
    bold_file = FMRIPREP_DIR / sub_id / 'func' / f'{sub_id}_task-haiku_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    
    # Create masker with all ROI coordinates
    coords = list(ROIS.values())
    masker = NiftiSpheresMasker(
        seeds=coords,
        radius=ROI_RADIUS,
        standardize=True,  # Z-score within each ROI
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=TR
    )
    
    # Extract timeseries
    timeseries = masker.fit_transform(str(bold_file))
    
    return timeseries


def compute_condition_connectivity(timeseries, tr_mask):
    """
    Compute pairwise correlations for TRs in a condition.
    
    Parameters
    ----------
    timeseries : np.array
        Shape (n_timepoints, n_rois)
    tr_mask : np.array
        Boolean mask for condition TRs
    
    Returns
    -------
    dict
        Pairwise correlations and metadata
    """
    # Extract condition-specific timeseries
    cond_ts = timeseries[tr_mask]
    n_trs = cond_ts.shape[0]
    
    if n_trs < MIN_TRS_PER_CONDITION:
        return None
    
    # Compute pairwise correlations
    roi_names = list(ROIS.keys())
    results = []
    
    for i, roi1 in enumerate(roi_names):
        for j, roi2 in enumerate(roi_names):
            if i < j:  # Upper triangle only
                r, p = stats.pearsonr(cond_ts[:, i], cond_ts[:, j])
                results.append({
                    'roi1': roi1,
                    'roi2': roi2,
                    'r': r,
                    'n_trs': n_trs
                })
    
    return results


# =============================================================================
# Main Extraction
# =============================================================================

def main():
    print("=" * 70)
    print("Haiku fMRI: Connectivity Extraction")
    print("=" * 70)
    
    # Get subjects
    print("\nFinding subjects with complete data...")
    subjects = get_subjects()
    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    
    # Process each subject
    all_results = []
    
    for sub_id in subjects:
        print(f"\nProcessing {sub_id}...")
        
        try:
            # Load events
            events = load_events(sub_id)
            
            # Extract timeseries
            timeseries = extract_roi_timeseries(sub_id)
            n_vols = timeseries.shape[0]
            print(f"  Extracted {n_vols} timepoints from {len(ROIS)} ROIs")
            
            # Process each condition
            for condition in CONDITIONS:
                tr_mask = get_condition_trs(events, n_vols, condition)
                n_cond_trs = tr_mask.sum()
                print(f"  {condition}: {n_cond_trs} TRs")
                
                if n_cond_trs < MIN_TRS_PER_CONDITION:
                    print(f"    WARNING: Too few TRs, skipping")
                    continue
                
                # Compute connectivity
                conn_results = compute_condition_connectivity(timeseries, tr_mask)
                
                if conn_results:
                    for res in conn_results:
                        res['subject'] = sub_id
                        res['condition'] = condition
                        all_results.append(res)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['subject', 'condition', 'roi1', 'roi2', 'r', 'n_trs']]
        
        output_file = OUTPUT_DIR / 'condition_specific_connectivity.csv'
        df.to_csv(output_file, index=False)
        print(f"\n{'=' * 70}")
        print(f"Saved connectivity data to: {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"Subjects: {df['subject'].nunique()}")
        print(f"Conditions: {df['condition'].unique().tolist()}")
    else:
        print("\nNo results to save!")


if __name__ == '__main__':
    main()
