"""DEVELOPMENT-TIME EXPERIMENT -- NOT PART OF THE EVALUATION PROTOCOL.

This script was used once during development (January 2026) to test whether
the SCR ground truth contained systematic label-index shifts: for every GT file
it searched integer label shifts in [-3, 3] that would minimise the centroid
distance to an early prediction and wrote the result to
``Rib/Data/Centerlines_GT_corrected``.  The search selected a shift of 0 for
every subject, so the "corrected" directory contains exactly the same labels
and coordinates as the original ground truth (parsed and compared for all 60
files, see
``experiments/rebuttal/provenance.json`` -> ``gt_corrected_vs_gt``).  The
released U-Net was trained from that directory, i.e. on the ORIGINAL geometry.
The script is kept only for transparency.  For the one genuine annotation
error we found (``RIB_007`` R07/R08 merged) use ``ribline.scripts.fix_gt_annotations``.
"""

import os
import numpy as np
import pandas as pd
from ribline.data.centerline_parser import parse_rib_xls


# Re-implement loaders locally since they were missing (or mis-named)

def parse_centerlines(path):
    # Returns DataFrame
    try:
        # Check size valid
        # Removed explicit size check, pandas should handle empty files gracefully
        return pd.read_csv(path, sep='\t', header=None, names=['RibLabel', 'X', 'Y', 'Z', 'Extra1', 'Extra2', 'Extra3'], on_bad_lines='skip')
    except Exception as e:
        print(f"Warning: Failed to parse {path}: {e}")
        return pd.DataFrame()

def load_centerlines_from_txt(path):
    # Predicted is Space separated? Or Tab?
    # View file pat3...txt: "RibLabel X Y Z"
    # It might be space separated.
    ribs = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4: continue
                label = parts[0]
                coords = [float(x) for x in parts[1:4]]
                if label not in ribs: ribs[label] = []
                ribs[label].append(coords)
        return {k: np.array(v) for k, v in ribs.items()}
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def load_gt_df(path):
    # Parse generic GT format
    return parse_centerlines(path)

def save_gt_df(df, path):
    # Save back to tab-separated
    # Columns: RibLabel, X, Y, Z, ...
    # Keep original columns if possible, but RibLabel is the first.
    df.to_csv(path, sep='\t', index=False)

def calculate_chamfer(pred_ribs, gt_ribs):
    # Simplified Chamfer: Distance between centroids of same-label ribs
    # We only care about label alignment.
    
    total_dist = 0.0
    count = 0
    
    for label, pred_points in pred_ribs.items():
        if label in gt_ribs:
            gt_points = gt_ribs[label]
            # Centroid dist
            p_mean = np.mean(pred_points, axis=0)
            g_mean = np.mean(gt_points, axis=0)
            total_dist += np.linalg.norm(p_mean - g_mean)
            count += 1
            
    return total_dist / count if count > 0 else float('inf')

def correct_labels(gt_path, pred_path, save_path):
    if not os.path.exists(pred_path):
        print(f"Skipping {os.path.basename(gt_path)} (No prediction)")
        # Just copy
        df = pd.read_csv(gt_path, sep='\t')
        df.to_csv(save_path, sep='\t', index=False)
        return

    # Load
    gt_df = parse_centerlines(gt_path)
    pred_ribs = load_centerlines_from_txt(pred_path) # Returns dict
    
    # Convert GT to dict
    gt_ribs = {}
    for _, row in gt_df.iterrows():
        label = row['RibLabel']
        pts = row[['X', 'Y', 'Z']].values.astype(float)
        if label not in gt_ribs: gt_ribs[label] = []
        gt_ribs[label].append(pts)
    gt_ribs = {k: np.array(v) for k, v in gt_ribs.items()}
    
    # Try shifts: -2 to +2
    best_shift = 0
    min_dist = calculate_chamfer(pred_ribs, gt_ribs)
    
    print(f"Subject {os.path.basename(gt_path)}: Initial Dist {min_dist:.2f}")
    
    shifts = range(-3, 4)
    
    for shift in shifts:
        if shift == 0: continue
        
        # Apply shift to GT keys
        shifted_gt = {}
        for k, v in gt_ribs.items():
            side = k[0]
            num = int(k[1:])
            new_num = num + shift
            if 1 <= new_num <= 12:
                new_key = f"{side}{new_num:02d}"
                shifted_gt[new_key] = v
                
        dist = calculate_chamfer(pred_ribs, shifted_gt)
        if dist < min_dist:
            min_dist = dist
            best_shift = shift
            
    print(f"  Best Shift: {best_shift} (Dist {min_dist:.2f})")
    
    # Apply best shift to DataFrame
    new_rows = []
    for i, row in gt_df.iterrows():
        label = row['RibLabel']
        side = label[0]
        try:
            num = int(label[1:])
            new_num = num + best_shift
            if 1 <= new_num <= 12:
                row['RibLabel'] = f"{side}{new_num:02d}"
                new_rows.append(row)
        except:
             new_rows.append(row) # Keep non-standard if any
             
    new_df = pd.DataFrame(new_rows)
    save_gt_df(new_df, save_path)

def main():
    gt_dir = "Rib/Data/Centerlines_GT"
    corrected_dir = "Rib/Data/Centerlines_GT_corrected"
    pred_dir = "Rib/experiments/sota/test_integration/centerlines"
    
    os.makedirs(corrected_dir, exist_ok=True)
    
    for fname in os.listdir(gt_dir):
        if not fname.endswith('.txt') and not fname.endswith('.xls'): continue
        
        gt_path = os.path.join(gt_dir, fname)
        save_path = os.path.join(corrected_dir, fname)
        
        # Find prediction (always txt)
        # GT filenames: e.g. pat3_Anonymous_0002.txt
        pred_path = os.path.join(pred_dir, fname)
        # Handle extension mismatch if any (GT might be .xls?) 
        # But loader supports both. Check strict name match first.
        
        if not os.path.exists(pred_path):
             # Try replacing extension
             name_no_ext = os.path.splitext(fname)[0]
             pred_path = os.path.join(pred_dir, name_no_ext + ".txt")
             
        correct_labels(gt_path, pred_path, save_path)

if __name__ == "__main__":
    main()
