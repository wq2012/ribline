"""Per-rib arc-length statistics of predicted (or ground-truth) centerline files.

Usage::

    python -m ribline.scripts.analyze_rib_lengths --pred-dir experiments/ablation/9_hybrid_full \
        [--subjects subj1 subj2 ...]

No subject identifiers are hard-coded: pass ``--subjects`` to print per-subject
tables, otherwise only the per-rib-number aggregate is printed.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_ribs(path):
    ribs = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4: continue
                label = parts[0]
                try:
                    coords = np.array([float(x) for x in parts[1:4]])
                except ValueError:
                    continue
                if label not in ribs: ribs[label] = []
                ribs[label].append(coords)
        return {k: np.array(v) for k, v in ribs.items()}
    except OSError:
        return {}


def compute_arc_length(points):
    # Sum of segment lengths
    if len(points) < 2: return 0.0
    diffs = points[1:] - points[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)


def analyze_lengths(pred_dir, pattern="*.txt"):
    files = glob.glob(os.path.join(pred_dir, pattern))
    data = []
    for p in files:
        subject = os.path.basename(p).replace('.txt', '').replace('_interp.xls', '')
        ribs = load_ribs(p)
        for label, pts in ribs.items():
            length = compute_arc_length(pts)
            side = label[0]
            try:
                num = int(label[1:])
            except ValueError:
                continue
            data.append({'subject': subject, 'rib': label, 'side': side, 'num': num, 'length': length})
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--pattern", default="*.txt", help="glob pattern, e.g. '*_interp.xls' for GT")
    ap.add_argument("--subjects", nargs="*", default=[])
    args = ap.parse_args()
    df = analyze_lengths(args.pred_dir, args.pattern)
    if df.empty:
        print("No centerline files found.")
        return
    stats = df.groupby('num')['length'].agg(['mean', 'std', 'min', 'max', 'count'])
    print("Rib Length Stats (mm) per rib number:")
    print(stats)
    for subj in args.subjects:
        print(f"\nSubject: {subj}")
        subj_df = df[df['subject'] == subj].sort_values('num')
        print(subj_df[['rib', 'length']])


if __name__ == "__main__":
    main()
