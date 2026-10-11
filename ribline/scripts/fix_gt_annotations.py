"""Audits SCR ground-truth centerline files for merged-rib annotations and writes a fixed copy.

Background. In the original SCR archive the file
``RIB_007_20091210_bone_1.5_0.6_12_38_interp.xls`` labels 640 points as ``R07``
although they form two disjoint curves separated by a 220 mm jump (the second
curve is the eighth right rib, whose contralateral ``L08`` sits at the same
height). Scoring a prediction against this merged curve is meaningless, so we
split any GT rib whose consecutive-point jump exceeds ``--jump-mm`` into the
rib itself and the *next missing* label on the same side, and record every
change in ``<out_dir>/fix_log.json``. All other files are copied verbatim.

Usage::

    python -m ribline.scripts.fix_gt_annotations --gt-dir Data/Centerlines_GT \
        --out-dir Data/Centerlines_GT_fixed
"""

import argparse
import json
import os
import shutil

import numpy as np


def split_merged_ribs(rows: list, jump_mm: float) -> tuple:
    """Splits merged ribs in a list of (label, fields) rows. Returns (rows, log)."""
    by_label = {}
    order = []
    for label, fields in rows:
        if label not in by_label:
            by_label[label] = []
            order.append(label)
        by_label[label].append(fields)
    log = []
    out = {}
    for label in order:
        fields = by_label[label]
        pts = np.array([[float(f[1]), float(f[2]), float(f[3])] for f in fields])
        if len(pts) < 2 or not (label[1:].isdigit() and label[0] in "RL"):
            out[label] = fields
            continue
        jumps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        big = np.where(jumps > jump_mm)[0]
        if len(big) == 0:
            out[label] = fields
            continue
        side, num = label[0], int(label[1:])
        segments = np.split(np.arange(len(pts)), big + 1)
        # order segments superior->inferior (higher mean Z = more superior rib)
        seg_order = sorted(segments, key=lambda s: -float(pts[s, 2].mean()))
        assigned = []
        k = num
        for seg in seg_order:
            while f"{side}{k:02d}" in by_label and k != num:
                k += 1
            if k > 12:
                raise RuntimeError(f"Cannot assign split segment of {label}")
            new_label = f"{side}{k:02d}"
            new_fields = [list(fields[i]) for i in seg]
            for nf in new_fields:
                nf[0] = new_label
            out.setdefault(new_label, []).extend(new_fields)
            assigned.append({"label": new_label, "n_points": int(len(seg)),
                             "z_mean": float(pts[seg, 2].mean())})
            k += 1
        log.append({"original_label": label, "n_points": int(len(pts)),
                    "max_jump_mm": float(jumps.max()), "split_into": assigned})
    new_rows = []
    for label in sorted(out):
        for f in out[label]:
            new_rows.append((label, f))
    return new_rows, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jump-mm", type=float, default=50.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    full_log = {}
    for fname in sorted(os.listdir(args.gt_dir)):
        src = os.path.join(args.gt_dir, fname)
        dst = os.path.join(args.out_dir, fname)
        if not fname.endswith("_interp.xls"):
            continue
        rows = []
        with open(src, "r", newline="") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                try:
                    float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                rows.append((parts[0], parts))
        new_rows, log = split_merged_ribs(rows, args.jump_mm)
        if log:
            full_log[fname] = log
            with open(dst, "w", newline="") as f:
                for label, fields in new_rows:
                    f.write("\t".join(fields) + "\n")
            print(f"[fixed ] {fname}: {json.dumps(log)}")
        else:
            shutil.copyfile(src, dst)
    with open(os.path.join(args.out_dir, "fix_log.json"), "w") as f:
        json.dump({"source_dir": os.path.abspath(args.gt_dir), "jump_mm": args.jump_mm,
                   "changes": full_log}, f, indent=2)
    print(f"Wrote {len(full_log)} fixed file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
