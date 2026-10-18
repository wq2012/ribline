"""Recomputes every thoracic-frame / polar-angle statistic quoted in the paper.

For every annotated rib in the selected subjects this script reports

* strict monotonicity of the unwrapped kyphosis-aware polar angle
  ``get_polar_angle`` (after orienting so that theta_end > theta_start),
* the magnitude of the largest local reversal (max_i theta_i - min_{j>=i} theta_j),
  with counts above 0.02 / 0.05 / 0.1 / 0.3 rad,
* the same quantities for a naive torso-centre polar angle
  ``atan2(-(Y - y_lung), |X - x_spine|)`` for comparison,
* the Z offset between the posterior (spine-side) end of ``R01``/``L01`` and the
  detected lung apex ``z_apex``,
* the spine-midline error |X_spine(Z) - 0.5 (X_R,post + X_L,post)| per rib pair.

Usage::

    python -m ribline.scripts.polar_angle_stats --image-dir Data/vol2mm \
        --gt-dir Data/Centerlines_GT --split split.json --out polar_stats.json
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image, resolve_subject_image_path
from ribline.model.anatomy import extract_thoracic_frame, get_polar_angle


def _reversal_stats(ang: np.ndarray) -> tuple:
    if ang[-1] < ang[0]:
        ang = ang[::-1]
    d = np.diff(ang)
    strictly_mono = bool(np.all(d > 0))
    rev = float(np.max(ang - np.minimum.accumulate(ang[::-1])[::-1]))
    return strictly_mono, rev, bool(ang[-1] > ang[0])


def naive_polar_angle(pts: np.ndarray, frame: dict) -> np.ndarray:
    dx = np.abs(pts[:, 0] - frame["x_spine"])
    ang = np.arctan2(-(pts[:, 1] - frame["y_lung"]), dx)
    return np.unwrap(ang)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--split-key", default="all", help="train | val | all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    split = json.load(open(args.split))
    subjects = (split["train"] + split["val"]) if args.split_key == "all" else split[args.split_key]

    thr = [0.02, 0.05, 0.1, 0.3]
    res = {"kyphosis_aware": {"n_ribs": 0, "strictly_monotonic": 0, "end_gt_start": 0, "rev": []},
           "naive_torso_centre": {"n_ribs": 0, "strictly_monotonic": 0, "end_gt_start": 0, "rev": []}}
    rib1_offsets = []
    spine_err = []
    n_subj = 0
    per_rib_nonmono = {}
    for s in subjects:
        gt_path = os.path.join(args.gt_dir, f"{s}_interp.xls")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.gt_dir, f"{s}.xls")
        img_path = resolve_subject_image_path(args.image_dir, s)
        if not (os.path.exists(gt_path) and os.path.exists(img_path)):
            continue
        gt = parse_rib_xls(gt_path)
        if not gt:
            continue
        n_subj += 1
        img = load_image(img_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        frame = extract_thoracic_frame(arr, spacing=img.GetSpacing(), origin=img.GetOrigin())
        oz, sz = frame["origin"][2], frame["spacing"][2]
        posts = {}
        for r, pts in gt.items():
            if len(pts) < 3:
                continue
            for key, fn in (("kyphosis_aware", lambda p: get_polar_angle(p, frame, unwrap=True)),
                            ("naive_torso_centre", lambda p: naive_polar_angle(p, frame))):
                mono, rev, egs = _reversal_stats(np.asarray(fn(pts), dtype=np.float64))
                e = res[key]
                e["n_ribs"] += 1
                e["strictly_monotonic"] += int(mono)
                e["end_gt_start"] += int(egs)
                e["rev"].append(rev)
                if key == "kyphosis_aware" and not mono:
                    per_rib_nonmono[r[1:]] = per_rib_nonmono.get(r[1:], 0) + 1
            # posterior (spine-side) endpoint
            e0 = abs(pts[0, 0] - frame["x_spine"])
            e1 = abs(pts[-1, 0] - frame["x_spine"])
            p_post = pts[0] if e0 < e1 else pts[-1]
            posts[r] = p_post
            if r in ("R01", "L01"):
                rib1_offsets.append(float(p_post[2] - frame["z_apex"]))
        for k in range(1, 13):
            rr, ll = f"R{k:02d}", f"L{k:02d}"
            if rr in posts and ll in posts:
                z = 0.5 * (posts[rr][2] + posts[ll][2])
                iz = int(np.clip(round((z - oz) / sz), 0, len(frame["x_spine_z"]) - 1))
                x_mid = 0.5 * (posts[rr][0] + posts[ll][0])
                spine_err.append(abs(float(frame["x_spine_z"][iz]) - x_mid))

    out = {"n_subjects": n_subj}
    for key, e in res.items():
        rev = np.array(e["rev"])
        out[key] = {
            "n_ribs": e["n_ribs"],
            "strictly_monotonic": e["strictly_monotonic"],
            "strictly_monotonic_frac": e["strictly_monotonic"] / max(e["n_ribs"], 1),
            "end_gt_start": e["end_gt_start"],
            "non_monotonic": e["n_ribs"] - e["strictly_monotonic"],
            "max_reversal_rad": float(rev.max()) if len(rev) else None,
            "n_reversal_gt": {str(t): int((rev > t).sum()) for t in thr},
        }
    out["kyphosis_aware"]["non_monotonic_per_rib_index"] = dict(sorted(per_rib_nonmono.items()))
    r1 = np.array(rib1_offsets)
    out["rib1_posterior_end_minus_z_apex_mm"] = {
        "n": int(len(r1)), "mean": float(r1.mean()), "std": float(r1.std()),
        "median": float(np.median(r1)), "p5": float(np.percentile(r1, 5)), "p95": float(np.percentile(r1, 95)),
    }
    se = np.array(spine_err)
    out["spine_midline_abs_error_mm"] = {"n_pairs": int(len(se)), "mean": float(se.mean()),
                                         "std": float(se.std()), "median": float(np.median(se))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
