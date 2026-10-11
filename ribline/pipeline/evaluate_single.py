"""Quick single-subject sanity check: skeletonise one probability map and report unlabeled Chamfer.

Usage::

    python -m ribline.pipeline.evaluate_single --subject <id> --prob-dir <dir with <id>_prob.nii.gz> \
        --image-dir Data/vol2mm --gt-dir Data/Centerlines_GT [--min-length 30] [--threshold 0.5]

All paths and the subject are explicit arguments (nothing is hard-coded).
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image
from ribline.model.metrics import chamfer_distance
from ribline.model.reconstruction import CenterlineExtractor


def evaluate_single(subject_id, prob_dir, image_dir, gt_dir, min_length=30.0, threshold=0.5):
    pred_path = os.path.join(prob_dir, f"{subject_id}_prob.nii.gz")
    gt_files = [g for g in glob.glob(os.path.join(gt_dir, f"{subject_id}*"))
                if not g.endswith(('.img', '.hdr'))]
    if not gt_files:
        raise FileNotFoundError(f"No GT file for {subject_id} in {gt_dir}")
    gt_ribs = parse_rib_xls(gt_files[0])
    all_gt_points = np.vstack(list(gt_ribs.values()))

    orig_img = load_image(os.path.join(image_dir, f"{subject_id}.img"))
    spacing = orig_img.GetSpacing()
    origin = orig_img.GetOrigin()

    prob_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    extractor = CenterlineExtractor(threshold=threshold)
    pred_lines = extractor.extract(prob_arr, spacing=spacing, origin=origin, min_length=min_length)

    print(f"Eval: {subject_id}  (min_length={min_length}, threshold={threshold})")
    print(f"Extracted {len(pred_lines)} components.")
    lengths = [float(np.sum(np.linalg.norm(np.diff(l, axis=0), axis=1))) for l in pred_lines if len(l) > 1]
    if lengths:
        print(f"Lengths (min/avg/max): {min(lengths):.1f} / {np.mean(lengths):.1f} / {max(lengths):.1f}")
    all_pred_points = np.vstack(pred_lines) if pred_lines else np.empty((0, 3))
    if len(all_pred_points) > 0:
        print(f"Unlabeled Chamfer: {chamfer_distance(all_gt_points, all_pred_points):.4f} mm")
    else:
        print("Unlabeled Chamfer: Inf (no paths found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--min-length", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate_single(args.subject, args.prob_dir, args.image_dir, args.gt_dir,
                    args.min_length, args.threshold)
