"""Comprehensive evaluation pipeline for 3D rib centerline extraction and labeling."""

import argparse
import glob
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image, resolve_subject_image_path
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.metrics import (
    chamfer_distance,
    dilated_dice,
    evaluate_subject_ribs,
)
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher


def _discover_prediction_subjects(pred_dir: str) -> list:
    subjects = {}
    if not os.path.exists(pred_dir):
        return []
    for fname in sorted(os.listdir(pred_dir)):
        for suffix in ("_prob.nii.gz", "_prob.hdr"):
            if fname.endswith(suffix):
                sub = fname[: -len(suffix)]
                if sub not in subjects:
                    subjects[sub] = os.path.join(pred_dir, fname)
    return list(subjects.items())


def evaluate(config_path: str, force_extract=False) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)

    data_cfg = config["data_config"]
    train_cfg = config["training_config"]

    pred_dir = os.path.join(train_cfg["output_directory"], "predictions")
    gt_dir = data_cfg["centerline_directory"]
    image_dir = data_cfg["image_directory"]

    centerline_subdir = (
        "centerlines_refined" if force_extract == "refined" else "centerlines"
    )
    pred_centerline_dir = os.path.join(
        train_cfg["output_directory"], centerline_subdir
    )

    subject_items = _discover_prediction_subjects(pred_dir)
    extractor = CenterlineExtractor(threshold=0.45)
    matcher = RibTopologyMatcher()
    atlas = matcher.atlas

    results = {}
    chamfer_scores = []
    labeled_chamfer_scores = []
    dice_scores = []
    labeled_cldice_scores = []
    cldice_scores = []
    acc_all_scores = []

    for subject, pred_path in subject_items:
        img = load_image(pred_path)
        prob_arr = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()

        img_path = resolve_subject_image_path(image_dir, subject)
        ct_arr = None
        if os.path.exists(img_path):
            orig_img = load_image(img_path)
            spacing = orig_img.GetSpacing()
            origin = orig_img.GetOrigin()
            ct_arr = sitk.GetArrayFromImage(orig_img).astype(np.float32)

        pred_centerline_path = os.path.join(pred_centerline_dir, f"{subject}.txt")
        if not force_extract and os.path.exists(pred_centerline_path):
            pred_ribs = parse_rib_xls(pred_centerline_path)
        else:
            frame = (
                extract_thoracic_frame(ct_arr, spacing=spacing, origin=origin)
                if ct_arr is not None
                else None
            )
            curves = extractor.extract(
                prob_arr,
                spacing=spacing,
                origin=origin,
                min_length=22.0,
                hu_arr=ct_arr,
                frame=frame,
                atlas=atlas,
            )
            pred_ribs = matcher.match(curves, frame=frame)
            if ct_arr is not None and frame is not None:
                snake = RibSnake(spacing=spacing)
                pred_ribs = snake.refine(
                    pred_ribs,
                    ct_arr,
                    origin=origin,
                    prob_np=prob_arr,
                    frame=frame,
                    atlas=atlas,
                )

        gt_files = glob.glob(os.path.join(gt_dir, f"{subject}*"))
        gt_file = None
        for g in gt_files:
            if not g.endswith(".img") and not g.endswith(".hdr"):
                gt_file = g
                break
        if not gt_file:
            continue

        gt_ribs = parse_rib_xls(gt_file)
        if not gt_ribs:
            continue

        sub_eval = evaluate_subject_ribs(
            gt_ribs=gt_ribs,
            pred_ribs=pred_ribs,
            shape=prob_arr.shape,
            spacing=spacing,
        )
        if not sub_eval:
            continue

        cd = sub_eval["unlabeled_chamfer_mm"]
        lcd = sub_eval["labeled_chamfer_mm"]
        dice = sub_eval.get("dilated_dice_3mm", 0.0)
        lcld3 = sub_eval["labeled_cldice_3mm"]
        cld3 = sub_eval["unlabeled_cldice_3mm"]
        acc_a = sub_eval["label_acc_all"]
        detected_ribs = int(round(sub_eval["detection_recall"] * sub_eval["num_gt_ribs"]))

        chamfer_scores.append(cd)
        labeled_chamfer_scores.append(lcd)
        dice_scores.append(dice)
        labeled_cldice_scores.append(lcld3)
        cldice_scores.append(cld3)
        acc_all_scores.append(acc_a)

        print(
            f"{subject}: LabeledCD={lcd:.2f}mm, UnlabeledCD={cd:.2f}mm, "
            f"LabeledClDice3mm={lcld3:.3f}, UnlabeledClDice3mm={cld3:.3f}, "
            f"Dice3mm={dice:.3f}, AccA={acc_a:.3f} "
            f"({detected_ribs}/{sub_eval['num_gt_ribs']})"
        )

        results[subject] = {
            "chamfer": cd,
            "dice": dice,
            "detected_ribs": detected_ribs,
            "total_gt_ribs": sub_eval["num_gt_ribs"],
            "pred_rib_count": sub_eval["num_pred_ribs"],
            **sub_eval,
        }

    if chamfer_scores:
        print("-" * 50)
        print(f"Mean Labeled Chamfer (mm):    {np.mean(labeled_chamfer_scores):.4f}")
        print(f"Mean Unlabeled Chamfer (mm):  {np.mean(chamfer_scores):.4f}")
        print(f"Mean Labeled clDice (3mm):    {np.mean(labeled_cldice_scores):.4f}")
        print(f"Mean Unlabeled clDice (3mm):  {np.mean(cldice_scores):.4f}")
        print(f"Mean Dilated Dice (3mm):      {np.mean(dice_scores):.4f}")
        print(f"Mean Label Accuracy (All):    {np.mean(acc_all_scores):.4f}")

    report_path = os.path.join(
        os.path.dirname(pred_dir), "evaluation_metrics.json"
    )
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D rib centerline predictions against ground truth."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force-extract",
        nargs="?",
        const=True,
        default=False,
        help="True/False or 'refined'",
    )
    args = parser.parse_args()
    evaluate(args.config, force_extract=args.force_extract)


if __name__ == "__main__":
    main()

