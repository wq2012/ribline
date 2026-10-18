"""Standalone refinement pipeline for extracted 3D rib centerlines."""

import argparse
import glob
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image, resolve_subject_image_path
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher


def refine_centerlines(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    train_cfg = config["training_config"]
    data_cfg = config["data_config"]

    output_dir = train_cfg["output_directory"]
    centerline_dir = os.path.join(output_dir, "centerlines")
    refined_dir = os.path.join(output_dir, "centerlines_refined")
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(refined_dir, exist_ok=True)

    image_dir = data_cfg["image_directory"]
    matcher = RibTopologyMatcher()
    atlas = matcher.atlas

    txt_files = sorted(glob.glob(os.path.join(centerline_dir, "*.txt")))
    print(f"Found {len(txt_files)} subjects to refine.")

    for txt_path in txt_files:
        subject = os.path.splitext(os.path.basename(txt_path))[0]
        print(f"Refining {subject}...")

        ribs = parse_rib_xls(txt_path)
        if not ribs:
            print("  No ribs found.")
            continue

        img_path = resolve_subject_image_path(image_dir, subject)
        if not os.path.exists(img_path):
            print(f"  Image not found at {img_path}")
            continue

        image = load_image(img_path)
        image_np = sitk.GetArrayFromImage(image).astype(np.float32)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        prob_np = None
        for suffix in ("_prob.nii.gz", "_prob.hdr", "_prob.img"):
            p_cand = os.path.join(pred_dir, f"{subject}{suffix}")
            if os.path.exists(p_cand):
                prob_np = sitk.GetArrayFromImage(load_image(p_cand)).astype(np.float32)
                break

        frame = extract_thoracic_frame(image_np, spacing=spacing, origin=origin)
        snake = RibSnake(spacing=spacing)
        refined_ribs = snake.refine(
            ribs, image_np, origin=origin, prob_np=prob_np, frame=frame, atlas=atlas
        )

        out_path = os.path.join(refined_dir, f"{subject}.txt")
        with open(out_path, "w") as f:
            f.write("RibName\tX\tY\tZ\n")
            for k in sorted(refined_ribs.keys()):
                for p in refined_ribs[k]:
                    f.write(f"{k}\t{p[0]:.4f}\t{p[1]:.4f}\t{p[2]:.4f}\n")
        print(f"  Saved refined centerlines to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    refine_centerlines(args.config)
