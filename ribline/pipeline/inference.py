"""Pass-1 Semantic Context Forest inference and 3D rib centerline extraction."""

import argparse
import json
import os
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from ribline.data.dataset import RibDataset
from ribline.data.image_reader import load_image
from ribline.model.anatomy import (
    RibCageStatisticalAtlas,
    extract_thoracic_frame,
    standardize_hu,
)
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def predict_volume_forest(
    clf: RibForest,
    extractor: FeatureExtractor,
    image: sitk.Image,
    feature_images: list = None,
    bone_band_only: bool = True,
) -> np.ndarray:
    """Predicts a 3D probability volume using a trained RibForest classifier."""
    if feature_images is None:
        feature_images = extractor.compute_features(image)

    size = image.GetSize()  # (x, y, z)
    width, height, depth = size
    prob_map_np = np.zeros((depth, height, width), dtype=np.float32)

    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if bone_band_only and float(np.max(img_arr)) > 5.0:
        hu = standardize_hu(img_arr)
        hu_smooth = ndi.gaussian_filter(hu, sigma=1.2)
        roi_mask = hu_smooth > -20.0
        zz, yy, xx = np.where(roi_mask)
        if len(zz) > 0:
            pts_xyz = np.column_stack([xx, yy, zz])
            # Predict in chunks of 250k voxels to bound peak memory
            chunk = 250000
            probs = np.zeros(len(pts_xyz), dtype=np.float32)
            for s in range(0, len(pts_xyz), chunk):
                e = min(len(pts_xyz), s + chunk)
                X_batch = extractor.extract_at_points(
                    image, pts_xyz[s:e], feature_images=feature_images
                )
                probs[s:e] = clf.predict_proba(X_batch)[:, 1]
            prob_map_np[zz, yy, xx] = probs
        return prob_map_np

    for z in range(depth):
        X_slice = extractor.extract_slice(image, z, feature_images=feature_images)
        if X_slice.shape[0] == 0:
            continue
        p = clf.predict_proba(X_slice)
        prob_map_np[z, :, :] = p[:, 1].reshape(height, width)

    return prob_map_np


def save_labeled_centerlines(labeled_ribs: dict, txt_path: str):
    """Writes labeled 3D rib polylines in tab-separated format."""
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("RibName\tX\tY\tZ\n")
        for rib_name in sorted(labeled_ribs.keys()):
            line = labeled_ribs[rib_name]
            for pt in line:
                f.write(f"{rib_name}\t{pt[0]:.4f}\t{pt[1]:.4f}\t{pt[2]:.4f}\n")


def inference(config_path: str, split: str = None, output_dir: str = None,
              prob_only: bool = False, skip_existing: bool = False):
    """Runs forest inference.

    Args:
        config_path: JSON config with ``data_config`` / ``training_config``.
        split: Optional ``split.json`` partition (``val``, ``train`` or ``all``).
            Default keeps the historical behaviour (``test`` > ``val`` > ``train``).
        output_dir: Optional directory for ``_prob.nii.gz`` maps (default:
            ``<output_directory>/predictions``).
        prob_only: If True, only probability maps are written (no Stage 3).
        skip_existing: Skip subjects whose probability map already exists.
    """
    config = load_config(config_path)
    data_cfg = config["data_config"]
    train_cfg = config["training_config"]

    image_dir = data_cfg["image_directory"]
    centerline_dir_gt = data_cfg.get("centerline_directory", None)
    model_dir = train_cfg["output_directory"]
    model_path = os.path.join(model_dir, "rib_forest.joblib")
    prediction_dir = output_dir or os.path.join(model_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    clf = RibForest.from_path(model_path)

    select_files = data_cfg.get("select_files", None)
    if "split_file" in data_cfg and data_cfg["split_file"]:
        with open(data_cfg["split_file"], "r") as f:
            splits = json.load(f)
        if split == "all":
            select_files = list(splits.get("train", [])) + list(splits.get("val", []))
        elif split is not None:
            select_files = list(splits[split])
        elif "test" in splits:
            select_files = splits["test"]
        elif "val" in splits:
            select_files = splits["val"]
        elif "train" in splits:
            select_files = splits["train"]

    dataset = RibDataset(
        image_dir,
        centerline_dir_gt if centerline_dir_gt else image_dir,
        selected_files=select_files,
    )
    print(f"Inference on {len(dataset)} subjects.")

    feature_config_path = os.path.join(model_dir, "feature_config.json")
    if os.path.exists(feature_config_path):
        with open(feature_config_path, "r") as f:
            feature_cfg = json.load(f)
        extractor = FeatureExtractor.from_config(feature_cfg)
    else:
        extractor = FeatureExtractor(scales=[1.0, 2.0, 4.0])

    ext_cfg = config.get("extraction_config", {})
    threshold = ext_cfg.get("threshold", 0.45)
    min_length = ext_cfg.get("min_length", 22.0)

    cl_extractor = CenterlineExtractor(threshold=threshold)
    matcher = RibTopologyMatcher()
    atlas = matcher.atlas

    centerline_out_dir = os.path.join(model_dir, "centerlines")
    refined_out_dir = os.path.join(model_dir, "centerlines_refined")
    if not prob_only:
        os.makedirs(centerline_out_dir, exist_ok=True)
        os.makedirs(refined_out_dir, exist_ok=True)

    for i in range(len(dataset)):
        subject_id = dataset.subjects[i]
        out_path = os.path.join(prediction_dir, f"{subject_id}_prob.nii.gz")
        if skip_existing and os.path.exists(out_path):
            continue
        print(f"Processing {subject_id}...")
        try:
            try:
                image, _, _ = dataset.load_item(i)
            except Exception:
                img_path = os.path.join(image_dir, f"{subject_id}.img")
                image = load_image(img_path)

            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            ct_arr = sitk.GetArrayFromImage(image).astype(np.float32)
            frame = extract_thoracic_frame(ct_arr, spacing=spacing, origin=origin)

            feature_images = extractor.compute_features(image, frame=frame)
            prob_map_np = predict_volume_forest(
                clf, extractor, image, feature_images=feature_images
            )

            out_img = sitk.GetImageFromArray(prob_map_np)
            out_img.CopyInformation(image)
            sitk.WriteImage(out_img, out_path)
            if prob_only:
                continue

            curves = cl_extractor.extract(
                prob_map_np,
                spacing=spacing,
                origin=origin,
                min_length=min_length,
                hu_arr=ct_arr,
                frame=frame,
                atlas=atlas,
            )
            labeled = matcher.match(curves, frame=frame)
            save_labeled_centerlines(
                labeled, os.path.join(centerline_out_dir, f"{subject_id}.txt")
            )

            snake = RibSnake(spacing=spacing)
            refined = snake.refine(
                labeled, ct_arr, origin=origin, prob_np=prob_map_np, frame=frame, atlas=atlas
            )
            save_labeled_centerlines(
                refined, os.path.join(refined_out_dir, f"{subject_id}.txt")
            )
            print(f"Saved {len(refined)} labeled ribs for {subject_id}")

        except Exception as e:
            print(f"Error inference {subject_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    parser.add_argument("--split", default=None, choices=[None, "val", "train", "all"],
                        help="split.json partition to run on (default: test>val>train)")
    parser.add_argument("--output-dir", default=None, help="Directory for _prob.nii.gz maps")
    parser.add_argument("--prob-only", action="store_true", help="Only write probability maps")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    inference(args.config, split=args.split, output_dir=args.output_dir,
              prob_only=args.prob_only, skip_existing=args.skip_existing)
