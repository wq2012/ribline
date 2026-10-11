"""Pass-2 Semantic Context Forest (Auto-Context) inference and 3D rib tracing."""

import argparse
import json
import os
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from ribline.data.dataset import RibDataset
from ribline.data.image_reader import load_image
from ribline.model.anatomy import extract_thoracic_frame, standardize_hu
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher
from ribline.pipeline.inference import (
    load_config,
    predict_volume_forest,
    save_labeled_centerlines,
)


def predict_volume_pass2(
    clf2: RibForest,
    extractor_img: FeatureExtractor,
    extractor_ctx: FeatureExtractor,
    image: sitk.Image,
    prob_map1_img: sitk.Image,
    feat_imgs1: list,
    feat_imgs2: list,
    bone_band_only: bool = True,
) -> np.ndarray:
    """Predicts Pass-2 Auto-Context probability volume."""
    size = image.GetSize()
    width, height, depth = size
    prob_map2_np = np.zeros((depth, height, width), dtype=np.float32)

    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if bone_band_only and float(np.max(img_arr)) > 5.0:
        hu = standardize_hu(img_arr)
        hu_smooth = ndi.gaussian_filter(hu, sigma=1.2)
        roi_mask = hu_smooth > -20.0
        zz, yy, xx = np.where(roi_mask)
        if len(zz) > 0:
            pts_xyz = np.column_stack([xx, yy, zz])
            chunk = 200000
            probs = np.zeros(len(pts_xyz), dtype=np.float32)
            for s in range(0, len(pts_xyz), chunk):
                e = min(len(pts_xyz), s + chunk)
                X_img = extractor_img.extract_at_points(
                    image, pts_xyz[s:e], feature_images=feat_imgs1
                )
                X_ctx = extractor_ctx.extract_at_points(
                    prob_map1_img, pts_xyz[s:e], feature_images=feat_imgs2
                )
                X_batch = np.hstack([X_img, X_ctx])
                probs[s:e] = clf2.predict_proba(X_batch)[:, 1]
            prob_map2_np[zz, yy, xx] = probs
        return prob_map2_np

    for z in range(depth):
        X_slice_img = extractor_img.extract_slice(image, z, feature_images=feat_imgs1)
        X_slice_ctx = extractor_ctx.extract_slice(
            prob_map1_img, z, feature_images=feat_imgs2
        )
        if X_slice_img.shape[0] > 0:
            X_slice = np.hstack([X_slice_img, X_slice_ctx])
            p = clf2.predict_proba(X_slice)
            prob_map2_np[z, :, :] = p[:, 1].reshape(height, width)

    return prob_map2_np


def inference_pass2(config_path: str):
    config = load_config(config_path)
    data_cfg = config["data_config"]
    train_cfg = config["training_config"]

    pass1_dir = train_cfg["pass1_directory"]
    pass2_dir = train_cfg["output_directory"]

    p1_model_path = os.path.join(pass1_dir, "rib_forest.joblib")
    p1_feat_cfg_path = os.path.join(pass1_dir, "feature_config.json")
    p2_model_path = os.path.join(pass2_dir, "rib_forest_pass2.joblib")
    p2_ctx_cfg_path = os.path.join(pass2_dir, "feature_config_ctx.json")
    p2_img_cfg_path = os.path.join(pass2_dir, "feature_config_img.json")

    prediction_dir = os.path.join(pass2_dir, "predictions")
    centerline_dir = os.path.join(pass2_dir, "centerlines")
    refined_dir = os.path.join(pass2_dir, "centerlines_refined")
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(centerline_dir, exist_ok=True)
    os.makedirs(refined_dir, exist_ok=True)

    clf1 = RibForest.from_path(p1_model_path)
    clf2 = RibForest.from_path(p2_model_path)

    cfg_img_path = p2_img_cfg_path if os.path.exists(p2_img_cfg_path) else p1_feat_cfg_path
    with open(cfg_img_path, "r") as f:
        extractor_img = FeatureExtractor.from_config(json.load(f))
    with open(p2_ctx_cfg_path, "r") as f:
        extractor_ctx = FeatureExtractor.from_config(json.load(f))

    image_dir = data_cfg["image_directory"]
    centerline_dir_gt = data_cfg.get("centerline_directory", None)
    select_files = data_cfg.get("select_files", None)

    if "split_file" in data_cfg and data_cfg["split_file"]:
        with open(data_cfg["split_file"], "r") as f:
            splits = json.load(f)
            if "test" in splits:
                select_files = splits["test"]
            elif "val" in splits:
                select_files = splits["val"]

    dataset = RibDataset(
        image_dir, centerline_dir_gt or image_dir, selected_files=select_files
    )
    print(f"Inference Pass 2 on {len(dataset)} subjects.")

    ext_cfg = config.get("extraction_config", {})
    threshold = ext_cfg.get("threshold", 0.45)
    min_length = ext_cfg.get("min_length", 22.0)
    cl_extractor = CenterlineExtractor(threshold=threshold)
    matcher = RibTopologyMatcher()
    atlas = matcher.atlas

    for i in range(len(dataset)):
        subject_id = dataset.subjects[i]
        print(f"Processing {subject_id}...")
        try:
            try:
                image, _, _ = dataset.load_item(i)
            except Exception:
                img_path = os.path.join(image_dir, f"{subject_id}.img")
                image = load_image(img_path)

            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            ct_arr = sitk.GetArrayFromImage(image).astype(np.float32)
            frame = extract_thoracic_frame(ct_arr, spacing=spacing, origin=origin)

            feat_imgs1 = extractor_img.compute_features(image, frame=frame)
            prob_map1_np = predict_volume_forest(
                clf1, extractor_img, image, feature_images=feat_imgs1
            )
            prob_map1_img = sitk.GetImageFromArray(prob_map1_np)
            prob_map1_img.CopyInformation(image)

            feat_imgs2 = extractor_ctx.compute_features(prob_map1_img, frame=frame)
            prob_map2_np = predict_volume_pass2(
                clf2,
                extractor_img,
                extractor_ctx,
                image,
                prob_map1_img,
                feat_imgs1,
                feat_imgs2,
            )

            out_img = sitk.GetImageFromArray(prob_map2_np)
            out_img.CopyInformation(image)
            out_path = os.path.join(prediction_dir, f"{subject_id}_prob.nii.gz")
            sitk.WriteImage(out_img, out_path)

            curves = cl_extractor.extract(
                prob_map2_np,
                spacing=spacing,
                origin=origin,
                min_length=min_length,
                hu_arr=ct_arr,
                frame=frame,
                atlas=atlas,
            )
            labeled = matcher.match(curves, frame=frame)
            save_labeled_centerlines(
                labeled, os.path.join(centerline_dir, f"{subject_id}.txt")
            )

            snake = RibSnake(spacing=spacing)
            refined = snake.refine(
                labeled, ct_arr, origin=origin, prob_np=prob_map2_np, frame=frame, atlas=atlas
            )
            save_labeled_centerlines(
                refined, os.path.join(refined_dir, f"{subject_id}.txt")
            )
            print(f"  Saved {len(refined)} labeled ribs for {subject_id}")

        except Exception as e:
            print(f"Error {subject_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    inference_pass2(args.config)
