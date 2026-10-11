"""Pass-2 Semantic Context Forest (Auto-Context) training pipeline."""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.dataset import RibDataset
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.pipeline.inference import predict_volume_forest


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def train_pass2(config_path: str):
    config = load_config(config_path)
    data_cfg = config["data_config"]
    model_cfg = config["model_config"]
    train_cfg = config["training_config"]

    pass1_dir = train_cfg["pass1_directory"]
    pass1_model_path = os.path.join(pass1_dir, "rib_forest.joblib")
    pass1_feature_cfg_path = os.path.join(pass1_dir, "feature_config.json")

    print(f"Loading Pass 1 Model from {pass1_model_path}...")
    clf1 = RibForest.from_path(pass1_model_path)

    with open(pass1_feature_cfg_path, "r") as f:
        pass1_feat_cfg = json.load(f)
    extractor1 = FeatureExtractor.from_config(pass1_feat_cfg)

    context_cfg = config.get("context_feature_config", {})
    extractor2 = FeatureExtractor.from_config(context_cfg)

    image_dir = data_cfg["image_directory"]
    centerline_dir = data_cfg["centerline_directory"]
    select_files = data_cfg.get("select_files", None)

    if "split_file" in data_cfg and data_cfg["split_file"]:
        with open(data_cfg["split_file"], "r") as f:
            splits = json.load(f)
            if "train" in splits:
                select_files = splits["train"]

    dataset = RibDataset(image_dir, centerline_dir, selected_files=select_files)
    print(f"Pass 2 Training on {len(dataset)} subjects.")

    output_dir = train_cfg["output_directory"]
    os.makedirs(output_dir, exist_ok=True)

    X_all = []
    y_all = []
    samples_per_vol = train_cfg.get("samples_per_volume", 2000)

    for i in range(len(dataset)):
        print(f"Processing {dataset.subjects[i]}...")
        try:
            image, centerlines, _ = dataset.load_item(i)

            if data_cfg.get("augmentation", False):
                image, centerlines = dataset.augment_item(image, centerlines)

            pos_indices, neg_indices = dataset.get_samples_from_data(
                image, centerlines, num_samples=samples_per_vol
            )
            if not pos_indices:
                continue

            frame = None
            if extractor1.use_anatomical_features or extractor2.use_anatomical_features:
                ct_arr = sitk.GetArrayFromImage(image).astype(np.float32)
                frame = extract_thoracic_frame(
                    ct_arr, spacing=image.GetSpacing(), origin=image.GetOrigin()
                )

            feat_imgs1 = extractor1.compute_features(image, frame=frame)
            prob_map_np = predict_volume_forest(
                clf1, extractor1, image, feature_images=feat_imgs1
            )

            prob_map_img = sitk.GetImageFromArray(prob_map_np)
            prob_map_img.CopyInformation(image)

            feat_imgs2 = extractor2.compute_features(prob_map_img, frame=frame)

            pos_pts = [p[0] for p in pos_indices]
            neg_pts = [p[0] for p in neg_indices]

            X_pos_img = extractor1.extract_at_points(
                image, pos_pts, feature_images=feat_imgs1
            )
            X_neg_img = extractor1.extract_at_points(
                image, neg_pts, feature_images=feat_imgs1
            )

            X_pos_ctx = extractor2.extract_at_points(
                prob_map_img, pos_pts, feature_images=feat_imgs2
            )
            X_neg_ctx = extractor2.extract_at_points(
                prob_map_img, neg_pts, feature_images=feat_imgs2
            )

            X_pos = np.hstack([X_pos_img, X_pos_ctx])
            X_neg = np.hstack([X_neg_img, X_neg_ctx])

            X_all.append(X_pos)
            X_all.append(X_neg)

            y_all.append(np.ones(len(pos_pts), dtype=np.int32))
            y_all.append(np.zeros(len(neg_pts), dtype=np.int32))

        except Exception as e:
            print(f"Error {dataset.subjects[i]}: {e}")
            import traceback

            traceback.print_exc()

    if not X_all:
        print("No Data.")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print(f"Training Pass 2 on {len(X)} samples, {X.shape[1]} features.")

    clf2 = RibForest(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        n_jobs=model_cfg.get("n_jobs", -1),
    )
    clf2.fit(X, y)

    clf2.save(os.path.join(output_dir, "rib_forest_pass2.joblib"))

    with open(os.path.join(output_dir, "feature_config_img.json"), "w") as f:
        json.dump(extractor1.get_feature_config(), f, indent=2)
    with open(os.path.join(output_dir, "feature_config_ctx.json"), "w") as f:
        json.dump(extractor2.get_feature_config(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_pass2(args.config)
