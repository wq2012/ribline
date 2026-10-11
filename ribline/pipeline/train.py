"""Pass-1 Semantic Context Forest training pipeline."""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.dataset import RibDataset
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def train(config_path: str):
    config = load_config(config_path)
    data_cfg = config["data_config"]
    model_cfg = config["model_config"]
    train_cfg = config["training_config"]

    image_dir = data_cfg["image_directory"]
    centerline_dir = data_cfg["centerline_directory"]
    output_dir = train_cfg["output_directory"]
    os.makedirs(output_dir, exist_ok=True)

    select_files = data_cfg.get("select_files", None)
    if "split_file" in data_cfg and data_cfg["split_file"]:
        with open(data_cfg["split_file"], "r") as f:
            splits = json.load(f)
            if "train" in splits:
                select_files = splits["train"]

    dataset = RibDataset(image_dir, centerline_dir, selected_files=select_files)
    print(f"Found {len(dataset)} training subjects.")

    feature_cfg = config.get("feature_config", {})
    extractor = FeatureExtractor.from_config(feature_cfg)

    with open(os.path.join(output_dir, "feature_config.json"), "w") as f:
        json.dump(extractor.get_feature_config(), f, indent=2)

    X_all = []
    y_all = []
    samples_per_vol = train_cfg.get("samples_per_volume", 2000)

    for i in range(len(dataset)):
        print(f"Processing subject {i + 1}/{len(dataset)}: {dataset.subjects[i]}")
        try:
            image, centerlines, _ = dataset.load_item(i)

            if data_cfg.get("augmentation", False):
                image, centerlines = dataset.augment_item(image, centerlines)

            pos_indices, neg_indices = dataset.get_samples_from_data(
                image, centerlines, num_samples=samples_per_vol
            )

            if not pos_indices:
                print(f"Warning: No positive samples for subject {dataset.subjects[i]}")
                continue

            frame = None
            if extractor.use_anatomical_features:
                ct_arr = sitk.GetArrayFromImage(image).astype(np.float32)
                frame = extract_thoracic_frame(
                    ct_arr, spacing=image.GetSpacing(), origin=image.GetOrigin()
                )

            feature_images = extractor.compute_features(image, frame=frame)

            pos_pts = [p[0] for p in pos_indices]
            neg_pts = [p[0] for p in neg_indices]

            X_pos = extractor.extract_at_points(
                image, pos_pts, feature_images=feature_images
            )
            X_neg = extractor.extract_at_points(
                image, neg_pts, feature_images=feature_images
            )

            y_pos = np.ones(len(pos_pts), dtype=np.int32)
            y_neg = np.zeros(len(neg_pts), dtype=np.int32)

            X_all.append(X_pos)
            X_all.append(X_neg)
            y_all.append(y_pos)
            y_all.append(y_neg)

        except Exception as e:
            print(f"Error processing subject {i}: {e}")
            import traceback

            traceback.print_exc()

    if not X_all:
        print("No training data collected!")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")

    clf = RibForest(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        n_jobs=model_cfg.get("n_jobs", -1),
    )

    clf.fit(X, y)

    model_path = os.path.join(output_dir, "rib_forest.joblib")
    clf.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    args = parser.parse_args()
    train(args.config)
