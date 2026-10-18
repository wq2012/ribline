"""CLI entry point to train the anatomically anchored Semantic Context Forest (SCF).

Can be invoked with a JSON config file or with explicit command-line flags::

    python -m ribline.scripts.train_anatomical_scf --image-dir Data/vol2mm \\
        --gt-dir Data/Centerlines_GT_fixed --split split.json \\
        --output-dir experiments/scf_anatomical

    python -m ribline.scripts.train_anatomical_scf --config scf_config.json
"""

import argparse
import json
import os
import tempfile

from ribline.pipeline.train import train


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="Optional JSON configuration path")
    ap.add_argument("--image-dir", default="Data/vol2mm", help="Directory containing 3D CT volumes")
    ap.add_argument(
        "--gt-dir",
        default="Data/Centerlines_GT_fixed",
        help="Directory containing ground-truth centerline .xls files",
    )
    ap.add_argument("--split", default="split.json", help="Path to split.json")
    ap.add_argument(
        "--output-dir",
        "--out-dir",
        "--out",
        "--output",
        dest="output_dir",
        default="experiments/scf_anatomical",
        help="Output directory (or .joblib file path) for rib_forest.joblib and feature_config.json",
    )
    ap.add_argument("--n-estimators", type=int, default=150, help="Number of trees (default: 150)")
    ap.add_argument("--max-depth", type=int, default=24, help="Maximum tree depth (default: 24)")
    ap.add_argument(
        "--samples-per-volume",
        type=int,
        default=50000,
        help="Training voxel samples per scan (default: 50000)",
    )
    ap.add_argument("--num-pairs", type=int, default=250, help="Number of RSID offset pairs (default: 250)")
    ap.add_argument("--max-offset", type=int, default=12, help="RSID max voxel offset (default: 12)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for RSID offsets (default: 42)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel CPU workers (default: -1)")
    args = ap.parse_args()

    if args.config:
        train(args.config)
        return

    target_joblib = None
    out_dir = args.output_dir
    if out_dir.endswith(".joblib"):
        target_joblib = out_dir
        out_dir = os.path.dirname(os.path.abspath(out_dir)) or "."

    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "data_config": {
            "image_directory": args.image_dir,
            "centerline_directory": args.gt_dir,
            "split_file": args.split if os.path.exists(args.split) else None,
            "augmentation": False,
        },
        "model_config": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "n_jobs": args.n_jobs,
        },
        "training_config": {
            "output_directory": out_dir,
            "samples_per_volume": args.samples_per_volume,
        },
        "feature_config": {
            "scales": [1.0, 2.0, 4.0],
            "rsid_params": {
                "num_pairs": args.num_pairs,
                "max_offset": args.max_offset,
                "seed": args.seed,
            },
            "use_anatomical_features": True,
            "use_standard_hu": True,
        },
        "extraction_config": {
            "threshold": 0.45,
            "min_length": 22.0,
        },
    }
    cfg_path = os.path.join(out_dir, "scf_train_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    train(cfg_path)
    if target_joblib:
        src = os.path.join(out_dir, "rib_forest.joblib")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(target_joblib):
            import shutil
            shutil.copyfile(src, target_joblib)


if __name__ == "__main__":
    main()
