"""CLI entry point to train the lightweight 3D U-Net centerline detector.

Can be invoked with a JSON config file or with explicit command-line flags::

    python -m ribline.scripts.train_unet --config experiments/sota/full_config.json

    python -m ribline.scripts.train_unet --image-dir Data/vol2mm \\
        --gt-dir Data/Centerlines_GT_fixed --split split.json \\
        --output-dir experiments/unet --epochs 20
"""

import argparse
import json
import os

from ribline.pipeline.train_unet import train


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
        default="experiments/unet",
        help="Output directory (or .pth file path) for trained U-Net checkpoints",
    )
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    ap.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    ap.add_argument(
        "--samples-per-volume",
        type=int,
        default=8,
        help="Patches sampled per volume and epoch (default: 8)",
    )
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    ap.add_argument("--base-filters", type=int, default=16, help="Base filter channels (default: 16)")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    ap.add_argument("--seed", type=int, default=None, help="Optional random seed")
    ap.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    args = ap.parse_args()

    if args.config:
        train(args.config, resume_path=args.resume)
        return

    target_pth = None
    out_dir = args.output_dir
    if out_dir.endswith(".pth"):
        target_pth = out_dir
        out_dir = os.path.dirname(os.path.abspath(out_dir)) or "."

    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "data_config": {
            "image_directory": args.image_dir,
            "centerline_directory": args.gt_dir,
            "split_file": args.split,
            "augmentation": False,
        },
        "model_config": {
            "base_filters": args.base_filters,
        },
        "training_config": {
            "output_directory": out_dir,
            "samples_per_volume": args.samples_per_volume,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
    }
    cfg_path = os.path.join(out_dir, "unet_train_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    train(cfg_path, resume_path=args.resume)
    if target_pth:
        src = os.path.join(out_dir, "models", "unet_latest.pth")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(target_pth):
            import shutil
            shutil.copyfile(src, target_pth)


if __name__ == "__main__":
    main()
