"""Fits the 24-rib statistical atlas (``RibCageStatisticalAtlas``) from a training split.

The packaged ``ribline/configs/rib_atlas.joblib`` was produced with exactly this
procedure on the 47 valid training subjects of ``split.json`` (the 48th training
file, ``Gist_p2_19_Anonymous_0015``, has an empty annotation and is skipped).

Usage::

    python -m ribline.scripts.build_atlas --image-dir Data/vol2mm \
        --gt-dir Data/Centerlines_GT --split split.json --split-key train \
        --out rib_atlas.joblib
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image
from ribline.model.anatomy import RibCageStatisticalAtlas, extract_thoracic_frame


def build_atlas(image_dir: str, gt_dir: str, subjects: list, num_curve_points: int = 80):
    pairs = []
    used = []
    for s in subjects:
        gt_path = os.path.join(gt_dir, f"{s}_interp.xls")
        img_path = os.path.join(image_dir, f"{s}.img")
        if not (os.path.exists(gt_path) and os.path.exists(img_path)):
            continue
        gt = parse_rib_xls(gt_path)
        if not gt:
            continue
        img = load_image(img_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        frame = extract_thoracic_frame(arr, spacing=img.GetSpacing(), origin=img.GetOrigin())
        pairs.append((frame, gt))
        used.append(s)
    atlas = RibCageStatisticalAtlas(num_curve_points=num_curve_points).fit(pairs)
    return atlas, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--split", required=True, help="split.json")
    ap.add_argument("--split-key", default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    subjects = json.load(open(args.split))[args.split_key]
    atlas, used = build_atlas(args.image_dir, args.gt_dir, subjects)
    atlas.save(args.out)
    with open(args.out + ".subjects.json", "w") as f:
        json.dump({"gt_dir": os.path.abspath(args.gt_dir), "subjects": used}, f, indent=2)
    print(f"Fitted atlas on {len(used)} subjects -> {args.out}")


if __name__ == "__main__":
    main()
