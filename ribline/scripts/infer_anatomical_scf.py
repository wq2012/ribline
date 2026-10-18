"""CLI entry point to run anatomically anchored SCF probability map inference.

Supports single-volume inference (`--input` / `--output`), directory/split
inference (`--image-dir` / `--output-dir`), or config-driven inference (`--config`)::

    python -m ribline.scripts.infer_anatomical_scf --input scan.nii.gz \\
        --checkpoint pretrained_models/rib_forest_anatomical.joblib \\
        --output scan_scf_prob.nii.gz

    python -m ribline.scripts.infer_anatomical_scf --image-dir Data/vol2mm \\
        --checkpoint pretrained_models/rib_forest_anatomical.joblib \\
        --split split.json --split-key val --output-dir probs_scf --prob-only
"""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.image_reader import load_image, resolve_subject_image_path, strip_image_extension
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.pipeline.inference import inference as run_config_inference, predict_volume_forest


def _load_extractor(feature_config_path: str = None, checkpoint_path: str = None) -> FeatureExtractor:
    if feature_config_path is None and checkpoint_path is not None:
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        for cand_name in ("feature_config_anatomical.json", "feature_config.json"):
            cand = os.path.join(ckpt_dir, cand_name)
            if os.path.exists(cand):
                feature_config_path = cand
                break
    if feature_config_path and os.path.exists(feature_config_path):
        with open(feature_config_path, "r") as f:
            return FeatureExtractor.from_config(json.load(f))
    return FeatureExtractor(
        scales=[1.0, 2.0, 4.0],
        rsid_params={"num_pairs": 250, "max_offset": 12, "seed": 42},
        use_anatomical_features=True,
        use_standard_hu=True,
    )


def infer_single_volume(
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    feature_config_path: str = None,
):
    clf = RibForest.load(checkpoint_path)
    extractor = _load_extractor(feature_config_path, checkpoint_path)
    image = load_image(input_path)
    ct_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    frame = extract_thoracic_frame(
        ct_arr, spacing=image.GetSpacing(), origin=image.GetOrigin()
    )
    feature_images = extractor.compute_features(image, frame=frame)
    prob_np = predict_volume_forest(
        clf, extractor, image, feature_images=feature_images
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_img = sitk.GetImageFromArray(prob_np)
    out_img.CopyInformation(image)
    sitk.WriteImage(out_img, output_path)
    print(f"Saved SCF probability volume to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="Optional JSON config for ribline.pipeline.inference")
    ap.add_argument("--input", default=None, help="Path to a single 3D CT volume (.nii.gz, .img/.hdr)")
    ap.add_argument("--image-dir", default=None, help="Directory of 3D CT volumes for batch inference")
    ap.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint",
        default="pretrained_models/rib_forest_anatomical.joblib",
        help="Path to trained RibForest .joblib checkpoint",
    )
    ap.add_argument(
        "--feature-config",
        default=None,
        help="Optional path to feature_config_anatomical.json",
    )
    ap.add_argument("--atlas", default=None, help="Optional path to rib_atlas.joblib")
    ap.add_argument("--output", "--out", dest="output", default=None, help="Output file path (single volume) or directory")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", default=None, help="Output directory for <subject>_prob.nii.gz maps")
    ap.add_argument("--split", default=None, help="Path to split.json or partition name (val|train|all)")
    ap.add_argument("--split-key", default="val", choices=["val", "train", "all"])
    ap.add_argument("--prob-only", action="store_true", help="Only write probability maps")
    ap.add_argument("--skip-existing", action="store_true", help="Skip existing probability maps")
    args = ap.parse_args()

    out_target = args.output_dir or args.output

    if args.config:
        split_name = args.split if args.split in (None, "val", "train", "all") else args.split_key
        run_config_inference(
            args.config,
            split=split_name,
            output_dir=out_target,
            prob_only=args.prob_only,
            skip_existing=args.skip_existing,
        )
        return

    if args.input:
        if not out_target:
            ap.error("--output is required when --input is specified")
        infer_single_volume(args.input, args.checkpoint, out_target, args.feature_config)
        return

    if not args.image_dir or not out_target:
        ap.error("Provide either --config, (--input and --output), or (--image-dir and --output-dir)")

    os.makedirs(out_target, exist_ok=True)
    clf = RibForest.load(args.checkpoint)
    extractor = _load_extractor(args.feature_config, args.checkpoint)

    if args.split and os.path.exists(args.split):
        sp = json.load(open(args.split))
        subjects = (
            list(sp.get("train", [])) + list(sp.get("val", []))
            if args.split_key == "all"
            else list(sp[args.split_key])
        )
    else:
        seen = set()
        subjects = []
        for fn in sorted(os.listdir(args.image_dir)):
            sid, _ = strip_image_extension(fn)
            if sid and sid not in seen:
                seen.add(sid)
                subjects.append(sid)

    for sid in subjects:
        out_path = os.path.join(out_target, f"{sid}_prob.nii.gz")
        if args.skip_existing and os.path.exists(out_path):
            continue
        img_path = resolve_subject_image_path(args.image_dir, sid)
        if not os.path.exists(img_path):
            continue
        infer_single_volume(img_path, args.checkpoint, out_path, args.feature_config)


if __name__ == "__main__":
    main()
