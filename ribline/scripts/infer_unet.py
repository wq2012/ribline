"""CLI entry point to run 3D U-Net sliding-window probability map inference.

Supports single-volume inference (`--input` / `--output`), directory/split
inference (`--image-dir` / `--output-dir`), or config-driven inference (`--config`)::

    python -m ribline.scripts.infer_unet --input scan.nii.gz \\
        --checkpoint pretrained_models/rib_unet_3d.pth --output scan_unet_prob.nii.gz

    python -m ribline.scripts.infer_unet --config experiments/sota/full_config.json \\
        --checkpoint pretrained_models/rib_unet_3d.pth --output probs_unet --split val
"""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.image_reader import load_image, resolve_subject_image_path, strip_image_extension
from ribline.model.unet.model import RibUNet3D
from ribline.pipeline.inference_unet import run_inference, sliding_window_inference


def infer_single_volume(input_path: str, checkpoint_path: str, output_path: str):
    model = RibUNet3D.load(checkpoint_path, map_location="cpu")
    image = load_image(input_path)
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    prob_arr = sliding_window_inference(model, img_arr)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prob_img = sitk.GetImageFromArray(prob_arr)
    prob_img.CopyInformation(image)
    sitk.WriteImage(prob_img, output_path)
    print(f"Saved U-Net probability volume to {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None, help="Optional JSON config for ribline.pipeline.inference_unet")
    ap.add_argument("--input", default=None, help="Path to a single 3D CT volume (.nii.gz, .img/.hdr)")
    ap.add_argument("--image-dir", default=None, help="Directory of 3D CT volumes for batch inference")
    ap.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint",
        default="pretrained_models/rib_unet_3d.pth",
        help="Path to trained 3D U-Net .pth checkpoint",
    )
    ap.add_argument("--output", "--out", dest="output", default=None, help="Output file path (single volume) or directory")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", default=None, help="Output directory for <subject>_prob.nii.gz maps")
    ap.add_argument("--split", default="val", help="split.json partition (val|train|all) or path to split.json")
    ap.add_argument("--split-key", default="val", choices=["val", "train", "all"])
    ap.add_argument("--subjects", default=None, help="Optional text file with one subject ID per line")
    ap.add_argument("--skip-existing", action="store_true", help="Skip existing probability maps")
    args = ap.parse_args()

    out_target = args.output_dir or args.output
    subj = None
    if args.subjects:
        subj = [l.strip() for l in open(args.subjects) if l.strip()]

    if args.config:
        split_name = args.split if args.split in ("val", "train", "all") else args.split_key
        run_inference(
            args.config,
            args.checkpoint,
            out_target,
            split=split_name,
            subjects=subj,
            skip_existing=args.skip_existing,
        )
        return

    if args.input:
        if not out_target:
            ap.error("--output is required when --input is specified")
        infer_single_volume(args.input, args.checkpoint, out_target)
        return

    if not args.image_dir or not out_target:
        ap.error("Provide either --config, (--input and --output), or (--image-dir and --output)")

    os.makedirs(out_target, exist_ok=True)
    if subj is None:
        if os.path.exists(args.split):
            sp = json.load(open(args.split))
            subj = (
                list(sp.get("train", [])) + list(sp.get("val", []))
                if args.split_key == "all"
                else list(sp[args.split_key])
            )
        else:
            seen = set()
            subj = []
            for fn in sorted(os.listdir(args.image_dir)):
                sid, _ = strip_image_extension(fn)
                if sid and sid not in seen:
                    seen.add(sid)
                    subj.append(sid)

    model = RibUNet3D.load(args.checkpoint, map_location="cpu")
    for sid in subj:
        out_path = os.path.join(out_target, f"{sid}_prob.nii.gz")
        if args.skip_existing and os.path.exists(out_path):
            continue
        img_path = resolve_subject_image_path(args.image_dir, sid)
        if not os.path.exists(img_path):
            continue
        image = load_image(img_path)
        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
        prob_arr = sliding_window_inference(model, img_arr)
        prob_img = sitk.GetImageFromArray(prob_arr)
        prob_img.CopyInformation(image)
        sitk.WriteImage(prob_img, out_path)


if __name__ == "__main__":
    main()
