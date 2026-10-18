"""CLI entry point for Stage-3 hybrid fusion, skeleton extraction, bilateral Viterbi labeling, and RibSnake refinement.

Supports both batch directory processing and single-scan execution::

    # Batch mode on a directory of probability maps
    python -m ribline.scripts.run_stage3 --prob-dir probs_unet --rf-prob-dir probs_scf \\
        --image-dir Data/vol2mm --output-dir centerlines --gt-dir Data/Centerlines_GT_fixed

    # Single-scan mode
    python -m ribline.scripts.run_stage3 --unet-prob scan_unet_prob.nii.gz \\
        --scf-prob scan_scf_prob.nii.gz --ct Data/vol2mm/scan.hdr \\
        --output scan_centerlines.txt
"""

import argparse
import os
import numpy as np
import SimpleITK as sitk

from ribline.data.image_reader import load_image
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher
from ribline.pipeline.pipeline_sota import _save_centerlines, run_sota_pipeline


def run_single_stage3(
    primary_prob_path: str,
    ct_path: str,
    output_txt_path: str,
    secondary_prob_path: str = None,
    atlas_path: str = None,
    alpha: float = 0.75,
    threshold: float = 0.45,
    labeling: str = "viterbi",
    do_refinement: bool = True,
) -> dict:
    """Runs Stage 3 on a single scan and writes the labeled centerlines to ``output_txt_path``."""
    prob_img = load_image(primary_prob_path)
    prob_arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)

    if secondary_prob_path and os.path.exists(secondary_prob_path):
        rf_img = load_image(secondary_prob_path)
        rf_arr = sitk.GetArrayFromImage(rf_img).astype(np.float32)
        prob_arr = np.maximum(prob_arr, float(alpha) * rf_arr)

    ct_img = load_image(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    spacing = ct_img.GetSpacing()
    origin = ct_img.GetOrigin()

    frame = extract_thoracic_frame(ct_arr, spacing=spacing, origin=origin)
    matcher = RibTopologyMatcher(atlas_path=atlas_path)
    atlas = matcher.atlas
    extractor = CenterlineExtractor(threshold=threshold)

    curves = extractor.extract(
        prob_arr,
        spacing=spacing,
        origin=origin,
        min_length=22.0,
        hu_arr=ct_arr,
        frame=frame,
        atlas=atlas,
    )
    if labeling == "zsort":
        labeled = matcher.match_zsort(curves, frame=frame)
    else:
        labeled = matcher.match(curves, frame=frame)

    if do_refinement:
        snake = RibSnake(spacing=spacing)
        final_ribs = snake.refine(
            labeled, ct_arr, origin=origin, prob_np=prob_arr, frame=frame, atlas=atlas
        )
    else:
        final_ribs = labeled

    _save_centerlines(final_ribs, output_txt_path)
    print(f"Saved {len(final_ribs)} labeled ribs to {output_txt_path}")
    return final_ribs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prob-dir", "--unet-prob", "--unet-probs", dest="prob_dir", required=True,
                    help="Primary probability map directory (or single file)")
    ap.add_argument("--rf-prob-dir", "--scf-prob", "--scf-probs", dest="rf_prob_dir", default=None,
                    help="Optional secondary SCF probability directory (or single file)")
    ap.add_argument("--image-dir", "--ct", dest="image_dir", required=True,
                    help="Directory of 3D CT volumes (or single CT file)")
    ap.add_argument("--output-dir", "--out-dir", "--output", "--out", dest="output_dir", required=True,
                    help="Output directory for .txt files (or single .txt/.json path)")
    ap.add_argument("--gt-dir", default=None, help="Optional ground-truth directory for evaluation")
    ap.add_argument("--atlas", default=None, help="Optional path to rib_atlas.joblib")
    ap.add_argument("--split", default=None, help="Optional path to split.json")
    ap.add_argument("--split-key", default="val", choices=["val", "train", "all"],
                    help="Partition in split.json when --split is provided (default: val)")
    ap.add_argument("--rf-weight", "--alpha", dest="rf_weight", type=float, default=0.75,
                    help="Hybrid max-fusion weight for SCF probability map (default: 0.75)")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Skeletonization probability threshold (default: 0.45)")
    ap.add_argument("--labeling", choices=["viterbi", "zsort"], default="viterbi")
    ap.add_argument("--no-refine", action="store_true", help="Disable RibSnake refinement")
    ap.add_argument("--subjects", default=None, help="Optional text file with one subject ID per line")
    ap.add_argument("--match-tol-mm", type=float, default=10.0)
    args = ap.parse_args()

    if os.path.isfile(args.prob_dir):
        run_single_stage3(
            primary_prob_path=args.prob_dir,
            ct_path=args.image_dir,
            output_txt_path=args.output_dir,
            secondary_prob_path=args.rf_prob_dir,
            atlas_path=args.atlas,
            alpha=args.rf_weight,
            threshold=args.threshold,
            labeling=args.labeling,
            do_refinement=not args.no_refine,
        )
        return

    subj = None
    if args.subjects:
        subj = [l.strip() for l in open(args.subjects) if l.strip()]
    elif args.split and os.path.exists(args.split):
        import json
        sp = json.load(open(args.split))
        subj = (
            list(sp.get("train", [])) + list(sp.get("val", []))
            if args.split_key == "all"
            else list(sp[args.split_key])
        )

    target_json = None
    out_dir = args.output_dir
    if out_dir.endswith(".json"):
        target_json = out_dir
        out_dir = os.path.splitext(out_dir)[0] + "_centerlines"

    run_sota_pipeline(
        prob_map_dir=args.prob_dir,
        image_dir=args.image_dir,
        output_dir=out_dir,
        rf_prob_dir=args.rf_prob_dir,
        gt_dir=args.gt_dir,
        do_refinement=not args.no_refine,
        threshold=args.threshold,
        rf_weight=args.rf_weight,
        labeling=args.labeling,
        subjects=subj,
        atlas_path=args.atlas,
        match_tol_mm=args.match_tol_mm,
    )
    if target_json:
        src_eval = os.path.join(out_dir, "evaluation_results_sota.json")
        if os.path.exists(src_eval) and os.path.abspath(src_eval) != os.path.abspath(target_json):
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(target_json)) or ".", exist_ok=True)
            shutil.copyfile(src_eval, target_json)


if __name__ == "__main__":
    main()
