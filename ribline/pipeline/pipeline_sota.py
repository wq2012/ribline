"""End-to-end SOTA hybrid 3D rib centerline extraction, labeling, and refinement."""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.image_reader import load_image
from ribline.model.anatomy import extract_thoracic_frame
from ribline.model.metrics import evaluate_subject_ribs
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher


def _load_prob_file(prob_dir: str, subject_id: str) -> sitk.Image:
    for suffix in ("_prob.nii.gz", "_prob.hdr", "_prob.img"):
        cand = os.path.join(prob_dir, f"{subject_id}{suffix}")
        if os.path.exists(cand):
            return load_image(cand)
    raise FileNotFoundError(f"Probability map not found for {subject_id} in {prob_dir}")


def _save_centerlines(centerlines: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("RibName\tX\tY\tZ\n")
        for rib_name in sorted(centerlines.keys()):
            for pt in centerlines[rib_name]:
                f.write(f"{rib_name}\t{pt[0]:.4f}\t{pt[1]:.4f}\t{pt[2]:.4f}\n")


def run_sota_pipeline(
    prob_map_dir: str,
    image_dir: str,
    output_dir: str,
    rf_prob_dir: str = None,
    gt_dir: str = None,
    do_refinement: bool = True,
    threshold: float = 0.45,
    rf_weight: float = 0.75,
    labeling: str = "viterbi",
    subjects: list = None,
    atlas_path: str = None,
    match_tol_mm: float = 10.0,
) -> dict:
    """Runs the full Stage-3 anatomical centerline extraction, labeling, and refinement.

    Args:
        prob_map_dir: Directory with primary ``<subject>_prob.nii.gz`` maps (U-Net or forest).
        image_dir: Directory with the CT volumes (``<subject>.hdr/.img``).
        output_dir: Where ``<subject>.txt`` centerlines (and metrics JSON) are written.
        rf_prob_dir: Optional secondary forest probability directory; when given the
            hybrid map ``max(P_primary, rf_weight * P_rf)`` is used.
        gt_dir: Optional ground-truth directory (``<subject>_interp.xls``) for evaluation.
        do_refinement: Run ``RibSnake`` (apical trimming, template recovery, normal-plane
            centring) after labeling.
        threshold: Probability threshold for skeletonisation.
        rf_weight: Weight ``alpha`` of the forest map in the hybrid max-fusion.
        labeling: ``"viterbi"`` (bilateral Viterbi DP with the atlas) or ``"zsort"``
            (naive craniocaudal counting baseline: curves on each side sorted by mean Z,
            top 12 labelled 1..12).
        subjects: Optional list of subject IDs to restrict processing to.
        atlas_path: Optional path to a ``rib_atlas.joblib`` (default: packaged atlas).
        match_tol_mm: Tolerance used by ``evaluate_subject_ribs``.
    """
    os.makedirs(output_dir, exist_ok=True)

    extractor = CenterlineExtractor(threshold=threshold)
    matcher = RibTopologyMatcher(atlas_path=atlas_path)
    atlas = matcher.atlas
    if labeling not in ("viterbi", "zsort"):
        raise ValueError(f"Unknown labeling mode: {labeling}")

    # Discover subjects from .nii.gz or .hdr probability maps
    found = set()
    for fname in os.listdir(prob_map_dir):
        if fname.endswith("_prob.nii.gz"):
            found.add(fname[: -len("_prob.nii.gz")])
        elif fname.endswith("_prob.hdr"):
            found.add(fname[: -len("_prob.hdr")])
    if subjects is not None:
        found &= set(subjects)
    subjects = sorted(found)

    results_summary = {}
    eval_metrics = {}

    for subject_id in tqdm(subjects, desc="Running SOTA Pipeline"):
        prob_img = _load_prob_file(prob_map_dir, subject_id)
        prob_arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)

        if rf_prob_dir and os.path.exists(rf_prob_dir):
            try:
                rf_img = _load_prob_file(rf_prob_dir, subject_id)
                rf_arr = sitk.GetArrayFromImage(rf_img).astype(np.float32)
                prob_arr = np.maximum(prob_arr, rf_weight * rf_arr)
            except FileNotFoundError:
                pass

        img_path = os.path.join(image_dir, f"{subject_id}.hdr")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{subject_id}.img")
        image_img = load_image(img_path)
        image_arr = sitk.GetArrayFromImage(image_img).astype(np.float32)

        spacing = image_img.GetSpacing()
        origin = image_img.GetOrigin()

        frame = extract_thoracic_frame(image_arr, spacing=spacing, origin=origin)
        centerlines_unlabeled = extractor.extract(
            prob_arr,
            spacing=spacing,
            origin=origin,
            min_length=22.0,
            hu_arr=image_arr,
            frame=frame,
            atlas=atlas,
        )
        if labeling == "zsort":
            centerlines_labeled = matcher.match_zsort(centerlines_unlabeled, frame=frame)
        else:
            centerlines_labeled = matcher.match(centerlines_unlabeled, frame=frame)

        if do_refinement:
            snake = RibSnake(spacing=spacing)
            centerlines_final = snake.refine(
                centerlines_labeled,
                image_arr,
                origin=origin,
                prob_np=prob_arr,
                frame=frame,
                atlas=atlas,
            )
        else:
            centerlines_final = centerlines_labeled

        save_path = os.path.join(output_dir, f"{subject_id}.txt")
        _save_centerlines(centerlines_final, save_path)
        results_summary[subject_id] = len(centerlines_final)

        if gt_dir and os.path.exists(gt_dir):
            gt_path = os.path.join(gt_dir, f"{subject_id}_interp.xls")
            if os.path.exists(gt_path):
                gt_ribs = parse_rib_xls(gt_path)
                if gt_ribs:
                    eval_metrics[subject_id] = evaluate_subject_ribs(
                        gt_ribs=gt_ribs,
                        pred_ribs=centerlines_final,
                        spacing=spacing,
                        match_tol_mm=match_tol_mm,
                    )

    if eval_metrics:
        report_path = os.path.join(output_dir, "sota_evaluation_metrics.json")
        with open(report_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

    print(f"Pipeline Complete. Processed {len(results_summary)} subjects.")
    return eval_metrics if eval_metrics else results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run RibLine Stage-3 SOTA 3D rib centerline extraction, Viterbi labeling, and refinement."
    )
    parser.add_argument(
        "--prob-dir", type=str, required=True, help="Directory containing U-Net/RF _prob files"
    )
    parser.add_argument(
        "--rf-prob-dir", type=str, default=None, help="Optional secondary RF _prob directory for hybrid ensemble"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Directory containing original CT .img/.hdr"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for .txt centerlines"
    )
    parser.add_argument(
        "--gt-dir", type=str, default=None, help="Optional ground-truth Centerlines_GT directory"
    )
    parser.add_argument("--labeling", choices=["viterbi", "zsort"], default="viterbi")
    parser.add_argument("--no-refine", action="store_true", help="Skip RibSnake refinement")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--rf-weight", type=float, default=0.75)
    parser.add_argument("--atlas", type=str, default=None, help="Path to rib_atlas.joblib")
    parser.add_argument(
        "--subjects", type=str, default=None, help="Text file with one subject ID per line"
    )
    parser.add_argument("--match-tol-mm", type=float, default=10.0)
    args = parser.parse_args()
    subj = None
    if args.subjects:
        subj = [l.strip() for l in open(args.subjects) if l.strip()]
    run_sota_pipeline(
        args.prob_dir,
        args.image_dir,
        args.output_dir,
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


if __name__ == "__main__":
    main()

