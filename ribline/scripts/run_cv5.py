"""CLI entry point for the 5-fold cross-validation across all 59 annotated SCR scans.

For each fold k in 0..4, this script fits the 24-rib statistical atlas, the
anatomically anchored SCF, and the 3D U-Net on the fold's training subjects,
evaluates the frozen Stage-3 pipeline on the held-out fold, and aggregates
all 59 out-of-fold subject predictions into a unified benchmark JSON report::

    # Run a single fold (0..4) or all 5 folds and pool results
    python -m ribline.scripts.run_cv5 --image-dir Data/vol2mm \\
        --gt-dir Data/Centerlines_GT_fixed --folds experiments/cv5/folds.json \\
        --output-dir experiments/cv5

    # Pool existing per-fold Stage-3 outputs into benchmark_cv5.json
    python -m ribline.scripts.run_cv5 --gt-dir Data/Centerlines_GT_fixed \\
        --folds experiments/cv5/folds.json --output-dir experiments/cv5 --eval-only
"""

import argparse
import json
import os
import shutil

from ribline.pipeline.inference import inference as run_scf_inference
from ribline.pipeline.inference_unet import run_inference as run_unet_inference
from ribline.pipeline.pipeline_sota import run_sota_pipeline
from ribline.pipeline.train import train as train_scf
from ribline.pipeline.train_unet import train as train_unet
from ribline.scripts.benchmark_eval import (
    SUBJECT_KEYS,
    evaluate_method,
    paired_tests,
    summarize,
)
from ribline.scripts.build_atlas import build_atlas

METHODS = [
    {"name": "4_scf_anat_stage3", "prob": "scf_anat", "labeling": "viterbi", "refine": True},
    {"name": "7_unet_stage3", "prob": "unet", "labeling": "viterbi", "refine": True},
    {
        "name": "8_hybrid_viterbi_nosnake",
        "prob": "unet",
        "rf_prob": "scf_anat",
        "labeling": "viterbi",
        "refine": False,
    },
    {
        "name": "9_hybrid_full",
        "prob": "unet",
        "rf_prob": "scf_anat",
        "labeling": "viterbi",
        "refine": True,
    },
]


def run_single_fold(
    fold_idx: int,
    folds: list,
    image_dir: str,
    gt_dir: str,
    cv_root: str,
    epochs: int = 20,
    scf_samples: int = 50000,
    n_jobs: int = -1,
):
    fold_dir = os.path.join(cv_root, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    val_subj = sorted(folds[fold_idx])
    train_subj = sorted(
        s for i, flist in enumerate(folds) if i != fold_idx for s in flist
    )
    split_path = os.path.join(fold_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_subj, "val": val_subj}, f, indent=2)

    atlas_path = os.path.join(fold_dir, "rib_atlas.joblib")
    if not os.path.exists(atlas_path):
        atlas, _ = build_atlas(image_dir, gt_dir, train_subj)
        atlas.save(atlas_path)

    scf_cfg_path = os.path.join(fold_dir, "scf_config.json")
    scf_cfg = {
        "data_config": {
            "image_directory": image_dir,
            "centerline_directory": gt_dir,
            "split_file": split_path,
            "augmentation": False,
        },
        "model_config": {"n_estimators": 150, "max_depth": 24, "n_jobs": n_jobs},
        "training_config": {
            "output_directory": os.path.join(fold_dir, "scf"),
            "samples_per_volume": scf_samples,
        },
        "feature_config": {
            "scales": [1.0, 2.0, 4.0],
            "rsid_params": {"num_pairs": 250, "max_offset": 12, "seed": 42},
            "use_anatomical_features": True,
            "use_standard_hu": True,
        },
        "extraction_config": {"threshold": 0.45, "min_length": 22.0},
    }
    with open(scf_cfg_path, "w") as f:
        json.dump(scf_cfg, f, indent=2)
    if not os.path.exists(os.path.join(fold_dir, "scf", "rib_forest.joblib")):
        train_scf(scf_cfg_path)

    cache_dir = os.path.join(cv_root, "mask_cache_shared")
    os.makedirs(cache_dir, exist_ok=True)
    unet_cfg_path = os.path.join(fold_dir, "unet_config.json")
    unet_cfg = {
        "data_config": {
            "image_directory": image_dir,
            "centerline_directory": gt_dir,
            "split_file": split_path,
            "augmentation": False,
        },
        "model_config": {"base_filters": 16},
        "training_config": {
            "output_directory": os.path.join(fold_dir, "unet"),
            "samples_per_volume": 8,
            "epochs": epochs,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "num_workers": 2,
            "mask_cache_dir": cache_dir,
            "seed": 0,
        },
    }
    with open(unet_cfg_path, "w") as f:
        json.dump(unet_cfg, f, indent=2)
    ckpt = os.path.join(fold_dir, "unet", "models", "unet_latest.pth")
    if not os.path.exists(ckpt):
        train_unet(unet_cfg_path)

    probs_unet = os.path.join(fold_dir, "probs_unet")
    probs_scf = os.path.join(fold_dir, "probs_scf")
    run_unet_inference(
        unet_cfg_path, ckpt, probs_unet, split="val", skip_existing=True
    )
    run_scf_inference(
        scf_cfg_path,
        split="val",
        output_dir=probs_scf,
        prob_only=True,
        skip_existing=True,
    )

    prob_dirs = {"unet": probs_unet, "scf_anat": probs_scf}
    stage3_root = os.path.join(fold_dir, "stage3")
    for m in METHODS:
        run_sota_pipeline(
            prob_dirs[m["prob"]],
            image_dir,
            os.path.join(stage3_root, m["name"]),
            rf_prob_dir=prob_dirs.get(m["rf_prob"]) if m.get("rf_prob") else None,
            gt_dir=None,
            do_refinement=bool(m["refine"]),
            labeling=m["labeling"],
            subjects=val_subj,
            atlas_path=atlas_path,
        )


def pool_cv5_results(
    folds: list, gt_dir: str, cv_root: str, out_json: str, bootstrap: int = 10000
):
    pooled_dir = os.path.join(cv_root, "pooled_predictions")
    for m in METHODS:
        m_dir = os.path.join(pooled_dir, m["name"])
        os.makedirs(m_dir, exist_ok=True)
        for k in range(len(folds)):
            f_m_dir = os.path.join(cv_root, f"fold_{k}", "stage3", m["name"])
            if not os.path.isdir(f_m_dir):
                continue
            for fn in os.listdir(f_m_dir):
                if fn.endswith(".txt"):
                    shutil.copyfile(
                        os.path.join(f_m_dir, fn), os.path.join(m_dir, fn)
                    )

    all_subjects = sorted(s for flist in folds for s in flist)
    per_subject_all = {}
    summary_all = {}
    for m in METHODS:
        name = m["name"]
        m_dir = os.path.join(pooled_dir, name)
        ps = evaluate_method(m_dir, gt_dir, all_subjects, 10.0, 50.0, {})
        per_subject_all[name] = ps
        summary_all[name] = summarize(ps, bootstrap, 0)

    paired = {}
    ref = "9_hybrid_full"
    if ref in per_subject_all:
        keys = [
            "labeled_chamfer_mm",
            "labeled_cldice_3mm",
            "unlabeled_cldice_3mm",
            "nld",
            "label_acc_all",
            "detection_recall",
            "precision",
        ]
        for name, ps in per_subject_all.items():
            if name != ref:
                paired[name] = paired_tests(per_subject_all[ref], ps, keys)

    report = {
        "protocol": "5-fold cross-validation across all 59 annotated SCR scans (frozen Stage-3 constants)",
        "n_folds": len(folds),
        "n_subjects_total": len(all_subjects),
        "summary_pooled_59": summary_all,
        "paired_tests_vs_9_hybrid_full_n59": paired,
        "per_method": per_subject_all,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote 5-fold CV report to {out_json}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image-dir", "--data-dir", dest="image_dir", default="Data/vol2mm",
                    help="Directory containing 3D CT volumes")
    ap.add_argument("--gt-dir", default="Data/Centerlines_GT_fixed",
                    help="Directory containing ground-truth centerline .xls files")
    ap.add_argument("--folds", default="experiments/cv5/folds.json",
                    help="Path to folds.json defining the 5 cross-validation folds")
    ap.add_argument("--output-dir", "--out-dir", "--output", "--out", dest="output_dir", default="experiments/cv5",
                    help="Root output directory for 5-fold CV (or output .json report path)")
    ap.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="Optional single fold index (0..4); runs all 5 folds if omitted")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only pool and score existing per-fold Stage-3 predictions")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    folds_data = json.load(open(args.folds))
    folds = folds_data["folds"]

    cv_root = args.output_dir
    out_json = os.path.join(cv_root, "benchmark_cv5.json")
    if args.output_dir.endswith(".json"):
        out_json = args.output_dir
        cv_root = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)) or ".", "cv5_runs")

    if not args.eval_only:
        fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
        for k in fold_indices:
            run_single_fold(k, folds, args.image_dir, args.gt_dir, cv_root)

    if args.fold is None or args.eval_only:
        pool_cv5_results(folds, args.gt_dir, cv_root, out_json, args.bootstrap)


if __name__ == "__main__":
    main()
