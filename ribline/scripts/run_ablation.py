"""Regenerates every ablation row of the paper from stored probability maps.

The spec is a JSON file::

    {
      "image_dir": "Data/vol2mm",
      "gt_dir": "Data/Centerlines_GT",
      "subjects_file": "experiments/val_subjects.txt",      # optional
      "output_root": "experiments/ablation",
      "prob_dirs": {"unet": "...", "scf_anat": "...", "pass1": "...", "pass2": "..."},
      "methods": [
        {"name": "2_pass1_zsort", "prob": "pass1", "labeling": "zsort", "refine": false},
        {"name": "9_hybrid_full", "prob": "unet", "rf_prob": "scf_anat",
         "labeling": "viterbi", "refine": true}
      ],
      "reference": "9_hybrid_full",
      "exclude_ribs": ["RIB_007_20091210_bone_1.5_0.6_12_38:R07"]  # optional
    }

Each method is run through ``ribline.pipeline.pipeline_sota.run_sota_pipeline``
(frozen Stage-3 code) and all methods are then scored with
``ribline.scripts.benchmark_eval``.
"""

import argparse
import json
import os
import sys

from ribline.pipeline.pipeline_sota import run_sota_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Do not re-run methods whose output directory already has .txt files")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    out_root = spec["output_root"]
    os.makedirs(out_root, exist_ok=True)
    subjects = None
    if spec.get("subjects_file"):
        subjects = [l.strip() for l in open(spec["subjects_file"]) if l.strip()]

    for m in spec["methods"]:
        out_dir = os.path.join(out_root, m["name"])
        if args.skip_existing and os.path.isdir(out_dir) and any(
            f.endswith(".txt") for f in os.listdir(out_dir)
        ):
            print(f"[skip] {m['name']}")
            continue
        print(f"[run ] {m['name']}")
        run_sota_pipeline(
            spec["prob_dirs"][m["prob"]],
            spec["image_dir"],
            out_dir,
            rf_prob_dir=spec["prob_dirs"].get(m["rf_prob"]) if m.get("rf_prob") else None,
            gt_dir=None,
            do_refinement=bool(m.get("refine", True)),
            threshold=float(m.get("threshold", 0.45)),
            rf_weight=float(m.get("rf_weight", 0.75)),
            labeling=m.get("labeling", "viterbi"),
            subjects=subjects,
            atlas_path=spec.get("atlas_path"),
        )

    argv = ["benchmark_eval",
            "--gt-dir", spec["gt_dir"], "--out", os.path.join(out_root, "benchmark.json"),
            "--bootstrap", str(args.bootstrap)]
    if spec.get("subjects_file"):
        argv += ["--subjects", spec["subjects_file"]]
    if spec.get("reference"):
        argv += ["--reference", spec["reference"]]
    for ex in spec.get("exclude_ribs", []):
        argv += ["--exclude-rib", ex]
    for m in spec["methods"]:
        argv += ["--method", f"{m['name']}={os.path.join(out_root, m['name'])}"]
    print(" ".join(argv))
    from ribline.scripts import benchmark_eval

    old_argv = sys.argv
    sys.argv = argv
    try:
        benchmark_eval.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
