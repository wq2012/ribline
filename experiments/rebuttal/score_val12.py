"""Scores the val-12 ablation under several GT protocols (writes JSONs under ablation_val12/)."""
import argparse
import json
import os
import sys

from ribline.scripts import benchmark_eval


def main():
    ap = argparse.ArgumentParser(description="Score val-12 ablation under multiple GT protocols.")
    ap.add_argument("--data-root", default=os.environ.get("RIBLINE_DATA_ROOT", "data"),
                    help="Data directory containing vol2mm/, Centerlines_GT/, Centerlines_GT_fixed/")
    ap.add_argument("--rebuttal-dir", default="experiments/rebuttal",
                    help="Directory containing ablation_val12_spec.json and val_subjects.txt")
    args = ap.parse_args()

    reb = args.rebuttal_dir
    spec = json.load(open(os.path.join(reb, "ablation_val12_spec.json")))
    methods = [f"{m['name']}={spec['output_root']}/{m['name']}" for m in spec["methods"]]
    val_path = os.path.join(reb, "val_subjects.txt")
    val_nosko_path = os.path.join(reb, "val_subjects_noSko.txt")
    val = [l.strip() for l in open(val_path) if l.strip()]
    with open(val_nosko_path, "w") as f:
        f.write("\n".join(s for s in val if not s.startswith("Sko_")) + "\n")

    def run(out, gt_dir, subjects_file, extra=()):
        argv = ["benchmark_eval", "--gt-dir", gt_dir, "--out", out, "--bootstrap", "10000",
                "--subjects", subjects_file, "--reference", "9_hybrid_full"] + list(extra)
        for m in methods:
            argv += ["--method", m]
        sys.argv = argv
        print("\n==>", os.path.basename(out))
        benchmark_eval.main()

    out_root = spec["output_root"]
    gt_fixed = os.path.join(args.data_root, "Centerlines_GT_fixed")
    gt_orig = os.path.join(args.data_root, "Centerlines_GT")
    run(os.path.join(out_root, "benchmark.json"), gt_fixed, val_path)
    run(os.path.join(out_root, "benchmark_gt_original.json"), gt_orig, val_path)
    run(os.path.join(out_root, "benchmark_gt_original_excl_rib007_R07.json"), gt_orig, val_path,
        ["--exclude-rib", "RIB_007_20091210_bone_1.5_0.6_12_38:R07"])
    run(os.path.join(out_root, "benchmark_noSko.json"), gt_fixed, val_nosko_path)


if __name__ == "__main__":
    main()
