"""Frozen Stage-3 on the 47 annotated TRAINING subjects (detector in-sample, Stage-3 constants frozen).

Writes experiments/rebuttal/ablation_train47/benchmark.json.
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run frozen Stage-3 evaluation on the 47 annotated training subjects.")
    ap.add_argument("--data-root", default=os.environ.get("RIBLINE_DATA_ROOT", "data"),
                    help="Data directory containing vol2mm/ and Centerlines_GT_fixed/")
    ap.add_argument("--rebuttal-dir", default="experiments/rebuttal", help="Rebuttal output directory")
    ap.add_argument("--split", default="split.json", help="Path to split.json")
    args = ap.parse_args()

    out = args.rebuttal_dir
    os.makedirs(out, exist_ok=True)
    split = json.load(open(args.split))
    train = [s for s in split["train"] if s != "Gist_p2_19_Anonymous_0015"]
    subj_file = os.path.join(out, "train47_subjects.txt")
    with open(subj_file, "w") as f:
        f.write("\n".join(train) + "\n")
    spec = {
        "image_dir": os.path.join(args.data_root, "vol2mm"),
        "gt_dir": os.path.join(args.data_root, "Centerlines_GT_fixed"),
        "subjects_file": subj_file,
        "output_root": os.path.join(out, "ablation_train47"),
        "prob_dirs": {
            "unet": os.path.join(out, "train_probs_unet"),
            "scf_anat": os.path.join(out, "train_probs_scf"),
        },
        "methods": [
            {"name": "4_scf_anat_stage3", "prob": "scf_anat", "labeling": "viterbi", "refine": True},
            {"name": "7_unet_stage3", "prob": "unet", "labeling": "viterbi", "refine": True},
            {"name": "8_hybrid_viterbi_nosnake", "prob": "unet", "rf_prob": "scf_anat",
             "labeling": "viterbi", "refine": False},
            {"name": "9_hybrid_full", "prob": "unet", "rf_prob": "scf_anat",
             "labeling": "viterbi", "refine": True},
        ],
        "reference": "9_hybrid_full",
    }
    spec_path = os.path.join(out, "ablation_train47_spec.json")
    json.dump(spec, open(spec_path, "w"), indent=2)
    subprocess.check_call([sys.executable, "-m", "ribline.scripts.run_ablation", "--spec", spec_path])


if __name__ == "__main__":
    main()
