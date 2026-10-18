"""5-fold cross-validation over the 59 annotated SCR subjects with FROZEN Stage-3 constants.

Wrapper around ribline.scripts.run_cv5 for single-fold execution:
  python experiments/rebuttal/cv_fold.py <fold_index> [--data-root data]
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run a single fold of the 5-fold cross-validation benchmark.")
    ap.add_argument("fold", type=int, choices=range(5), help="Fold index (0..4)")
    ap.add_argument("--data-root", default=os.environ.get("RIBLINE_DATA_ROOT", "data"),
                    help="Data directory containing vol2mm/ and Centerlines_GT_fixed/")
    ap.add_argument("--cv-dir", default="experiments/rebuttal/cv5", help="Output directory for CV folds")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "ribline.scripts.run_cv5",
        "--image-dir", os.path.join(args.data_root, "vol2mm"),
        "--gt-dir", os.path.join(args.data_root, "Centerlines_GT_fixed"),
        "--folds", os.path.join(args.cv_dir, "folds.json"),
        "--output-dir", args.cv_dir,
        "--fold", str(args.fold),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
