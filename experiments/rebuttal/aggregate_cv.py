"""Aggregates the 5 CV folds into benchmark_cv5.json (pooled over the 59 held-out scans)."""
import argparse

from ribline.scripts.run_cv5 import aggregate_cv_folds


def main():
    ap = argparse.ArgumentParser(description="Aggregate 5 CV folds into benchmark_cv5.json.")
    ap.add_argument("--cv-dir", default="experiments/rebuttal/cv5", help="Directory containing fold_0..fold_4")
    ap.add_argument("--out", default="experiments/rebuttal/benchmark_cv5.json", help="Output JSON path")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    aggregate_cv_folds(args.cv_dir, args.out, n_boot=args.bootstrap)


if __name__ == "__main__":
    main()
