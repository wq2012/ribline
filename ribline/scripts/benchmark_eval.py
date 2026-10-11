"""Benchmark evaluation of labelled rib-centerline predictions with corrected metrics.

Evaluates one or more prediction directories (``<subject>.txt`` files written by
``ribline.pipeline.pipeline_sota``) against ground truth with
``ribline.model.metrics.evaluate_subject_ribs`` and reports

* subject-level mean / std / median and 95 % bootstrap confidence intervals
  (resampling subjects, ``--bootstrap`` iterations),
* rib-pooled labelling accuracy / recall / precision with Wilson 95 % CIs,
* per-rib pooled distributions (HD95 and clDice tails, worst ribs),
* paired Wilcoxon signed-rank tests of every method against ``--reference``.

Example::

    python -m ribline.scripts.benchmark_eval \
        --gt-dir Data/Centerlines_GT \
        --method "9_hybrid_full=experiments/ablation/9_hybrid_full" \
        --method "8_hybrid_nosnake=experiments/ablation/8_hybrid_nosnake" \
        --reference 9_hybrid_full --out experiments/ablation/benchmark.json
"""

import argparse
import json
import math
import os

import numpy as np

from ribline.data.centerline_parser import parse_rib_xls
from ribline.model.metrics import aggregate_subject_metrics, evaluate_subject_ribs

SUBJECT_KEYS = [
    "labeled_chamfer_mm",
    "labeled_chamfer_matched_only_mm",
    "labeled_hd95_mm",
    "unlabeled_chamfer_mm",
    "unlabeled_hd95_mm",
    "labeled_cldice_3mm",
    "labeled_cldice_4mm",
    "unlabeled_cldice_3mm",
    "unlabeled_cldice_4mm",
    "nld",
    "label_acc_all",
    "label_acc_all_onesided",
    "label_acc_first",
    "label_acc_intermediate",
    "label_acc_twelfth",
    "detection_recall",
    "detection_recall_onesided",
    "precision",
]


def load_pred_txt(path: str) -> dict:
    """Loads ``RibName X Y Z`` tab/space separated centerlines."""
    ribs = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                c = [float(x) for x in parts[1:4]]
            except ValueError:
                continue
            ribs.setdefault(parts[0], []).append(c)
    return {k: np.asarray(v, dtype=np.float32) for k, v in ribs.items()}


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.RandomState) -> tuple:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.randint(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_method(pred_dir: str, gt_dir: str, subjects: list, match_tol_mm: float,
                    missing_penalty_mm: float, exclude_ribs: dict) -> dict:
    per_subject = {}
    for s in subjects:
        gt_path = os.path.join(gt_dir, f"{s}_interp.xls")
        pred_path = os.path.join(pred_dir, f"{s}.txt")
        if not os.path.exists(gt_path):
            continue
        gt = parse_rib_xls(gt_path)
        if not gt:
            continue
        for r in exclude_ribs.get(s, []):
            gt.pop(r, None)
        pred = load_pred_txt(pred_path) if os.path.exists(pred_path) else {}
        per_subject[s] = evaluate_subject_ribs(
            gt, pred, match_tol_mm=match_tol_mm, missing_penalty_mm=missing_penalty_mm
        )
    return per_subject


def summarize(per_subject: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    agg = aggregate_subject_metrics(per_subject)
    out = {"n_subjects": len(per_subject), "subject_level": {}, "pooled": agg["pooled"]}
    for k in SUBJECT_KEYS:
        if k not in agg:
            continue
        vals = np.array([m[k] for m in per_subject.values()], dtype=np.float64)
        lo, hi = bootstrap_ci(vals, n_boot, rng)
        out["subject_level"][k] = dict(agg[k], ci95=[lo, hi])
    # Wilson CIs for pooled rates
    p = agg["pooled"]
    for name, k, n in [
        ("label_acc_all", p["num_correct_labels"], p["num_gt_ribs"]),
        ("detection_recall", p["num_detected"], p["num_gt_ribs"]),
        ("precision", p["num_pred_ribs"] - p["num_false_positive_curves"], p["num_pred_ribs"]),
    ]:
        out["pooled"][f"{name}_wilson95"] = list(wilson_ci(k, n))
    # Rule-of-three upper bound when zero errors observed
    if p["num_correct_labels"] == p["num_gt_ribs"] and p["num_gt_ribs"] > 0:
        out["pooled"]["rule_of_three_error_upper_bound"] = 3.0 / p["num_gt_ribs"]
    # Per-rib pooled tails
    cds, hd95s, cl3s, cl4s, nlds, worst = [], [], [], [], [], []
    for s, m in per_subject.items():
        for r, d in m["per_rib"].items():
            if math.isfinite(d["chamfer_mm"]):
                cds.append(d["chamfer_mm"])
                hd95s.append(d["hd95_mm"])
                nlds.append(d["nld"])
                worst.append((d["chamfer_mm"], d["d_p2g_mm"], d["d_g2p_mm"], d["hd95_mm"], s, r))
            cl3s.append(d["cldice_3mm"])
            cl4s.append(d["cldice_4mm"])
    cds, hd95s, cl3s = np.array(cds), np.array(hd95s), np.array(cl3s)
    worst.sort(reverse=True)
    out["per_rib_pooled"] = {
        "n_matched": int(len(cds)),
        "n_total": int(len(cl3s)),
        "chamfer_mean": float(cds.mean()) if len(cds) else float("nan"),
        "chamfer_median": float(np.median(cds)) if len(cds) else float("nan"),
        "chamfer_max": float(cds.max()) if len(cds) else float("nan"),
        "n_chamfer_gt_5mm": int((cds > 5).sum()),
        "n_chamfer_gt_10mm": int((cds > 10).sum()),
        "n_hd95_gt_10mm": int((hd95s > 10).sum()),
        "n_hd95_gt_20mm": int((hd95s > 20).sum()),
        "n_cldice3_lt_0p8": int((cl3s < 0.8).sum()),
        "n_cldice3_lt_0p5": int((cl3s < 0.5).sum()),
        "worst_10_ribs": [
            {"chamfer_mm": a, "d_p2g_mm": b, "d_g2p_mm": c, "hd95_mm": h, "subject": s, "rib": r}
            for (a, b, c, h, s, r) in worst[:10]
        ],
    }
    # Per rib-number mean CD / accuracy
    per_num = {}
    for s, m in per_subject.items():
        for r, d in m["per_rib"].items():
            if not r[1:].isdigit():
                continue
            k = int(r[1:])
            e = per_num.setdefault(k, {"cd": [], "correct": 0, "n": 0})
            e["n"] += 1
            e["correct"] += int(d["correctly_labeled"])
            if math.isfinite(d["chamfer_mm"]):
                e["cd"].append(d["chamfer_mm"])
    out["per_rib_number"] = {
        k: {
            "n": v["n"],
            "label_acc": v["correct"] / v["n"],
            "chamfer_mean": float(np.mean(v["cd"])) if v["cd"] else float("nan"),
            "n_missing": v["n"] - len(v["cd"]),
        }
        for k, v in sorted(per_num.items())
    }
    return out


def paired_tests(per_subject_ref: dict, per_subject_other: dict, keys: list) -> dict:
    from scipy import stats

    subs = sorted(set(per_subject_ref) & set(per_subject_other))
    res = {"n_pairs": len(subs)}
    for k in keys:
        a = np.array([per_subject_ref[s][k] for s in subs], dtype=np.float64)
        b = np.array([per_subject_other[s][k] for s in subs], dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        entry = {"mean_ref": float(a.mean()) if len(a) else float("nan"),
                 "mean_other": float(b.mean()) if len(b) else float("nan")}
        try:
            entry["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            entry["wilcoxon_p"] = float("nan")
        try:
            entry["paired_t_p"] = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            entry["paired_t_p"] = float("nan")
        res[k] = entry
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--method", action="append", required=True,
                    help="name=prediction_dir (repeatable)")
    ap.add_argument("--subjects", default=None, help="Text file with subject IDs (default: all GT)")
    ap.add_argument("--reference", default=None, help="Method name for paired tests")
    ap.add_argument("--match-tol-mm", type=float, default=10.0)
    ap.add_argument("--missing-penalty-mm", type=float, default=50.0)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-rib", action="append", default=[],
                    help="subject:RIB to drop from GT before scoring (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    if args.subjects:
        subjects = [l.strip() for l in open(args.subjects) if l.strip()]
    else:
        subjects = sorted(
            f[: -len("_interp.xls")] for f in os.listdir(args.gt_dir) if f.endswith("_interp.xls")
        )
    exclude = {}
    for item in args.exclude_rib:
        s, r = item.split(":")
        exclude.setdefault(s, []).append(r)

    methods = {}
    for item in args.method:
        name, path = item.split("=", 1)
        methods[name] = path

    report = {
        "config": {
            "gt_dir": args.gt_dir,
            "subjects": subjects,
            "match_tol_mm": args.match_tol_mm,
            "missing_penalty_mm": args.missing_penalty_mm,
            "symmetric_criterion": True,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "exclude_ribs": exclude,
            "methods": methods,
        },
        "summary": {},
        "paired_tests_vs_reference": {},
        "per_method": {},
    }
    per_subject_all = {}
    for name, path in methods.items():
        ps = evaluate_method(path, args.gt_dir, subjects, args.match_tol_mm,
                             args.missing_penalty_mm, exclude)
        per_subject_all[name] = ps
        report["per_method"][name] = ps
        report["summary"][name] = summarize(ps, args.bootstrap, args.seed)
        sl = report["summary"][name]["subject_level"]
        po = report["summary"][name]["pooled"]
        print(
            f"{name:40s} n={len(ps):2d} LCD={sl['labeled_chamfer_mm']['mean']:.2f}"
            f"+-{sl['labeled_chamfer_mm']['std']:.2f} CI[{sl['labeled_chamfer_mm']['ci95'][0]:.2f},"
            f"{sl['labeled_chamfer_mm']['ci95'][1]:.2f}] "
            f"lclD3={sl['labeled_cldice_3mm']['mean']:.3f} acc={po['label_acc_all']:.4f} "
            f"({po['num_correct_labels']}/{po['num_gt_ribs']}) rec={po['detection_recall']:.4f} "
            f"prec={po['precision']:.4f} FP={po['num_false_positive_curves']} NLD={sl['nld']['mean']:.3f}"
        )
    if args.reference and args.reference in per_subject_all:
        keys = ["labeled_chamfer_mm", "labeled_cldice_3mm", "unlabeled_cldice_3mm",
                "nld", "label_acc_all", "detection_recall", "precision"]
        for name, ps in per_subject_all.items():
            if name == args.reference:
                continue
            report["paired_tests_vs_reference"][name] = paired_tests(
                per_subject_all[args.reference], ps, keys
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
