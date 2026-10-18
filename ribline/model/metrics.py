"""Standardized evaluation metrics for 3D rib centerline extraction and labeling."""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


def chamfer_distance(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    """Computes symmetric Chamfer Distance (in mm) between two 3D point clouds."""
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)
    if len(gt_points) == 0 or len(pred_points) == 0:
        return float("inf")

    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)

    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)

    return float(0.5 * (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)))


def directional_chamfer(gt_points: np.ndarray, pred_points: np.ndarray) -> tuple:
    """Returns (pred_to_gt_mm, gt_to_pred_mm, sym_chamfer_mm)."""
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)
    if len(gt_points) == 0 or len(pred_points) == 0:
        return float("inf"), float("inf"), float("inf")

    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    d_p2g, _ = tree_gt.query(pred_points)
    d_g2p, _ = tree_pred.query(gt_points)
    m_p2g = float(np.mean(d_p2g))
    m_g2p = float(np.mean(d_g2p))
    return m_p2g, m_g2p, 0.5 * (m_p2g + m_g2p)


def hausdorff_distance_95(gt_points: np.ndarray, pred_points: np.ndarray) -> float:
    """Computes symmetric 95th-percentile Hausdorff Distance (HD95 in mm)."""
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)
    if len(gt_points) == 0 or len(pred_points) == 0:
        return float("inf")

    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    d_p2g, _ = tree_gt.query(pred_points)
    d_g2p, _ = tree_pred.query(gt_points)
    return float(0.5 * (np.percentile(d_p2g, 95.0) + np.percentile(d_g2p, 95.0)))


def centerline_dice(
    gt_points: np.ndarray, pred_points: np.ndarray, tolerance_mm: float = 3.0
) -> float:
    """Computes 3D Centerline Dice (clDice) within a physical radius tolerance."""
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)
    if len(gt_points) == 0 and len(pred_points) == 0:
        return 1.0
    if len(gt_points) == 0 or len(pred_points) == 0:
        return 0.0

    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)
    d_p2g, _ = tree_gt.query(pred_points)
    d_g2p, _ = tree_pred.query(gt_points)

    tprec = float(np.mean(d_p2g <= tolerance_mm))
    tsens = float(np.mean(d_g2p <= tolerance_mm))
    if tprec + tsens == 0.0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def dilated_dice(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    shape: tuple,
    spacing: tuple = (2.0, 2.0, 2.0),
    radius: float = 2.0,
) -> float:
    """Computes volumetric Dice between dilated tubes around 3D centerlines."""
    gt_points = np.asarray(gt_points, dtype=np.float32)
    pred_points = np.asarray(pred_points, dtype=np.float32)
    if len(gt_points) == 0 and len(pred_points) == 0:
        return 1.0
    if len(gt_points) == 0 or len(pred_points) == 0:
        return 0.0

    mask_gt = np.zeros(shape, dtype=bool)
    gx = np.clip((gt_points[:, 0] / spacing[0]).astype(int), 0, shape[2] - 1)
    gy = np.clip((gt_points[:, 1] / spacing[1]).astype(int), 0, shape[1] - 1)
    gz = np.clip((gt_points[:, 2] / spacing[2]).astype(int), 0, shape[0] - 1)
    mask_gt[gz, gy, gx] = True

    mask_pred = np.zeros(shape, dtype=bool)
    px = np.clip((pred_points[:, 0] / spacing[0]).astype(int), 0, shape[2] - 1)
    py = np.clip((pred_points[:, 1] / spacing[1]).astype(int), 0, shape[1] - 1)
    pz = np.clip((pred_points[:, 2] / spacing[2]).astype(int), 0, shape[0] - 1)
    mask_pred[pz, py, px] = True

    rad_vox = max(1, int(round(radius / float(np.mean(spacing)))))
    z, y, x = np.ogrid[
        -rad_vox : rad_vox + 1, -rad_vox : rad_vox + 1, -rad_vox : rad_vox + 1
    ]
    kernel = (x**2 + y**2 + z**2) <= (rad_vox**2)

    tube_gt = binary_dilation(mask_gt, structure=kernel)
    tube_pred = binary_dilation(mask_pred, structure=kernel)
    inter = int(np.sum(tube_gt & tube_pred))
    denom = int(np.sum(tube_gt)) + int(np.sum(tube_pred))
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def _curve_len(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def evaluate_subject_ribs(
    gt_ribs: dict,
    pred_ribs: dict,
    shape: tuple = None,
    spacing: tuple = (2.0, 2.0, 2.0),
    match_tol_mm: float = 10.0,
    missing_penalty_mm: float = 50.0,
    symmetric_criterion: bool = True,
) -> dict:
    """Evaluates predicted labeled 3D rib centerlines against ground truth.

    All distances are in millimetres. The definitions below are the ones used
    for every number reported in the paper / README; the legacy (pre-revision)
    quantities are still returned under explicit ``*_matched_only`` /
    ``*_onesided`` names for transparency.

    Per GT rib ``r`` with identically labelled prediction ``p`` (if any):
      * ``d_p2g`` = mean distance from prediction points to the GT curve,
        ``d_g2p`` = mean distance from GT points to the predicted curve,
        ``CD`` = (d_p2g + d_g2p) / 2.
      * **correctly labelled** iff ``p`` exists, ``d_p2g <= tol`` and (if
        ``symmetric_criterion``) ``d_g2p <= tol``, and ``r`` is the GT rib
        closest to ``p`` (within 1.5 mm of the closest).
      * **detected** iff some predicted curve (any label) has ``d_p2g <= tol``
        and (if ``symmetric_criterion``) ``d_g2p <= tol`` w.r.t. ``r``.

    Subject-level summaries:
      * ``labeled_chamfer_mm``  (LSCDError): mean over *all* GT ribs of CD, where
        a GT rib without an identically-labelled prediction contributes
        ``missing_penalty_mm`` (50 mm by default).
      * ``labeled_chamfer_matched_only_mm``: legacy mean over matched ribs only.
      * ``labeled_hd95_mm``: mean per-rib symmetric HD95 (missing ribs again
        count ``missing_penalty_mm``); ``labeled_hd95_matched_only_mm`` legacy.
      * ``labeled_cldice_{3,4}mm``: mean per-rib clDice (missing ribs = 0).
      * ``unlabeled_*``: label-agnostic pooled point-cloud metrics.
      * ``label_acc_{all,first,intermediate,twelfth}``: labelling accuracy per
        group; a group with no GT ribs yields ``nan`` (never a default of 1.0).
      * ``label_acc_all_onesided``: legacy p2g-only criterion.
      * ``detection_recall`` / ``detection_recall_onesided``.
      * ``precision``: fraction of predicted curves that lie on some GT rib
        (``d_p2g <= tol`` to the closest GT rib); ``num_false_positive_curves``
        is the complement count.
      * ``nld``: mean normalised length difference |L_pred - L_gt| / L_gt
        (missing ribs = 1.0).
    """
    gt_valid = {k: np.asarray(v, dtype=np.float32) for k, v in gt_ribs.items() if len(v) > 0}
    pred_valid = {
        k: np.asarray(v, dtype=np.float32) for k, v in pred_ribs.items() if len(v) > 0
    }

    if not gt_valid:
        return {}

    all_gt_pts = np.vstack(list(gt_valid.values()))
    all_pred_pts = (
        np.vstack(list(pred_valid.values()))
        if pred_valid
        else np.empty((0, 3), dtype=np.float32)
    )

    unlabeled_cd = chamfer_distance(all_gt_pts, all_pred_pts)
    unlabeled_hd95 = hausdorff_distance_95(all_gt_pts, all_pred_pts)
    unlabeled_cldice_3mm = centerline_dice(all_gt_pts, all_pred_pts, tolerance_mm=3.0)
    unlabeled_cldice_4mm = centerline_dice(all_gt_pts, all_pred_pts, tolerance_mm=4.0)

    # Pairwise directional distances between every prediction and every GT rib
    # pair_d[(p_name, g_name)] = (d_p2g, d_g2p, cd)
    pair_d = {}
    for p_name, p_pts in pred_valid.items():
        for g_name, g_pts in gt_valid.items():
            pair_d[(p_name, g_name)] = directional_chamfer(g_pts, p_pts)

    def _ok(d_p2g, d_g2p):
        if symmetric_criterion:
            return (d_p2g <= match_tol_mm) and (d_g2p <= match_tol_mm)
        return d_p2g <= match_tol_mm

    per_rib = {}
    labeled_cds_pen = []
    labeled_cds_matched = []
    labeled_hd95_pen = []
    labeled_hd95_matched = []
    labeled_cld3 = []
    labeled_cld4 = []
    nlds = []

    detected_flags = []
    detected_flags_onesided = []
    acc_all = []
    acc_all_onesided = []
    acc_first = []
    acc_inter = []
    acc_twelfth = []
    n_missing = 0

    for r_name, g_pts in sorted(gt_valid.items()):
        rib_num = int(r_name[1:]) if r_name[1:].isdigit() else 0

        # Detection by ANY predicted curve
        is_detected = False
        is_detected_onesided = False
        for p_name in pred_valid:
            dp, dg, _ = pair_d[(p_name, r_name)]
            if dp <= match_tol_mm:
                is_detected_onesided = True
            if _ok(dp, dg):
                is_detected = True
        detected_flags.append(float(is_detected))
        detected_flags_onesided.append(float(is_detected_onesided))

        if r_name in pred_valid:
            p_pts = pred_valid[r_name]
            d_p2g, d_g2p, cd_labeled = pair_d[(r_name, r_name)]
            hd95_labeled = hausdorff_distance_95(g_pts, p_pts)
            cld3 = centerline_dice(g_pts, p_pts, tolerance_mm=3.0)
            cld4 = centerline_dice(g_pts, p_pts, tolerance_mm=4.0)
            l_gt = max(_curve_len(g_pts), 1.0)
            l_pr = _curve_len(p_pts)
            nld = abs(l_pr - l_gt) / l_gt
            # Which GT rib is closest to this predicted curve (p2g direction)?
            closest_gt_for_pred = None
            min_p2g_across_gt = float("inf")
            for g_other_name in gt_valid:
                d_to_other = pair_d[(r_name, g_other_name)][0]
                if d_to_other < min_p2g_across_gt:
                    min_p2g_across_gt = d_to_other
                    closest_gt_for_pred = g_other_name
            not_shifted = (
                closest_gt_for_pred == r_name or d_p2g <= min_p2g_across_gt + 1.5
            )
            is_correct_onesided = (d_p2g <= match_tol_mm) and not_shifted
            is_correctly_labeled = _ok(d_p2g, d_g2p) and not_shifted
            labeled_cds_pen.append(cd_labeled)
            labeled_cds_matched.append(cd_labeled)
            labeled_hd95_pen.append(hd95_labeled)
            labeled_hd95_matched.append(hd95_labeled)
        else:
            n_missing += 1
            d_p2g = d_g2p = cd_labeled = hd95_labeled = float("inf")
            cld3 = 0.0
            cld4 = 0.0
            nld = 1.0
            is_correctly_labeled = False
            is_correct_onesided = False
            labeled_cds_pen.append(float(missing_penalty_mm))
            labeled_hd95_pen.append(float(missing_penalty_mm))

        labeled_cld3.append(cld3)
        labeled_cld4.append(cld4)
        nlds.append(nld)

        acc_val = float(is_correctly_labeled)
        acc_all.append(acc_val)
        acc_all_onesided.append(float(is_correct_onesided))
        if rib_num == 1:
            acc_first.append(acc_val)
        elif rib_num == 12:
            acc_twelfth.append(acc_val)
        else:
            acc_inter.append(acc_val)

        per_rib[r_name] = {
            "chamfer_mm": cd_labeled,
            "d_p2g_mm": d_p2g,
            "d_g2p_mm": d_g2p,
            "hd95_mm": hd95_labeled,
            "cldice_3mm": cld3,
            "cldice_4mm": cld4,
            "nld": nld,
            "detected": bool(is_detected),
            "correctly_labeled": bool(is_correctly_labeled),
            "correctly_labeled_onesided": bool(is_correct_onesided),
        }

    # Precision / false positives over predicted curves. A predicted curve is a
    # false positive if it does not lie on ANY GT rib (mean prediction-to-GT
    # distance > tol). Partial coverage of a rib is penalised by recall / clDice,
    # not here. We also compute ``curve_precision_symmetric`` requiring both
    # d_p2g <= tol and d_g2p <= tol to some GT rib.
    tp_curves = 0
    tp_curves_sym = 0
    false_positive_curves = []
    false_positive_curves_sym = []
    for p_name in pred_valid:
        best_dp = min(pair_d[(p_name, g_name)][0] for g_name in gt_valid)
        if best_dp <= match_tol_mm:
            tp_curves += 1
        else:
            false_positive_curves.append(p_name)
        if any(
            (pair_d[(p_name, g_name)][0] <= match_tol_mm)
            and (pair_d[(p_name, g_name)][1] <= match_tol_mm)
            for g_name in gt_valid
        ):
            tp_curves_sym += 1
        else:
            false_positive_curves_sym.append(p_name)

    def _mean_or_nan(vals):
        return float(np.mean(vals)) if len(vals) else float("nan")

    curve_prec = (float(tp_curves) / len(pred_valid)) if pred_valid else float("nan")
    curve_prec_sym = (
        (float(tp_curves_sym) / len(pred_valid)) if pred_valid else float("nan")
    )

    summary = {
        "num_gt_ribs": len(gt_valid),
        "num_pred_ribs": len(pred_valid),
        "num_missing_labels": int(n_missing),
        "match_tol_mm": float(match_tol_mm),
        "missing_penalty_mm": float(missing_penalty_mm),
        "symmetric_criterion": bool(symmetric_criterion),
        "labeled_chamfer_mm": _mean_or_nan(labeled_cds_pen),
        "labeled_chamfer_matched_only_mm": (
            float(np.mean(labeled_cds_matched)) if labeled_cds_matched else float("inf")
        ),
        "labeled_median_chamfer_mm": (
            float(np.median(labeled_cds_pen)) if labeled_cds_pen else float("inf")
        ),
        "labeled_hd95_mm": _mean_or_nan(labeled_hd95_pen),
        "labeled_hd95_matched_only_mm": (
            float(np.mean(labeled_hd95_matched)) if labeled_hd95_matched else float("inf")
        ),
        "unlabeled_chamfer_mm": unlabeled_cd,
        "unlabeled_hd95_mm": unlabeled_hd95,
        "labeled_cldice_3mm": _mean_or_nan(labeled_cld3),
        "labeled_cldice_4mm": _mean_or_nan(labeled_cld4),
        "unlabeled_cldice_3mm": unlabeled_cldice_3mm,
        "unlabeled_cldice_4mm": unlabeled_cldice_4mm,
        "nld": _mean_or_nan(nlds),
        "detection_recall": _mean_or_nan(detected_flags),
        "detection_recall_onesided": _mean_or_nan(detected_flags_onesided),
        "precision": curve_prec,
        "curve_precision": curve_prec,
        "curve_precision_symmetric": curve_prec_sym,
        "num_false_positive_curves": len(false_positive_curves),
        "num_false_positive_curves_symmetric": len(false_positive_curves_sym),
        "false_positive_curves": sorted(false_positive_curves),
        "label_acc_all": _mean_or_nan(acc_all),
        "label_acc_all_onesided": _mean_or_nan(acc_all_onesided),
        "label_acc_first": _mean_or_nan(acc_first),
        "label_acc_intermediate": _mean_or_nan(acc_inter),
        "label_acc_twelfth": _mean_or_nan(acc_twelfth),
        "num_first_ribs": len(acc_first),
        "num_intermediate_ribs": len(acc_inter),
        "num_twelfth_ribs": len(acc_twelfth),
        "per_rib": per_rib,
    }
    if shape is not None:
        summary["dilated_dice_2mm"] = dilated_dice(
            all_gt_pts, all_pred_pts, shape, spacing, radius=2.0
        )
        summary["dilated_dice_3mm"] = dilated_dice(
            all_gt_pts, all_pred_pts, shape, spacing, radius=3.0
        )
    return summary


def aggregate_subject_metrics(per_subject: dict) -> dict:
    """Aggregates ``evaluate_subject_ribs`` outputs over subjects.

    Returns, for each scalar key, the subject-level mean/std/median (``nan``
    values are ignored), plus rib-pooled accuracies/recall/precision
    (``*_pooled``) computed from per-rib counts so that subjects with
    different rib counts are weighted by their number of ribs.
    """
    if not per_subject:
        return {}
    keys = [
        k
        for k, v in next(iter(per_subject.values())).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_subject.values()], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "std": float(np.std(vals)) if len(vals) else float("nan"),
            "median": float(np.median(vals)) if len(vals) else float("nan"),
            "n": int(len(vals)),
        }
    # Rib-pooled counts
    n_gt = sum(m["num_gt_ribs"] for m in per_subject.values())
    n_pred = sum(m["num_pred_ribs"] for m in per_subject.values())
    n_correct = sum(
        sum(1 for r in m["per_rib"].values() if r["correctly_labeled"])
        for m in per_subject.values()
    )
    n_detected = sum(
        sum(1 for r in m["per_rib"].values() if r["detected"])
        for m in per_subject.values()
    )
    n_fp = sum(m["num_false_positive_curves"] for m in per_subject.values())
    groups = {"first": (1,), "twelfth": (12,), "intermediate": tuple(range(2, 12))}
    out["pooled"] = {
        "num_gt_ribs": int(n_gt),
        "num_pred_ribs": int(n_pred),
        "num_correct_labels": int(n_correct),
        "num_detected": int(n_detected),
        "num_false_positive_curves": int(n_fp),
        "label_acc_all": n_correct / n_gt if n_gt else float("nan"),
        "detection_recall": n_detected / n_gt if n_gt else float("nan"),
        "precision": (n_pred - n_fp) / n_pred if n_pred else float("nan"),
    }
    for g_name, nums in groups.items():
        tot = cor = 0
        for m in per_subject.values():
            for r_name, r in m["per_rib"].items():
                if r_name[1:].isdigit() and int(r_name[1:]) in nums:
                    tot += 1
                    cor += int(r["correctly_labeled"])
        out["pooled"][f"label_acc_{g_name}"] = cor / tot if tot else float("nan")
        out["pooled"][f"num_{g_name}_ribs"] = int(tot)
    return out
