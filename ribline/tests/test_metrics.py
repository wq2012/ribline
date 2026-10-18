"""Unit tests for the corrected evaluation metrics (penalty, symmetry, NaN groups, precision)."""

import math
import unittest

import numpy as np

from ribline.model.metrics import aggregate_subject_metrics, evaluate_subject_ribs


def _line(x0, x1, y=0.0, z=0.0, n=51):
    return np.column_stack(
        [np.linspace(x0, x1, n), np.full(n, y), np.full(n, z)]
    ).astype(np.float32)


class TestCorrectedMetrics(unittest.TestCase):
    def _gt(self):
        # Four GT ribs on distinct Z levels, 20 mm apart: R01, R02, R03, R12
        return {
            "R01": _line(0, 100, z=300),
            "R02": _line(0, 100, z=280),
            "R03": _line(0, 100, z=260),
            "R12": _line(0, 100, z=100),
        }

    def test_perfect_prediction(self):
        gt = self._gt()
        m = evaluate_subject_ribs(gt, gt)
        self.assertAlmostEqual(m["labeled_chamfer_mm"], 0.0)
        self.assertEqual(m["num_missing_labels"], 0)
        self.assertEqual(m["label_acc_all"], 1.0)
        self.assertEqual(m["label_acc_twelfth"], 1.0)
        self.assertEqual(m["detection_recall"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["num_false_positive_curves"], 0)

    def test_missing_label_penalty(self):
        gt = self._gt()
        pred = {k: v for k, v in gt.items() if k != "R12"}
        m = evaluate_subject_ribs(gt, pred, missing_penalty_mm=50.0)
        # 3 matched ribs at 0 mm + one missing at 50 mm -> mean 12.5 mm
        self.assertAlmostEqual(m["labeled_chamfer_mm"], 12.5)
        self.assertAlmostEqual(m["labeled_chamfer_matched_only_mm"], 0.0)
        self.assertEqual(m["num_missing_labels"], 1)
        self.assertEqual(m["label_acc_twelfth"], 0.0)
        self.assertAlmostEqual(m["label_acc_all"], 0.75)
        self.assertAlmostEqual(m["detection_recall"], 0.75)
        self.assertEqual(m["per_rib"]["R12"]["cldice_3mm"], 0.0)
        self.assertTrue(math.isinf(m["per_rib"]["R12"]["chamfer_mm"]))

    def test_twelfth_group_nan_when_absent(self):
        gt = {k: v for k, v in self._gt().items() if k != "R12"}
        m = evaluate_subject_ribs(gt, gt)
        self.assertTrue(math.isnan(m["label_acc_twelfth"]))
        self.assertEqual(m["num_twelfth_ribs"], 0)
        # Extra predicted 12th rib must not be rewarded by a default-1.0 either
        pred = dict(gt)
        pred["R12"] = _line(0, 100, z=100)
        m2 = evaluate_subject_ribs(gt, pred)
        self.assertTrue(math.isnan(m2["label_acc_twelfth"]))
        self.assertEqual(m2["num_false_positive_curves"], 1)
        self.assertAlmostEqual(m2["precision"], 3.0 / 4.0)

    def test_symmetric_vs_onesided_criterion(self):
        gt = self._gt()
        pred = dict(gt)
        # Prediction covers only the first 30 % of R02 (on the curve): p2g = 0, g2p large
        pred["R02"] = _line(0, 30, z=280, n=16)
        m = evaluate_subject_ribs(gt, pred, match_tol_mm=10.0)
        r = m["per_rib"]["R02"]
        self.assertAlmostEqual(r["d_p2g_mm"], 0.0, places=4)
        self.assertGreater(r["d_g2p_mm"], 10.0)
        self.assertTrue(r["correctly_labeled_onesided"])
        self.assertFalse(r["correctly_labeled"])
        self.assertAlmostEqual(m["label_acc_all_onesided"], 1.0)
        self.assertAlmostEqual(m["label_acc_all"], 0.75)
        self.assertAlmostEqual(m["detection_recall_onesided"], 1.0)
        self.assertAlmostEqual(m["detection_recall"], 0.75)
        # One-sided precision vs symmetric precision
        self.assertAlmostEqual(m["curve_precision"], 1.0)
        self.assertAlmostEqual(m["curve_precision_symmetric"], 0.75)
        self.assertEqual(m["num_false_positive_curves_symmetric"], 1)
        # One-sided mode reproduces the legacy behaviour
        m1 = evaluate_subject_ribs(gt, pred, match_tol_mm=10.0, symmetric_criterion=False)
        self.assertAlmostEqual(m1["label_acc_all"], 1.0)

    def test_label_shift_is_an_error(self):
        gt = self._gt()
        # Swap R02 and R03 labels: curves lie exactly on the wrong GT rib
        pred = dict(gt)
        pred["R02"], pred["R03"] = gt["R03"], gt["R02"]
        m = evaluate_subject_ribs(gt, pred)
        self.assertFalse(m["per_rib"]["R02"]["correctly_labeled"])
        self.assertFalse(m["per_rib"]["R03"]["correctly_labeled"])
        self.assertAlmostEqual(m["label_acc_all"], 0.5)
        # but they are still detected and not false positives
        self.assertAlmostEqual(m["detection_recall"], 1.0)
        self.assertEqual(m["num_false_positive_curves"], 0)

    def test_false_positive_curve(self):
        gt = self._gt()
        pred = dict(gt)
        pred["L05"] = _line(0, 100, y=200, z=200)  # far from everything
        m = evaluate_subject_ribs(gt, pred)
        self.assertEqual(m["num_false_positive_curves"], 1)
        self.assertEqual(m["false_positive_curves"], ["L05"])
        self.assertAlmostEqual(m["precision"], 4.0 / 5.0)

    def test_aggregate_pooled_counts(self):
        gt = self._gt()
        pred_a = dict(gt)
        pred_b = {k: v for k, v in gt.items() if k != "R12"}
        agg = aggregate_subject_metrics(
            {"a": evaluate_subject_ribs(gt, pred_a), "b": evaluate_subject_ribs(gt, pred_b)}
        )
        self.assertEqual(agg["pooled"]["num_gt_ribs"], 8)
        self.assertEqual(agg["pooled"]["num_correct_labels"], 7)
        self.assertAlmostEqual(agg["pooled"]["label_acc_all"], 7 / 8)
        self.assertAlmostEqual(agg["pooled"]["label_acc_twelfth"], 0.5)
        self.assertEqual(agg["pooled"]["num_twelfth_ribs"], 2)
        self.assertAlmostEqual(agg["labeled_chamfer_mm"]["mean"], 6.25)
        self.assertEqual(agg["labeled_chamfer_mm"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
