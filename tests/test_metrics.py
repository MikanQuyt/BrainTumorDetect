"""Kiểm thử mô-đun đo lường. Chạy: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import (  # noqa: E402
    ClassificationMeter, SegmentationMeter, classification_metrics, detection_metrics,
    dice_score, hd95, iou_score,
)


def sq(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


class SegmentationMetricsTest(unittest.TestCase):
    def test_dice_iou_known_values(self):
        a = np.zeros((4, 4), bool); a[0, :] = True          # 4 px
        b = np.zeros((4, 4), bool); b[0, :2] = True; b[1, :2] = True   # 4 px, giao 2 px
        self.assertAlmostEqual(dice_score(a, b), 2 * 2 / 8)
        self.assertAlmostEqual(iou_score(a, b), 2 / 6)

    def test_identical_and_disjoint(self):
        m = sq(20, 20, 5, 10, 5, 10)
        self.assertEqual(dice_score(m, m), 1.0)
        self.assertEqual(iou_score(m, m), 1.0)
        self.assertEqual(dice_score(m, sq(20, 20, 12, 18, 12, 18)), 0.0)

    def test_empty_conventions(self):
        e = np.zeros((8, 8), bool)
        m = sq(8, 8, 2, 4, 2, 4)
        self.assertEqual(dice_score(e, e), 1.0)
        self.assertEqual(iou_score(e, e), 1.0)
        self.assertEqual(hd95(e, e), 0.0)
        self.assertEqual(dice_score(e, m), 0.0)
        self.assertTrue(np.isnan(hd95(e, m)))
        self.assertTrue(np.isnan(hd95(m, e)))

    def test_hd95_identical_zero(self):
        m = sq(40, 40, 10, 30, 10, 30)
        self.assertEqual(hd95(m, m), 0.0)

    def test_hd95_parallel_lines(self):
        a = np.zeros((20, 30), bool); a[5, :] = True
        b = np.zeros((20, 30), bool); b[8, :] = True        # cách đúng 3 pixel
        self.assertAlmostEqual(hd95(a, b), 3.0)

    def test_hd95_single_pixels_and_spacing(self):
        a = np.zeros((20, 20), bool); a[2, 2] = True
        b = np.zeros((20, 20), bool); b[2, 7] = True
        self.assertAlmostEqual(hd95(a, b), 5.0)
        self.assertAlmostEqual(hd95(a, b, spacing=0.5), 2.5)

    def test_hd95_is_symmetric(self):
        a, b = sq(40, 40, 5, 25, 5, 25), sq(40, 40, 10, 35, 12, 38)
        self.assertAlmostEqual(hd95(a, b), hd95(b, a))

    def test_hd95_shifted_square_positive(self):
        a, b = sq(60, 60, 10, 40, 10, 40), sq(60, 60, 10, 40, 15, 45)
        d = hd95(a, b)
        self.assertGreater(d, 0)
        self.assertLessEqual(d, 5 * np.sqrt(2) + 1e-6)

    def test_meter_summary_excludes_undefined_hd95(self):
        m = SegmentationMeter()
        gt = sq(16, 16, 4, 8, 4, 8)
        m.update(gt, gt)                                    # hoàn hảo
        m.update(np.zeros_like(gt), gt)                     # dự đoán rỗng -> hd95 NaN
        s = m.summary()
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["n_pred_empty"], 1)
        self.assertEqual(s["hd95_n_undefined"], 1)
        self.assertAlmostEqual(s["dice"]["mean"], 0.5)
        self.assertEqual(s["hd95"]["mean"], 0.0)            # chỉ tính ca xác định


class ClassificationMetricsTest(unittest.TestCase):
    def test_binary_known_auc(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.4, 0.35, 0.8]                           # ví dụ kinh điển của scikit-learn: AUC = 0.75
        r = classification_metrics(y, p)
        self.assertAlmostEqual(r["auc"], 0.75)
        self.assertAlmostEqual(r["accuracy"], 0.75)
        self.assertAlmostEqual(r["precision"], 1.0)         # lớp dương
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertAlmostEqual(r["f1"], 2 / 3)
        self.assertEqual(r["average"], "binary")

    def test_multiclass_perfect(self):
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        p = np.eye(4)[y]
        r = classification_metrics(y, p)
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            self.assertAlmostEqual(r[k], 1.0, msg=k)
        self.assertEqual(np.trace(np.array(r["confusion_matrix"])), 8)
        self.assertEqual(r["average"], "macro")

    def test_multiclass_confusion_and_macro_recall(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 1, 2, 0])
        r = classification_metrics(y, np.eye(3)[pred])
        self.assertAlmostEqual(r["accuracy"], 4 / 6)
        self.assertAlmostEqual(r["recall"], np.mean([0.5, 1.0, 0.5]))
        self.assertEqual(r["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_missing_class_auc_is_reported(self):
        y = np.array([0, 0, 1, 1])                          # lớp 2 vắng mặt
        p = np.array([[.8, .1, .1], [.7, .2, .1], [.1, .8, .1], [.2, .7, .1]])
        r = classification_metrics(y, p)
        self.assertEqual(r["auc_classes_missing"], [2])
        self.assertAlmostEqual(r["auc"], 1.0)

    def test_detection_metrics(self):
        y = np.array([0, 0, 1, 2, 3, 1])
        p = np.zeros((6, 4))
        p[:, 0] = [.9, .4, .2, .1, .8, .3]                  # P(no_tumor)
        p[:, 1:] = ((1 - p[:, :1]) / 3)
        d = detection_metrics(y, p)                          # P(u) = 1 - P(no_tumor), ngưỡng 0.5
        self.assertEqual((d["tp"], d["fn"], d["fp"], d["tn"]), (3, 1, 1, 1))
        self.assertAlmostEqual(d["recall"], 3 / 4)
        self.assertAlmostEqual(d["specificity"], 1 / 2)

    def test_meter_matches_direct_call(self):
        rng = np.random.RandomState(0)
        y = rng.randint(0, 4, 60)
        p = rng.dirichlet(np.ones(4), 60)
        meter = ClassificationMeter()
        for i in range(0, 60, 16):
            meter.update(y[i:i + 16], p[i:i + 16])
        a, b = meter.compute(), classification_metrics(y, p)
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            self.assertAlmostEqual(a[k], b[k])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            classification_metrics([], np.zeros((0, 2)))
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], np.eye(2), class_names=["a"])


if __name__ == "__main__":
    unittest.main()
