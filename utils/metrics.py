"""Mô-đun đo lường tự động.

Phân loại : Accuracy, Recall, Precision, F1, AUC (nhị phân và đa lớp OvR), ma trận nhầm lẫn,
            kèm chỉ số sàng lọc u (nhị phân "có u / không u") với Recall (độ nhạy) và Specificity.
Phân vùng : Dice Score, IoU, HD95 (Hausdorff Distance phân vị 95).

Quy ước (ghi rõ để so sánh công bằng giữa các tuần):
  * Dice/IoU: pred và gt cùng rỗng -> 1.0; chỉ một bên rỗng -> 0.0.
  * HD95: pred và gt cùng rỗng -> 0.0; chỉ một bên rỗng -> NaN (không xác định, KHÔNG bị ép về 0).
    Khi tổng hợp, `SegmentationMeter` dùng nan-mean và báo `hd95_n_undefined` để không che giấu ca hỏng.
  * HD95 tính trên biên (surface) của mask, đơn vị = pixel * `spacing` (mm nếu có spacing).
  * Precision/Recall/F1 đa lớp: macro-average (mỗi lớp trọng số bằng nhau).
  * AUC đa lớp: One-vs-Rest macro; lớp vắng mặt trong tập đánh giá bị bỏ qua (ghi vào `auc_classes_missing`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

__all__ = [
    "classification_metrics", "detection_metrics", "ClassificationMeter",
    "dice_score", "iou_score", "hd95", "segmentation_metrics", "SegmentationMeter",
]


# ============================================================ phân loại
def _as_probs(y_prob, n_classes=None):
    p = np.asarray(y_prob, dtype=np.float64)
    if p.ndim == 1:                       # xác suất lớp dương -> (N, 2)
        p = np.stack([1.0 - p, p], axis=1)
    if n_classes is not None and p.shape[1] != n_classes:
        raise ValueError(f"y_prob có {p.shape[1]} cột nhưng n_classes={n_classes}")
    return p


def _auc(y_true, probs, labels):
    """AUC nhị phân hoặc OvR macro; trả (auc, danh_sách_lớp_vắng)."""
    present = [c for c in labels if (y_true == c).any()]
    missing = [int(c) for c in labels if c not in present]
    if len(labels) == 2:
        if len(present) < 2:
            return float("nan"), missing
        return float(roc_auc_score(y_true, probs[:, 1])), missing
    if len(present) < 2:
        return float("nan"), missing
    aucs = [roc_auc_score((y_true == c).astype(int), probs[:, c])
            for c in present if (y_true != c).any()]
    return (float(np.mean(aucs)) if aucs else float("nan")), missing


def classification_metrics(y_true, y_prob, class_names: Sequence[str] | None = None) -> dict:
    """Accuracy, Recall, Precision, F1 (macro), AUC từ nhãn thật và ma trận xác suất (N, C).

    Với bài toán nhị phân có thể truyền y_prob dạng (N,) = xác suất lớp dương;
    khi đó Precision/Recall/F1 là của LỚP DƯƠNG (nhãn 1) thay vì macro.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = _as_probs(y_prob)
    n_classes = probs.shape[1]
    labels = np.arange(n_classes)
    names = list(class_names) if class_names is not None else [str(c) for c in labels]
    if len(names) != n_classes:
        raise ValueError("class_names không khớp số lớp")
    if len(y_true) == 0:
        raise ValueError("Không có mẫu để đánh giá")

    pred = probs.argmax(axis=1)
    average = "binary" if n_classes == 2 else "macro"
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=labels, average=average, zero_division=0,
        **({"pos_label": 1} if average == "binary" else {}))
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    auc, missing = _auc(y_true, probs, labels)
    return {
        "n": int(len(y_true)),
        "accuracy": float((pred == y_true).mean()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "auc": auc,
        "average": average,
        "auc_classes_missing": missing,
        "per_class": {names[i]: {"precision": float(pc[i]), "recall": float(rc[i]),
                                 "f1": float(fc[i]), "support": int(sc[i])} for i in range(n_classes)},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "class_names": names,
    }


def detection_metrics(y_true, y_prob, no_tumor_class: int = 0, threshold: float = 0.5) -> dict:
    """Sàng lọc nhị phân 'có u / không u' từ xác suất đa lớp.

    P(u) = 1 - P(no_tumor). Recall ở đây = độ nhạy (Sensitivity) — chỉ số then chốt của sàng lọc.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = _as_probs(y_prob)
    p_tumor = 1.0 - probs[:, no_tumor_class]
    t = (y_true != no_tumor_class).astype(int)
    pred = (p_tumor >= threshold).astype(int)
    tp = int(((pred == 1) & (t == 1)).sum())
    tn = int(((pred == 0) & (t == 0)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, len(t)),
        "recall": recall, "specificity": div(tn, tn + fp), "precision": precision,
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "auc": float(roc_auc_score(t, p_tumor)) if 0 < t.sum() < len(t) else float("nan"),
    }


class ClassificationMeter:
    """Tích lũy xác suất/nhãn qua nhiều batch rồi tính toàn bộ chỉ số một lần."""

    def __init__(self, class_names: Sequence[str] | None = None):
        self.class_names = class_names
        self._y, self._p = [], []

    def update(self, y_true, y_prob):
        self._y.append(np.asarray(y_true).reshape(-1))
        self._p.append(_as_probs(y_prob))

    def compute(self, with_detection: bool = False) -> dict:
        y, p = np.concatenate(self._y), np.concatenate(self._p)
        out = classification_metrics(y, p, self.class_names)
        if with_detection:
            out["detection"] = detection_metrics(y, p)
        return out

    def arrays(self):
        return np.concatenate(self._y), np.concatenate(self._p)


# ============================================================ phân vùng
def _bin(mask):
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    return m.astype(bool)


def dice_score(pred, gt) -> float:
    p, g = _bin(pred), _bin(gt)
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * (p & g).sum() / s)


def iou_score(pred, gt) -> float:
    p, g = _bin(pred), _bin(gt)
    union = (p | g).sum()
    return 1.0 if union == 0 else float((p & g).sum() / union)


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, mask.ndim))


def hd95(pred, gt, spacing: float | Sequence[float] = 1.0) -> float:
    """Hausdorff Distance phân vị 95 giữa hai biên mask (đối xứng: lấy max của hai chiều)."""
    p, g = _bin(pred), _bin(gt)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    # Mask 1 pixel: biên rỗng sau erosion -> dùng chính mask
    sp = sp if sp.any() else p
    sg = sg if sg.any() else g
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg = d_to_g[sp]      # từ biên pred tới gt
    d_gp = d_to_p[sg]      # từ biên gt tới pred
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def segmentation_metrics(pred, gt, spacing=1.0) -> dict:
    return {"dice": dice_score(pred, gt), "iou": iou_score(pred, gt), "hd95": hd95(pred, gt, spacing)}


class SegmentationMeter:
    """Tích lũy chỉ số theo từng ảnh; `summary()` trả mean/sd/median và số ca HD95 không xác định."""

    def __init__(self, spacing=1.0):
        self.spacing = spacing
        self.rows: list[dict] = []

    def update(self, pred, gt, **extra):
        r = segmentation_metrics(pred, gt, self.spacing)
        r["pred_empty"] = bool(not _bin(pred).any())
        r["gt_empty"] = bool(not _bin(gt).any())
        r.update(extra)
        self.rows.append(r)
        return r

    def update_batch(self, preds, gts):
        for p, g in zip(preds, gts):
            self.update(p, g)

    def summary(self) -> dict:
        if not self.rows:
            raise ValueError("Chưa có mẫu nào")
        out = {"n": len(self.rows), "n_pred_empty": int(sum(r["pred_empty"] for r in self.rows))}
        for k in ("dice", "iou", "hd95"):
            v = np.array([r[k] for r in self.rows], dtype=float)
            ok = v[~np.isnan(v)]
            out[k] = {"mean": float(ok.mean()) if len(ok) else float("nan"),
                      "sd": float(ok.std(ddof=1)) if len(ok) > 1 else float("nan"),
                      "median": float(np.median(ok)) if len(ok) else float("nan")}
        out["hd95_n_undefined"] = int(np.isnan([r["hd95"] for r in self.rows]).sum())
        return out
