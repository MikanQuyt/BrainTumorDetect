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
    "classification_metrics", "detection_metrics", "optimize_detection_threshold", "ClassificationMeter",
    "dice_score", "iou_score", "hd95", "segmentation_metrics", "SegmentationMeter",
    "box_iou_score", "pointing_game_hit", "BoxLocalizationMeter",
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


def optimize_detection_threshold(
    y_true,
    y_prob,
    no_tumor_class: int = 0,
    target_recall: float = 0.95,
    thresholds: Sequence[float] | None = None,
) -> dict:
    """Quét ngưỡng quyết định τ (Threshold Moving) để đạt Recall >= target_recall với F1 cao nhất."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.81, 0.05)]
    candidates = [detection_metrics(y_true, y_prob, no_tumor_class=no_tumor_class, threshold=float(t))
                  for t in thresholds]
    feasible = [c for c in candidates if not np.isnan(c["recall"]) and c["recall"] >= target_recall]
    if feasible:
        best = max(feasible, key=lambda c: (c["f1"], c["specificity"], -c["threshold"]))
    else:
        best = max(candidates, key=lambda c: (c["recall"] if not np.isnan(c["recall"]) else -1.0, c["f1"]))
    return {"best": best, "target_recall": target_recall, "curve": candidates}


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


# ============================================================ định vị tự động (Box IoU & Pointing Game)
def box_iou_score(pred_box: Sequence[float] | np.ndarray, gt_box: Sequence[float] | np.ndarray) -> float:
    """Tính IoU giữa 2 hộp giới hạn 1D [x_min, y_min, x_max, y_max] (hệ pixel hoặc chuẩn hóa [0, 1])."""
    p = np.asarray(pred_box, dtype=np.float64).reshape(4)
    g = np.asarray(gt_box, dtype=np.float64).reshape(4)
    px1, py1, px2, py2 = min(p[0], p[2]), min(p[1], p[3]), max(p[0], p[2]), max(p[1], p[3])
    gx1, gy1, gx2, gy2 = min(g[0], g[2]), min(g[1], g[3]), max(g[0], g[2]), max(g[1], g[3])

    iw = max(0.0, min(px2, gx2) - max(px1, gx1))
    ih = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter = iw * ih
    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    if union <= 0.0:
        return 1.0 if area_p == 0.0 and area_g == 0.0 else 0.0
    return float(inter / union)


def pointing_game_hit(
    point_xy: Sequence[float] | np.ndarray,
    gt_mask: np.ndarray | None = None,
    gt_box: Sequence[float] | np.ndarray | None = None,
) -> bool:
    """Kiểm tra tâm điểm cực đại (x, y) có rơi trúng vào trong khối u (gt_mask > 0 hoặc trong gt_box) hay không."""
    pt = np.asarray(point_xy, dtype=np.float64).reshape(2)
    x, y = float(pt[0]), float(pt[1])
    if gt_mask is not None:
        m = _bin(gt_mask)
        if m.any():
            h, w = m.shape[:2]
            xi = int(np.clip( round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            return bool(m[yi, xi])
    if gt_box is not None:
        g = np.asarray(gt_box, dtype=np.float64).reshape(4)
        gx1, gy1, gx2, gy2 = min(g[0], g[2]), min(g[1], g[3]), max(g[0], g[2]), max(g[1], g[3])
        if gx2 > gx1 and gy2 > gy1:
            return bool((gx1 <= x <= gx2) and (gy1 <= y <= gy2))
    return False


class BoxLocalizationMeter:
    """Tích lũy và tính Mean Box IoU, IoU@0.5, IoU@0.75 và Pointing Game Accuracy cho bộ tự sinh lời nhắc."""

    def __init__(self):
        self.ious: list[float] = []
        self.hits: list[bool] = []

    def update(
        self,
        pred_box: Sequence[float] | np.ndarray | None = None,
        gt_box: Sequence[float] | np.ndarray | None = None,
        peak_point_xy: Sequence[float] | np.ndarray | None = None,
        gt_mask: np.ndarray | None = None,
    ):
        if pred_box is not None and gt_box is not None:
            self.ious.append(box_iou_score(pred_box, gt_box))
        if peak_point_xy is not None and (gt_mask is not None or gt_box is not None):
            self.hits.append(pointing_game_hit(peak_point_xy, gt_mask=gt_mask, gt_box=gt_box))

    def summary(self) -> dict:
        arr = np.asarray(self.ious, dtype=np.float64)
        hits = np.asarray(self.hits, dtype=np.float64)
        return {
            "n_boxes": int(len(arr)),
            "mean_box_iou": float(arr.mean()) if len(arr) else float("nan"),
            "sd_box_iou": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
            "median_box_iou": float(np.median(arr)) if len(arr) else float("nan"),
            "iou_at_50": float((arr >= 0.50).mean()) if len(arr) else float("nan"),
            "iou_at_75": float((arr >= 0.75).mean()) if len(arr) else float("nan"),
            "n_points": int(len(hits)),
            "pointing_game_acc": float(hits.mean()) if len(hits) else float("nan"),
        }

