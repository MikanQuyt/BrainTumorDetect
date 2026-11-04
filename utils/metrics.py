"""Chỉ số đánh giá phân loại và phân vùng (per-case)."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def classification_metrics(y_true, y_pred, y_score=None):
    """Chỉ số phân loại nhị phân (1 = có u).

    y_pred: nhãn dự đoán 0/1; y_score: xác suất lớp 1 để tính ROC-AUC (nếu None -> nan).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def div(a, b):
        return a / b if b > 0 else float('nan')

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)

    auc = float('nan')
    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, np.asarray(y_score).ravel()))

    return {
        'accuracy': div(tp + tn, tp + tn + fp + fn),
        'sensitivity': recall,
        'specificity': div(tn, tn + fp),
        'precision': precision,
        'f1': f1,
        'roc_auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def _binarize(mask):
    return np.asarray(mask) > 0


def dice_score(pred, gt):
    """DSC = 2|P∩G| / (|P| + |G| + 1e-6). Cả hai rỗng -> 1.0."""
    p, g = _binarize(pred), _binarize(gt)
    if not p.any() and not g.any():
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + EPS))


def iou_score(pred, gt):
    """IoU = |P∩G| / (|P∪G| + 1e-6). Cả hai rỗng -> 1.0."""
    p, g = _binarize(pred), _binarize(gt)
    if not p.any() and not g.any():
        return 1.0
    return float(np.logical_and(p, g).sum() / (np.logical_or(p, g).sum() + EPS))


def _surface(mask):
    """Điểm biên của mặt nạ: mask trừ phần co lại."""
    return mask & ~binary_erosion(mask)


def hd95(pred, gt, spacing=None):
    """Hausdorff Distance phân vị 95 giữa đường viền P và G (distance_transform_edt).

    Cả hai rỗng -> 0.0; chỉ một bên rỗng -> nan (không xác định).
    """
    p, g = _binarize(pred), _binarize(gt)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float('nan')

    sp, sg = _surface(p), _surface(g)
    dist_to_g = distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = distance_transform_edt(~sp, sampling=spacing)
    distances = np.concatenate([dist_to_g[sp], dist_to_p[sg]])
    return float(np.percentile(distances, 95))


def segmentation_metrics(pred, gt, spacing=None):
    """Dice, IoU, HD95 cho một ảnh."""
    return {'dice': dice_score(pred, gt), 'iou': iou_score(pred, gt), 'hd95': hd95(pred, gt, spacing)}


def evaluate_segmentation_cases(preds, gts, spacing=None):
    """Đánh giá từng ảnh. Trả về (danh sách chỉ số từng ảnh, trung bình bỏ qua nan)."""
    cases = [segmentation_metrics(p, g, spacing) for p, g in zip(preds, gts)]
    mean = {k: float(np.nanmean([c[k] for c in cases])) if cases else float('nan')
            for k in ('dice', 'iou', 'hd95')}
    return cases, mean

