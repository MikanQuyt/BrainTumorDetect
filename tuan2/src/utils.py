"""
Shared utility functions for BRISC 2025 experiments.
"""

import random
import json
import csv
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("brisc2025")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "cpu"):
    try:
        import torch
        if pref == "cuda" and torch.cuda.is_available():
            dev = "cuda"
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            dev = "cpu"
            log.info("Using CPU")
        return torch.device(dev)
    except ImportError:
        return "cpu"


# ─── Mask utilities ──────────────────────────────────────────────────────────

def load_binary_mask(mask_path) -> np.ndarray:
    """
    Load a segmentation mask from BRISC 2025 as a clean binary array.
    BRISC masks have values near 0 (background) and near 255 (tumor),
    with some anti-aliasing noise around edges (values 1-7, 248-254).
    We threshold at 127 to get a clean binary mask.
    Returns H×W bool numpy array.
    """
    from PIL import Image
    pil = Image.open(mask_path).convert("L")
    arr = np.array(pil)
    return arr > 127


def mask_to_bbox(mask_np: np.ndarray, padding: int = 0):
    """
    Given a binary mask (H×W uint8/bool), return tight bounding box
    [x_min, y_min, x_max, y_max] with optional padding.
    Returns None if mask is empty.
    """
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask_np.shape
    x_min = max(0, int(cmin) - padding)
    y_min = max(0, int(rmin) - padding)
    x_max = min(W - 1, int(cmax) + padding)
    y_max = min(H - 1, int(rmax) + padding)
    return [x_min, y_min, x_max, y_max]


def compute_seg_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Compute Dice, IoU, Precision, Recall for binary masks.
    Both inputs: H×W bool/uint8.
    """
    pred = pred_mask.astype(bool).flatten()
    gt   = gt_mask.astype(bool).flatten()

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return dict(dice=dice, iou=iou, precision=precision, recall=recall,
                tp=tp, fp=fp, fn=fn, tn=tn)


def compute_cls_metrics(y_true, y_pred, class_names=None):
    """
    Compute accuracy + per-class precision/recall/F1 from lists.
    Returns dict with 'accuracy', 'macro_f1', 'per_class'.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix,
    )
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred).tolist()
    return dict(accuracy=acc, macro_f1=f1, macro_precision=prec,
                macro_recall=rec, report=report, confusion_matrix=cm)


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def save_json(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved → {path}")


def save_csv(rows: list[dict], path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved CSV → {path}")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
