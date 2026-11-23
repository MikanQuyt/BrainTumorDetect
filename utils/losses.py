"""Các hàm mất mát cho Nhánh 1 (ViT-LoRA Classification & Box Regression) — Tuần 3.

Bao gồm:
  1. Phân loại (Classification):
     - `WeightedLabelSmoothingCrossEntropy`: Weighted Cross-Entropy kết hợp Label Smoothing (ε = 0.05).
     - `FocalLossWithLabelSmoothing`: Multi-class / Binary Focal Loss kết hợp Class Weights và Label Smoothing (ε = 0.05).
     - `compute_class_weights`: Tự động tính trọng số lớp cân bằng nghịch đảo tần suất (Inverse-Frequency).
  2. Hồi quy hộp giới hạn (Box Regression Head / ViTDet SFP):
     - `box_iou_tensor`, `generalized_box_iou_tensor`: IoU và GIoU khả vi giữa các cặp hộp [x_min, y_min, x_max, y_max].
     - `GIoULoss`: Hàm mất mát 1 - GIoU(pred, target).
     - `BoxRegressionLoss`: Tổ hợp GIoU Loss + L1 Loss có lọc theo mặt nạ `has_box` (chỉ tính trên mẫu có nhãn Box chuẩn).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "compute_class_weights",
    "WeightedLabelSmoothingCrossEntropy",
    "FocalLossWithLabelSmoothing",
    "build_classification_loss",
    "box_iou_tensor",
    "generalized_box_iou_tensor",
    "GIoULoss",
    "BoxRegressionLoss",
    "DiceLoss",
    "BinaryFocalLoss",
    "SegmentationLoss",
]


def compute_class_weights(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tính trọng số lớp nghịch đảo tần suất: w_c = N / (C_present * (count_c + smooth)).

    Các lớp vắng mặt (count_c == 0) nhận trọng số 0.0 để tránh bùng nổ trọng số lớp rỗng khi
    kết hợp với Label Smoothing; chuẩn hóa trung bình = 1.0 trên các lớp có mặt.
    """
    if isinstance(labels, torch.Tensor):
        arr = labels.detach().cpu().numpy().astype(np.int64).reshape(-1)
    else:
        arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(arr[(arr >= 0) & (arr < num_classes)], minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        total = max(float(counts.sum()), 1.0)
        n_present = int(present.sum())
        weights[present] = total / (n_present * (counts[present] + smooth))
        weights[present] = weights[present] / max(float(weights[present].mean()), 1e-8)
    else:
        weights[:] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """Weighted Cross-Entropy Loss kết hợp Label Smoothing (mặc định ε = 0.05).

    Công thức:
        q_c = (1 - ε) * 1[c == y] + ε / C
        L(x, y) = - sum_{c=1..C} w_c * q_c * log softmax(x)_c
    """

    def __init__(
        self,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing phải nằm trong khoảng [0.0, 1.0)")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction phải thuộc ('mean', 'sum', 'none')")
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if class_weights is not None:
            w = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.class_weights.to(device=logits.device, dtype=logits.dtype) if self.class_weights is not None else None
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=w,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class FocalLossWithLabelSmoothing(nn.Module):
    """Focal Loss kết hợp Trọng số Lớp (Class Weights) và Label Smoothing (ε = 0.05).

    Công thức:
        q_c = (1 - ε) * 1[c == y] + ε / C
        p_c = softmax(logits)_c
        L(x, y) = - sum_{c=1..C} w_c * q_c * (1 - p_c)^γ * log(p_c)
    """

    def __init__(
        self,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        if gamma < 0.0:
            raise ValueError("gamma phải >= 0")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing phải nằm trong khoảng [0.0, 1.0)")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction phải thuộc ('mean', 'sum', 'none')")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if class_weights is not None:
            w = class_weights if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[-1]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()

        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).to(dtype=log_probs.dtype)
        if self.label_smoothing > 0.0:
            smooth_targets = (1.0 - self.label_smoothing) * targets_one_hot + (self.label_smoothing / num_classes)
        else:
            smooth_targets = targets_one_hot

        focal_weight = (1.0 - probs).clamp(min=0.0) ** self.gamma
        loss_per_class = -smooth_targets * focal_weight * log_probs

        if self.class_weights is not None:
            w = self.class_weights.to(device=logits.device, dtype=log_probs.dtype).view(1, -1)
            loss_per_class = loss_per_class * w

        loss = loss_per_class.sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_classification_loss(
    loss_type: str = "focal",
    class_weights: Sequence[float] | torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> nn.Module:
    """Khởi tạo hàm mất mát phân loại theo tên ('focal' hoặc 'weighted_ce')."""
    loss_type = loss_type.lower().strip()
    if loss_type in ("focal", "focal_loss"):
        return FocalLossWithLabelSmoothing(
            class_weights=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    if loss_type in ("weighted_ce", "ce", "cross_entropy"):
        return WeightedLabelSmoothingCrossEntropy(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"loss_type không hợp lệ: {loss_type!r}. Chọn 'focal' hoặc 'weighted_ce'.")


# ============================================================ Hồi quy Bounding Box (GIoU + L1)
def _canonical_boxes(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Đảm bảo x_min <= x_max và y_min <= y_max cho tensor (..., 4)."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    xa = torch.minimum(x1, x2)
    xb = torch.maximum(x1, x2)
    ya = torch.minimum(y1, y2)
    yb = torch.maximum(y1, y2)
    return xa, ya, xb, yb


def box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Tính IoU theo từng cặp (elementwise) giữa boxes1 (..., 4) và boxes2 (..., 4) dạng [xmin, ymin, xmax, ymax]."""
    x1_a, y1_a, x2_a, y2_a = _canonical_boxes(boxes1)
    x1_b, y1_b, x2_b, y2_b = _canonical_boxes(boxes2)

    inter_w = (torch.minimum(x2_a, x2_b) - torch.maximum(x1_a, x1_b)).clamp(min=0.0)
    inter_h = (torch.minimum(y2_a, y2_b) - torch.maximum(y1_a, y1_b)).clamp(min=0.0)
    inter = inter_w * inter_h

    area_a = (x2_a - x1_a).clamp(min=0.0) * (y2_a - y1_a).clamp(min=0.0)
    area_b = (x2_b - x1_b).clamp(min=0.0) * (y2_b - y1_b).clamp(min=0.0)
    union = area_a + area_b - inter
    return inter / (union + eps)


def generalized_box_iou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Tính Generalized IoU (Rezatofighi et al., CVPR 2019) theo từng cặp giữa boxes1 (..., 4) và boxes2 (..., 4)."""
    x1_a, y1_a, x2_a, y2_a = _canonical_boxes(boxes1)
    x1_b, y1_b, x2_b, y2_b = _canonical_boxes(boxes2)

    inter_w = (torch.minimum(x2_a, x2_b) - torch.maximum(x1_a, x1_b)).clamp(min=0.0)
    inter_h = (torch.minimum(y2_a, y2_b) - torch.maximum(y1_a, y1_b)).clamp(min=0.0)
    inter = inter_w * inter_h

    area_a = (x2_a - x1_a).clamp(min=0.0) * (y2_a - y1_a).clamp(min=0.0)
    area_b = (x2_b - x1_b).clamp(min=0.0) * (y2_b - y1_b).clamp(min=0.0)
    union = area_a + area_b - inter
    iou = inter / (union + eps)

    enclose_w = (torch.maximum(x2_a, x2_b) - torch.minimum(x1_a, x1_b)).clamp(min=0.0)
    enclose_h = (torch.maximum(y2_a, y2_b) - torch.minimum(y1_a, y1_b)).clamp(min=0.0)
    enclose_area = enclose_w * enclose_h

    return iou - (enclose_area - union) / (enclose_area + eps)


class GIoULoss(nn.Module):
    """Hàm mất mát GIoU: L_GIoU = 1 - GIoU(pred_boxes, gt_boxes)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction phải thuộc ('mean', 'sum', 'none')")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        loss = 1.0 - generalized_box_iou_tensor(pred_boxes.float(), gt_boxes.float())
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BoxRegressionLoss(nn.Module):
    """Hàm mất mát kết hợp GIoU Loss + L1 Loss cho nhánh Box Regression Head / ViTDet SFP.

    L_box = λ_giou * (1 - GIoU(pred, gt)) + λ_l1 * ||pred - gt||_1

    Nếu truyền `has_box` (bool tensor hình (B,)), chỉ các mẫu có `has_box=True` mới được đưa vào tính loss;
    nếu cả batch không có mẫu nào có box (ví dụ toàn ảnh No Tumor hoặc Dataset 1), trả về loss = 0.0 (có grad=0).
    """

    def __init__(self, giou_weight: float = 1.0, l1_weight: float = 1.0):
        super().__init__()
        self.giou_weight = float(giou_weight)
        self.l1_weight = float(l1_weight)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        has_box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred = pred_boxes.float()
        gt = gt_boxes.float()

        if has_box is not None:
            mask = has_box.bool().view(-1)
            pred = pred[mask]
            gt = gt[mask]

        if pred.numel() == 0:
            zero = pred_boxes.sum() * 0.0
            return {
                "loss": zero,
                "giou_loss": zero.detach(),
                "l1_loss": zero.detach(),
                "mean_iou": zero.detach(),
                "n_boxes": torch.tensor(0, device=pred_boxes.device),
            }

        iou = box_iou_tensor(pred, gt)
        giou = generalized_box_iou_tensor(pred, gt)
        giou_loss = (1.0 - giou).mean()
        l1_loss = F.l1_loss(pred, gt, reduction="mean")
        total = self.giou_weight * giou_loss + self.l1_weight * l1_loss

        return {
            "loss": total,
            "giou_loss": giou_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "mean_iou": iou.detach().mean(),
            "n_boxes": torch.tensor(int(pred.shape[0]), device=pred_boxes.device),
        }


# ============================================================ Phân vùng (MedSAM-LoRA, Tuần 4)
class DiceLoss(nn.Module):
    """Soft Dice Loss nhị phân trên logit: 1 - (2·Σ(p·g) + s) / (Σp + Σg + s), tính theo từng ảnh rồi lấy trung bình."""

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction phải thuộc ('mean', 'sum', 'none')")
        self.smooth, self.reduction = float(smooth), reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.float()).flatten(1)
        g = targets.float().flatten(1)
        inter = (p * g).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (p.sum(dim=1) + g.sum(dim=1) + self.smooth)
        loss = 1.0 - dice
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss (Lin et al., 2017) trên logit: -α_t (1 - p_t)^γ log(p_t). γ = 0 và α = None ⇒ BCE."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma phải >= 0")
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha phải nằm trong [0, 1] hoặc None")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction phải thuộc ('mean', 'sum', 'none')")
        self.gamma, self.alpha, self.reduction = float(gamma), alpha, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x, t = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        p_t = torch.exp(-bce)
        loss = (1.0 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            loss = (self.alpha * t + (1.0 - self.alpha) * (1.0 - t)) * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SegmentationLoss(nn.Module):
    """L_Seg = 0.5 · L_Dice + 0.5 · L_Focal (hoặc BCE) cho Mask Decoder của MedSAM-LoRA.

    `mode="focal"` dùng Binary Focal Loss (mặc định γ = 2, α = 0.25); `mode="bce"` dùng Binary Cross-Entropy.
    Trả về dict {"loss", "dice_loss", "pixel_loss"} để ghi log từng thành phần.
    """

    def __init__(self, mode: str = "focal", dice_weight: float = 0.5, pixel_weight: float = 0.5,
                 gamma: float = 2.0, alpha: float | None = 0.25, smooth: float = 1.0):
        super().__init__()
        mode = mode.lower().strip()
        if mode not in ("focal", "bce"):
            raise ValueError("mode phải là 'focal' hoặc 'bce'")
        self.mode, self.dice_weight, self.pixel_weight = mode, float(dice_weight), float(pixel_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.pixel = BinaryFocalLoss(gamma=gamma, alpha=alpha) if mode == "focal" else BinaryFocalLoss(gamma=0.0, alpha=None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        d = self.dice(logits, targets)
        p = self.pixel(logits, targets)
        return {"loss": self.dice_weight * d + self.pixel_weight * p, "dice_loss": d.detach(), "pixel_loss": p.detach()}
