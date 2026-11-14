"""Bộ Tự sinh Lời nhắc (Auto-Prompt Generator) cho Nhánh 1 (ViT-LoRA) — Tuần 3.

Hỗ trợ đầy đủ 2 cơ chế tự sinh lời nhắc theo thiết kế đề tài:
  1. Cơ chế 1 — Giám sát yếu qua Bản đồ nhiệt (Weakly-Supervised Heatmap: Attention Rollout + LayerCAM):
     - Trích xuất ma trận Self-Attention của toàn bộ 12 tầng ViT, tính **Attention Rollout** có kết nối tắt
       (Residual Identity) từ token `[CLS]` tới 14x14 patch tokens.
     - Tính **LayerCAM** (Jiang et al., IEEE TIP 2021) bằng tích phần tử giữa hoạt hóa token patch và
       gradient dương ReLU(∂y^c / ∂F) tại các tầng Transformer cuối để giữ độ phân giải định vị cục bộ.
     - Kết hợp (Fusion) Attention Rollout + LayerCAM và áp dụng **Mặt nạ Nhu mô não (Brain Parenchyma Mask)**
       để loại bỏ nhiễu viền xương sọ sáng màu, từ đó trích xuất:
         * Hộp giới hạn tự sinh `[x_min, y_min, x_max, y_max]` (qua ngưỡng thích nghi + thành phần liên thông chứa đỉnh).
         * Tâm điểm dương cực đại `Peak Positive Point (x+, y+)` (label = 1) và điểm âm nền não `Negative Point (x-, y-)` (label = 0).
     - Dùng trực tiếp cho **Dataset 1 (Navoneel)** hoàn toàn không có nhãn Box/Mask, và cung cấp điểm Peak Point cho Hybrid Prompt.

  2. Cơ chế 2 — Nhánh Hồi quy Hộp giới hạn có Giám sát (`ViTDetBoxHead` huấn luyện bằng GIoU Loss + L1 Loss):
     - Dự đoán trực tiếp tọa độ hộp giới hạn chuẩn hóa `[x_min, y_min, x_max, y_max] ∈ [0, 1]^4` từ nhánh ViTDet SFP,
       mở rộng biên hộp `+δ%` (mặc định `δ = 8%`) và ghép với `Peak Positive/Negative Points` từ bản đồ nhiệt thành
       **Hybrid Prompt (Bounding Box + Peak Points)** chuẩn 1024x1024 cho MedSAM/SAM (dành cho **Dataset 2, 3, 4**).
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from models.vit_baseline import IMAGENET_MEAN, IMAGENET_STD
from models.vit_lora import pad_normalized_boxes

__all__ = [
    "compute_attention_rollout",
    "compute_layercam",
    "extract_brain_parenchyma_mask",
    "fuse_rollout_and_layercam",
    "heatmap_to_box_and_points",
    "AutoPromptGenerator",
]


def _minmax_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Chuẩn hóa min-max về [0, 1] trên 2 chiều không gian cuối (..., H, W)."""
    flat = x.flatten(-2)
    xmin = flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    xmax = flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    return ((x - xmin) / (xmax - xmin + eps)).clamp(0.0, 1.0)


def compute_attention_rollout(
    attentions: Sequence[torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """Tính bản đồ nhiệt Attention Rollout (Abnar & Zuidema, ACL 2020) qua 12 tầng Self-Attention của ViT.

    Args:
        attentions: danh sách L tensor có hình (B, num_heads, N, N) với N = 1 + H_p * W_p (197 cho ViT-B/16).
        discard_ratio: tỷ lệ trọng số attention nhỏ nhất bị đặt về 0 trước khi nhân chuỗi (mặc định 0.0).
        head_fusion: cách gộp các đầu chú ý ('mean', 'max', hoặc 'min').

    Returns:
        rollout_grid: tensor hình (B, H_p, W_p) đã chuẩn hóa về [0, 1].
    """
    if not attentions:
        raise ValueError("Danh sách attentions rỗng — cần chạy ViT với output_attentions=True và attn_implementation='eager'.")

    first = attentions[0]
    b, _, n, _ = first.shape
    num_patches = n - 1
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Số lượng patch token ({num_patches}) không phải số chính phương.")

    device = first.device
    eye = torch.eye(n, device=device, dtype=torch.float32).unsqueeze(0).expand(b, n, n)
    joint_att = eye.clone()

    for att_layer in attentions:
        att = att_layer.detach().float()  # (B, H, N, N)
        if head_fusion == "mean":
            fused = att.mean(dim=1)
        elif head_fusion == "max":
            fused = att.max(dim=1).values
        elif head_fusion == "min":
            fused = att.min(dim=1).values
        else:
            raise ValueError(f"head_fusion không hợp lệ: {head_fusion!r}")

        if 0.0 < discard_ratio < 1.0:
            flat = fused.view(b, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                thresh = torch.kthvalue(flat, k, dim=1).values.view(b, 1, 1)
                fused = torch.where(fused <= thresh, torch.zeros_like(fused), fused)

        # Cộng ma trận đơn vị I_N biểu diễn kết nối tắt (Residual Connection) và chuẩn hóa hàng
        a_hat = 0.5 * fused + 0.5 * eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        joint_att = torch.bmm(a_hat, joint_att)

    # Lấy sự chú ý từ token [CLS] (hàng 0) tới các token patch (cột 1..N-1)
    cls_to_patches = joint_att[:, 0, 1:].reshape(b, side, side)
    return _minmax_norm(cls_to_patches)


def _get_vit_encoder_layers(model: nn.Module) -> nn.ModuleList:
    """Tìm danh sách 12 tầng ViTLayer bên trong ViTBaseline hoặc ViTLoRA (hỗ trợ cả transformers 4.x và 5.x)."""
    backbone = getattr(model, "backbone", model)
    candidates = [backbone]
    if hasattr(backbone, "base_model"):
        candidates.append(backbone.base_model)
        if hasattr(backbone.base_model, "model"):
            candidates.append(backbone.base_model.model)
    for cand in candidates:
        if hasattr(cand, "layers") and isinstance(cand.layers, nn.ModuleList):
            return cand.layers
        if hasattr(cand, "encoder") and hasattr(cand.encoder, "layer") and isinstance(cand.encoder.layer, nn.ModuleList):
            return cand.encoder.layer
    raise AttributeError("Không tìm thấy `layers` hoặc `encoder.layer` trong mô hình ViT.")


def compute_layercam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: torch.Tensor | int | None = None,
    target_layers: Sequence[int] = (-4, -3, -2, -1),
    no_tumor_class: int = 0,
) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
    """Tính bản đồ nhiệt LayerCAM (Jiang et al., IEEE TIP 2021) kết hợp trích xuất Attention trong 1 lượt chạy.

    Khác với Grad-CAM (lấy trung bình toàn cục gradient), LayerCAM nhân từng phần tử giữa trọng số
    gradient dương W_{i,k} = ReLU(∂y^c / ∂F_{i,k}) với hoạt hóa F_{i,k} tại từng vị trí patch i, giúp
    bảo toàn chi tiết không gian của khối u nhỏ và đa ổ.

    Nếu `target_class=None`, tự động chọn lớp khối u có logit cao nhất (bỏ qua lớp `no_tumor_class` khi
    số lớp >= 2) để luôn tìm vùng nghi ngờ có u rõ nhất trên ảnh.
    """
    layers = _get_vit_encoder_layers(model)
    num_layers = len(layers)
    layer_indices = sorted({(idx if idx >= 0 else num_layers + idx) for idx in target_layers})
    for idx in layer_indices:
        if not (0 <= idx < num_layers):
            raise IndexError(f"Chỉ số tầng LayerCAM {idx} vượt ngoài phạm vi [0, {num_layers - 1}].")

    activations: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _fwd_hook(_module, _inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            if tensor_out.requires_grad:
                tensor_out.retain_grad()
                tensor_out.register_hook(lambda grad: gradients.__setitem__(layer_idx, grad.detach()))
            activations[layer_idx] = tensor_out

        return _fwd_hook

    was_training = model.training
    model.eval()
    try:
        for idx in layer_indices:
            handles.append(layers[idx].register_forward_hook(_make_hook(idx)))

        with torch.enable_grad():
            x_in = x.detach().clone().requires_grad_(True)
            if hasattr(model, "forward"):
                try:
                    out = model(x_in, return_dict=True, output_attentions=True)
                    logits = out["logits"]
                    pred_boxes = out.get("pred_boxes")
                    attentions = out.get("attentions")
                except TypeError:
                    # Tương thích với ViTBaseline
                    x_prep = model._prepare(x_in) if hasattr(model, "_prepare") else x_in
                    b_out = model.backbone(pixel_values=x_prep, output_attentions=True, return_dict=True)
                    cls_tok = b_out.last_hidden_state[:, 0]
                    logits = model.head(model.drop(cls_tok))
                    pred_boxes = None
                    attentions = b_out.attentions

            b, num_classes = logits.shape
            if target_class is None:
                if num_classes == 2:
                    cls_idx = torch.ones(b, dtype=torch.long, device=logits.device)
                else:
                    # Chọn lớp bệnh lý (khác no_tumor_class) có logit lớn nhất để định vị u
                    tumor_logits = logits.clone()
                    tumor_logits[:, no_tumor_class] = -1e9
                    cls_idx = tumor_logits.argmax(dim=-1)
            elif isinstance(target_class, int):
                cls_idx = torch.full((b,), int(target_class), dtype=torch.long, device=logits.device)
            else:
                cls_idx = target_class.to(device=logits.device, dtype=torch.long).view(b)

            selected = logits.gather(1, cls_idx.unsqueeze(1)).sum()
            model.zero_grad(set_to_none=True)
            selected.backward()

        b = x.shape[0]
        layer_maps = []
        for idx in layer_indices:
            act = activations[idx].detach().float()[:, 1:, :]    # (B, 196, D)
            grad = gradients.get(idx)
            if grad is None:
                continue
            grad_patch = grad.float()[:, 1:, :]                  # (B, 196, D)
            weights = F.relu(grad_patch)                         # ReLU(∂y^c / ∂F)
            cam = F.relu((weights * act).sum(dim=-1))            # (B, 196)
            side = int(math.isqrt(cam.shape[1]))
            cam_2d = cam.reshape(b, side, side)
            layer_maps.append(_minmax_norm(cam_2d))

        if layer_maps:
            fused_layercam = _minmax_norm(torch.stack(layer_maps, dim=0).mean(dim=0))
        else:
            fused_layercam = torch.zeros((b, 14, 14), device=x.device, dtype=torch.float32)

        return {
            "layercam_grid": fused_layercam,
            "logits": logits.detach(),
            "pred_boxes": pred_boxes.detach() if pred_boxes is not None else None,
            "attentions": [a.detach() for a in attentions] if attentions is not None else None,
            "target_class": cls_idx.detach(),
        }
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def extract_brain_parenchyma_mask(
    images_224: torch.Tensor,
    out_size: tuple[int, int] = (1024, 1024),
    erode_ratio: float = 0.055,
    input_norm: str = "imagenet",
) -> torch.Tensor:
    """Trích xuất Mặt nạ Nhu mô não (Brain Parenchyma Mask) loại bỏ viền hộp sọ sáng màu.

    Từ ảnh đầu vào `images_224` (B, 3, 224, 224):
      1. Khôi phục ảnh xám [0, 255] và nhị phân hóa vùng đầu (> 12) + lấp lỗ thủng (binary_fill_holes).
      2. Co hình thái học (Morphological Erosion) vào phía trong một khoảng `erode_ratio * min(H, W)`
         để cắt bỏ hoàn toàn vành xương sọ sáng màu (nơi Attention của ViT dễ bị nhiễu).
      3. Nội suy lên `out_size` (mặc định 1024x1024).
    """
    b, _, h, w = images_224.shape
    img = images_224.detach().cpu().float()
    if input_norm == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        img = (img * std + mean).clamp(0.0, 1.0)
    else:
        img = ((img * 0.5) + 0.5).clamp(0.0, 1.0)

    gray_batch = (img.mean(dim=1).numpy() * 255.0).astype(np.uint8)
    k_size = max(3, int(round(min(h, w) * erode_ratio)))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    masks_out = []
    for i in range(b):
        g = gray_batch[i]
        # Ngưỡng tách vùng đầu khỏi nền đen
        thr_val = max(12, int(np.percentile(g, 15)))
        binary = (g > thr_val).astype(np.uint8)
        filled = ndimage.binary_fill_holes(binary > 0).astype(np.uint8)
        eroded = cv2.erode(filled, kernel, iterations=1)

        # Dự phòng cho ảnh kiểm thử nhân tạo (nếu co xong bị rỗng hoặc quá nhỏ < 5% diện tích)
        if eroded.sum() < 0.05 * (h * w):
            fallback = np.zeros((h, w), dtype=np.uint8)
            pad_y, pad_x = max(2, int(0.06 * h)), max(2, int(0.06 * w))
            fallback[pad_y : h - pad_y, pad_x : w - pad_x] = 1
            eroded = fallback

        if out_size != (h, w):
            eroded = cv2.resize(eroded, (out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST)
        masks_out.append(eroded.astype(np.float32))

    return torch.from_numpy(np.stack(masks_out, axis=0)).to(images_224.device)


def fuse_rollout_and_layercam(
    rollout_grid: torch.Tensor,
    layercam_grid: torch.Tensor,
    parenchyma_mask: torch.Tensor | None = None,
    out_size: tuple[int, int] = (1024, 1024),
    alpha: float = 0.65,
    beta: float = 0.35,
) -> dict[str, torch.Tensor]:
    """Kết hợp bản đồ nhiệt Attention Rollout và LayerCAM, phóng lên `out_size` và lọc viền hộp sọ.

    Công thức kết hợp:
        M_linear = α * M_LayerCAM + (1 - α) * M_Rollout
        M_mult   = M_LayerCAM ⊙ M_Rollout
        M_fused  = normalize( ((1 - β) * M_linear + β * M_mult) ⊙ Mask_Parenchyma )
    """
    roll_up = F.interpolate(
        rollout_grid.unsqueeze(1).float(),
        size=out_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    cam_up = F.interpolate(
        layercam_grid.unsqueeze(1).float(),
        size=out_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    roll_up = _minmax_norm(roll_up)
    cam_up = _minmax_norm(cam_up)

    linear_comb = alpha * cam_up + (1.0 - alpha) * roll_up
    mult_comb = cam_up * roll_up
    fused = (1.0 - beta) * linear_comb + beta * mult_comb

    if parenchyma_mask is not None:
        pm = parenchyma_mask.to(device=fused.device, dtype=fused.dtype)
        if pm.shape[-2:] != out_size:
            pm = F.interpolate(pm.unsqueeze(1), size=out_size, mode="nearest").squeeze(1)
        fused = fused * pm

    fused = _minmax_norm(fused)
    return {
        "fused_heatmap": fused,
        "rollout_map": roll_up,
        "layercam_map": cam_up,
    }


def heatmap_to_box_and_points(
    heatmap: torch.Tensor | np.ndarray,
    parenchyma_mask: torch.Tensor | np.ndarray | None = None,
    threshold_ratio: float = 0.35,
    box_padding: float = 0.08,
) -> dict[str, np.ndarray]:
    """Trích xuất Hộp giới hạn [x_min, y_min, x_max, y_max] + Tâm điểm Dương/Âm từ bản đồ nhiệt 2D (H, W).

    Quy trình:
      1. Tìm tâm điểm cực đại `peak_pos = (x+, y+)` có giá trị nhiệt lớn nhất trong vùng nhu mô não.
      2. Phân ngưỡng bản đồ nhiệt tại `threshold_ratio * max_val` (kết hợp ngưỡng Otsu trên vùng nhu mô),
         tìm các thành phần liên thông (Connected Components) và chọn vùng liên thông chứa `peak_pos`.
      3. Đóng khung hộp giới hạn bao quanh thành phần liên thông đó và mở rộng biên thêm `+box_padding` (mặc định +8%).
      4. Chọn điểm âm `neg_point = (x-, y-)` nằm trong nhu mô não nhưng bên ngoài hộp giới hạn và có nhiệt độ thấp nhất.
    """
    hm = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else np.asarray(heatmap, dtype=np.float32)
    h, w = hm.shape[:2]

    if parenchyma_mask is not None:
        pm = (
            parenchyma_mask.detach().cpu().numpy()
            if isinstance(parenchyma_mask, torch.Tensor)
            else np.asarray(parenchyma_mask, dtype=np.float32)
        )
        if pm.shape[:2] != (h, w):
            pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        pm_bool = pm > 0.5
    else:
        pm_bool = np.ones((h, w), dtype=bool)

    masked_hm = np.where(pm_bool, hm, -1.0)
    if float(masked_hm.max()) <= 0.0:
        masked_hm = hm

    # 1. Tâm điểm cực đại (Peak Positive Point)
    flat_idx = int(np.argmax(masked_hm))
    py, px = divmod(flat_idx, w)
    peak_pos = np.array([float(px), float(py)], dtype=np.float32)
    peak_val = max(float(hm[py, px]), 1e-6)

    # 2. Nhị phân hóa vùng nóng và chọn thành phần liên thông chứa Peak Point
    thr = threshold_ratio * peak_val
    binary = ((hm >= thr) & pm_bool).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comp_label = int(labels[py, px]) if num_labels > 1 else 0
    if comp_label > 0:
        x0 = float(stats[comp_label, cv2.CC_STAT_LEFT])
        y0 = float(stats[comp_label, cv2.CC_STAT_TOP])
        bw = float(stats[comp_label, cv2.CC_STAT_WIDTH])
        bh = float(stats[comp_label, cv2.CC_STAT_HEIGHT])
        x1, y1 = x0 + bw, y0 + bh
    else:
        # Dự phòng hộp quanh tâm điểm nếu bản đồ nhiệt đồng nhất
        r_x, r_y = 0.12 * w, 0.12 * h
        x0, y0 = max(0.0, px - r_x), max(0.0, py - r_y)
        x1, y1 = min(float(w), px + r_x), min(float(h), py + r_y)

    # 3. Mở rộng biên hộp giới hạn (+box_padding)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    dx, dy = 0.5 * box_padding * bw, 0.5 * box_padding * bh
    x0 = float(np.clip(x0 - dx, 0.0, w))
    y0 = float(np.clip(y0 - dy, 0.0, h))
    x1 = float(np.clip(x1 + dx, 0.0, w))
    y1 = float(np.clip(y1 + dy, 0.0, h))
    if x1 <= x0:
        x1 = min(float(w), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(h), y0 + 1.0)
    box_px = np.array([x0, y0, x1, y1], dtype=np.float32)
    box_norm = box_px / np.array([w, h, w, h], dtype=np.float32)

    # 4. Tìm điểm âm (Negative Point) bên trong nhu mô não nhưng nằm ngoài hộp giới hạn
    outside_box = np.ones((h, w), dtype=bool)
    outside_box[int(y0) : int( math.ceil(y1)), int(x0) : int(math.ceil(x1))] = False
    cand_neg = pm_bool & outside_box
    if not cand_neg.any():
        cand_neg = outside_box if outside_box.any() else np.ones((h, w), dtype=bool)
    neg_map = np.where(cand_neg, hm, np.inf)
    neg_idx = int(np.argmin(neg_map))
    ny, nx = divmod(neg_idx, w)
    neg_point = np.array([float(nx), float(ny)], dtype=np.float32)

    point_coords = np.stack([peak_pos, neg_point], axis=0)      # (2, 2)
    point_labels = np.array([1, 0], dtype=np.int64)             # (2,)
    return {
        "box_px": box_px,
        "box_norm": box_norm,
        "peak_pos_xy": peak_pos,
        "neg_point_xy": neg_point,
        "point_coords": point_coords,
        "point_labels": point_labels,
    }


class AutoPromptGenerator(nn.Module):
    """Mô-đun Tự sinh Lời nhắc (Auto-Prompt Generator) tích hợp cả 2 cơ chế cho MedSAM / SAM.

    Args:
        model: mô hình `ViTLoRA` (hoặc `ViTBaseline`).
        mode:
            - `"auto"`: dùng hộp từ `ViTDetBoxHead` nếu mô hình có `box_head`, kết hợp với tâm điểm từ Heatmap;
                        nếu không có `box_head` thì dùng hộp từ Heatmap.
            - `"weakly_supervised"` (Cơ chế 1): sinh cả Hộp giới hạn lẫn Tâm điểm từ bản đồ nhiệt
              Attention Rollout + LayerCAM (dành cho Dataset 1 không có nhãn Box).
            - `"box_head"` (Cơ chế 2): lấy Hộp giới hạn từ nhánh `ViTDetBoxHead` + Tâm điểm từ bản đồ nhiệt
              (dành cho Dataset 2, 3, 4 có nhãn Box chuẩn).
            - `"hybrid_box"`: kết hợp có trọng số giữa hộp từ `ViTDetBoxHead` và hộp từ Heatmap.
        threshold: ngưỡng xác suất phát hiện khối u P(Tumor) >= τ (mặc định τ = 0.35 để ưu tiên Recall > 95%).
        box_padding: hệ số mở rộng hộp giới hạn δ (mặc định 0.08 = +8%).
        heatmap_threshold: ngưỡng tương đối để tách vùng nóng trên bản đồ nhiệt (mặc định 0.50).
        out_size: kích thước không gian đầu ra của bản đồ nhiệt và tọa độ prompt (mặc định 1024x1024 cho MedSAM).
    """

    VALID_MODES = ("auto", "weakly_supervised", "box_head", "hybrid_box")

    def __init__(
        self,
        model: nn.Module,
        mode: str = "auto",
        threshold: float = 0.35,
        box_padding: float = 0.08,
        heatmap_threshold: float = 0.35,
        alpha_cam: float = 0.65,
        beta_mult: float = 0.35,
        out_size: tuple[int, int] = (1024, 1024),
        no_tumor_class: int = 0,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode phải thuộc {self.VALID_MODES}, nhận được {mode!r}")
        self.model = model
        self.mode = mode
        self.threshold = float(threshold)
        self.box_padding = float(box_padding)
        self.heatmap_threshold = float(heatmap_threshold)
        self.alpha_cam = float(alpha_cam)
        self.beta_mult = float(beta_mult)
        self.out_size = tuple(out_size)
        self.no_tumor_class = int(no_tumor_class)

    def generate(
        self,
        images_224: torch.Tensor,
        mode: str | None = None,
        threshold: float | None = None,
        box_padding: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sinh Hybrid Prompt (Bounding Box + Peak Positive/Negative Points + Fused Heatmap) cho một batch ảnh 224x224."""
        active_mode = mode or self.mode
        if active_mode not in self.VALID_MODES:
            raise ValueError(f"mode phải thuộc {self.VALID_MODES}")
        tau = self.threshold if threshold is None else float(threshold)
        pad = self.box_padding if box_padding is None else float(box_padding)

        # Bước 1: Chạy LayerCAM + trích xuất Attention của 12 tầng và dự đoán từ Box Regression Head
        cam_out = compute_layercam(
            self.model,
            images_224,
            no_tumor_class=self.no_tumor_class,
        )
        logits = cam_out["logits"]
        probs = F.softmax(logits.float(), dim=-1)
        if probs.shape[-1] == 2:
            tumor_prob = probs[:, 1]
        else:
            tumor_prob = 1.0 - probs[:, self.no_tumor_class]
        has_tumor_pred = tumor_prob >= tau
        pred_class = probs.argmax(dim=-1)

        # Bước 2: Tính Attention Rollout qua 12 tầng Self-Attention
        attentions = cam_out["attentions"]
        if attentions is not None and len(attentions) > 0:
            rollout_grid = compute_attention_rollout(attentions)
        else:
            rollout_grid = cam_out["layercam_grid"].clone()

        # Bước 3: Lọc viền hộp sọ bằng Mặt nạ Nhu mô não & kết hợp Rollout + LayerCAM
        input_norm = getattr(self.model, "input_norm", "imagenet")
        parenchyma_mask = extract_brain_parenchyma_mask(
            images_224,
            out_size=self.out_size,
            input_norm=input_norm,
        )
        maps = fuse_rollout_and_layercam(
            rollout_grid=rollout_grid,
            layercam_grid=cam_out["layercam_grid"],
            parenchyma_mask=parenchyma_mask,
            out_size=self.out_size,
            alpha=self.alpha_cam,
            beta=self.beta_mult,
        )
        fused_heatmap = maps["fused_heatmap"]

        # Bước 4: Trích xuất Hộp giới hạn & Điểm Dương/Âm từ bản đồ nhiệt (Cơ chế 1)
        b = images_224.shape[0]
        hm_boxes_norm, hm_boxes_px, peak_pts, point_coords, point_labels = [], [], [], [], []
        for i in range(b):
            item = heatmap_to_box_and_points(
                fused_heatmap[i],
                parenchyma_mask=parenchyma_mask[i],
                threshold_ratio=self.heatmap_threshold,
                box_padding=pad,
            )
            hm_boxes_norm.append(item["box_norm"])
            hm_boxes_px.append(item["box_px"])
            peak_pts.append(item["peak_pos_xy"])
            point_coords.append(item["point_coords"])
            point_labels.append(item["point_labels"])

        device = images_224.device
        hm_boxes_norm_t = torch.from_numpy(np.stack(hm_boxes_norm, axis=0)).to(device=device, dtype=torch.float32)
        hm_boxes_1024_t = torch.from_numpy(np.stack(hm_boxes_px, axis=0)).to(device=device, dtype=torch.float32)
        peak_pts_t = torch.from_numpy(np.stack(peak_pts, axis=0)).to(device=device, dtype=torch.float32)
        point_coords_t = torch.from_numpy(np.stack(point_coords, axis=0)).to(device=device, dtype=torch.float32)
        point_labels_t = torch.from_numpy(np.stack(point_labels, axis=0)).to(device=device, dtype=torch.long)

        # Bước 5: Chọn Hộp giới hạn từ Nhánh Box Regression Head (Cơ chế 2) hoặc Heatmap (Cơ chế 1)
        raw_reg_boxes = cam_out["pred_boxes"]
        if raw_reg_boxes is not None:
            reg_boxes_norm = pad_normalized_boxes(raw_reg_boxes, padding_ratio=pad)
            scale = torch.tensor(
                [self.out_size[1], self.out_size[0], self.out_size[1], self.out_size[0]],
                device=device,
                dtype=torch.float32,
            )
            reg_boxes_1024 = reg_boxes_norm * scale
        else:
            reg_boxes_norm = None
            reg_boxes_1024 = None

        if active_mode == "weakly_supervised" or reg_boxes_norm is None:
            if active_mode == "box_head" and reg_boxes_norm is None:
                raise RuntimeError("Mô hình không có nhánh `box_head` nhưng yêu cầu mode='box_head'.")
            final_boxes_norm = hm_boxes_norm_t
            final_boxes_1024 = hm_boxes_1024_t
        elif active_mode in ("box_head", "auto"):
            final_boxes_norm = reg_boxes_norm
            final_boxes_1024 = reg_boxes_1024
        else:  # hybrid_box
            final_boxes_norm = 0.7 * reg_boxes_norm + 0.3 * hm_boxes_norm_t
            scale = torch.tensor(
                [self.out_size[1], self.out_size[0], self.out_size[1], self.out_size[0]],
                device=device,
                dtype=torch.float32,
            )
            final_boxes_1024 = final_boxes_norm * scale

        # Đảm bảo tâm điểm dương nằm bên trong hộp giới hạn cuối cùng; nếu lệch ra ngoài thì kẹp về vùng hợp lệ
        clamped_peak_x = torch.clamp(peak_pts_t[:, 0], min=final_boxes_1024[:, 0], max=final_boxes_1024[:, 2])
        clamped_peak_y = torch.clamp(peak_pts_t[:, 1], min=final_boxes_1024[:, 1], max=final_boxes_1024[:, 3])
        clamped_peak = torch.stack([clamped_peak_x, clamped_peak_y], dim=-1)
        point_coords_t = point_coords_t.clone()
        point_coords_t[:, 0, :] = clamped_peak

        return {
            "logits": logits,
            "probs": probs,
            "tumor_prob": tumor_prob,
            "has_tumor_pred": has_tumor_pred,
            "pred_class": pred_class,
            "boxes_norm": final_boxes_norm,
            "boxes_1024": final_boxes_1024,
            "heatmap_boxes_norm": hm_boxes_norm_t,
            "heatmap_boxes_1024": hm_boxes_1024_t,
            "regressed_boxes_norm": reg_boxes_norm,
            "regressed_boxes_1024": reg_boxes_1024,
            "peak_points_1024": peak_pts_t,
            "point_coords_1024": point_coords_t,
            "point_labels": point_labels_t,
            "fused_heatmap": fused_heatmap,
            "rollout_map": maps["rollout_map"],
            "layercam_map": maps["layercam_map"],
            "parenchyma_mask": parenchyma_mask,
        }

    def forward(self, images_224: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        return self.generate(images_224, **kwargs)
