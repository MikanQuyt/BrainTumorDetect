"""Bộ bọc (wrapper) SAM nguyên bản và MedSAM (ViT-B) dưới cùng một giao diện phân vùng theo hộp.

    seg = load_segmenter("sam" | "medsam", device="cuda")
    mask = seg.segment(image_rgb_uint8, box_xyxy)        # bool (H, W)

Mã nguồn mô hình dùng gói `segment-anything==1.0` của Meta (Apache-2.0); MedSAM dùng CHÍNH kiến trúc đó
với trọng số `medsam_vit_b.pth` nên cả hai nạp bằng `sam_model_registry["vit_b"]`.
Trọng số được tải bằng `python -m utils.checkpoints` vào `checkpoints/` (có kiểm tra MD5).

Hai đường suy luận khác nhau đúng như tài liệu chính thức của từng mô hình:
  * SAM   : `SamPredictor` (tự resize cạnh dài về 1024, chuẩn hóa mean/std của SAM), multimask_output=False.
  * MedSAM: resize 1024x1024, chuẩn hóa min-max về [0, 1], hộp quy đổi sang hệ 1024, sigmoid > 0.5
            (quy trình `medsam_inference` của bowang-lab/MedSAM).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.checkpoints import CHECKPOINTS, checkpoint_path, md5sum

MODEL_TYPE = "vit_b"


def _load_model(name: str, ckpt_dir=None, device: str = "cpu", verify: bool = True):
    from segment_anything import sam_model_registry

    path = checkpoint_path(name, ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}. Chạy: python -m utils.checkpoints --only {name}")
    if verify and md5sum(path) != CHECKPOINTS[name]["md5"]:
        raise RuntimeError(f"MD5 của {path.name} không khớp bản chuẩn; tải lại bằng utils.checkpoints.")
    model = sam_model_registry[MODEL_TYPE](checkpoint=None)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _clip_box(box, h, w):
    x0, y0, x1, y1 = (float(v) for v in box)
    return np.array([np.clip(x0, 0, w), np.clip(y0, 0, h), np.clip(x1, 0, w), np.clip(y1, 0, h)], dtype=np.float32)


class SamBoxSegmenter:
    """SAM gốc với prompt là một hộp giới hạn."""

    name = "SAM"

    def __init__(self, model, device: str = "cpu"):
        from segment_anything import SamPredictor

        self.device = device
        self.predictor = SamPredictor(model)

    @torch.no_grad()
    def segment(
        self,
        image_rgb: np.ndarray,
        box_xyxy=None,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        self.predictor.set_image(np.ascontiguousarray(image_rgb), image_format="RGB")
        box = _clip_box(box_xyxy, h, w) if box_xyxy is not None else None
        pts = np.asarray(point_coords, dtype=np.float32) if point_coords is not None else None
        lbls = np.asarray(point_labels, dtype=np.int32) if point_labels is not None else None
        masks, _, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            box=box,
            multimask_output=False,
        )
        return masks[0].astype(bool)


class MedSamBoxSegmenter:
    """MedSAM với prompt là một hộp giới hạn hoặc Hybrid Prompt (Box + Points)."""

    name = "MedSAM"

    def __init__(self, model, device: str = "cpu", threshold: float = 0.5):
        self.model, self.device, self.threshold = model, device, threshold

    @staticmethod
    def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
        img = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def box_to_1024(box_xyxy, h, w) -> np.ndarray:
        b = _clip_box(box_xyxy, h, w)
        return (b / np.array([w, h, w, h], dtype=np.float32) * 1024.0)[None, :]

    @torch.no_grad()
    def segment(
        self,
        image_rgb: np.ndarray,
        box_xyxy=None,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        use_points: bool = False,
    ) -> np.ndarray:
        """Suy luận MedSAM theo quy trình chuẩn bowang-lab/MedSAM.

        Lưu ý: Trọng số gốc `medsam_vit_b.pth` được huấn luyện thuần với Box Prompt (`points=None`).
        Tham số `use_points=True` chỉ nên bật sau khi đã tinh chỉnh Mask Decoder với Hybrid Prompt ở Tuần 4.
        """
        h, w = image_rgb.shape[:2]
        x = self.preprocess(image_rgb).to(self.device)
        embed = self.model.image_encoder(x)
        box = (
            torch.as_tensor(self.box_to_1024(box_xyxy, h, w), dtype=torch.float, device=self.device)[:, None, :]
            if box_xyxy is not None
            else None
        )
        points = None
        if use_points and point_coords is not None and point_labels is not None:
            pts = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            pts_1024 = pts / np.array([w, h], dtype=np.float32) * 1024.0
            coords_t = torch.as_tensor(pts_1024[None, :, :], dtype=torch.float, device=self.device)
            labels_t = torch.as_tensor(np.asarray(point_labels, dtype=np.int64).reshape(1, -1), device=self.device)
            points = (coords_t, labels_t)
        sparse, dense = self.model.prompt_encoder(points=points, boxes=box, masks=None)
        logits, _ = self.model.mask_decoder(
            image_embeddings=embed, image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        prob = F.interpolate(torch.sigmoid(logits), size=(h, w), mode="bilinear", align_corners=False)
        return (prob.squeeze().cpu().numpy() > self.threshold)


def load_segmenter(name: str, ckpt_dir=None, device: str = "cpu", verify: bool = True):
    """name ∈ {"sam", "medsam"}."""
    name = name.lower()
    if name not in CHECKPOINTS:
        raise ValueError(f"Mô hình không hỗ trợ: {name} (chọn {list(CHECKPOINTS)})")
    model = _load_model(name, ckpt_dir, device, verify)
    return SamBoxSegmenter(model, device) if name == "sam" else MedSamBoxSegmenter(model, device)
