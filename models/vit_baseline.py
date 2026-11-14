"""ViT-Base/16 baseline (google/vit-base-patch16-224-in21k) cho phân loại ảnh MRI u não.

Hai chế độ huấn luyện dùng CHUNG một kiến trúc (backbone + một lớp Linear trên token [CLS]):
  * linear_probe : đóng băng toàn bộ backbone, chỉ huấn luyện đầu phân loại (đo chất lượng đặc trưng tiền huấn luyện).
  * full_finetune: huấn luyện toàn bộ tham số (cận trên của việc thích nghi toàn phần).

Đầu vào là ảnh 224x224 đã chuẩn hóa ImageNet (đúng đầu ra `image_224` của UnifiedBrainMRIDataset).
Checkpoint google/vit-base-patch16-224-in21k được tiền huấn luyện với mean=std=0.5 nên mô-đun này tự
đổi từ chuẩn hóa ImageNet sang (x - 0.5) / 0.5 bên trong forward, người dùng không cần xử lý thêm.
"""
from __future__ import annotations

import torch
from torch import nn

MODEL_NAME = "google/vit-base-patch16-224-in21k"
MODES = ("linear_probe", "full_finetune")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ViTBaseline(nn.Module):
    """Args:
        num_classes: số lớp đầu ra (2 cho có/không u, 4 cho No Tumor/Glioma/Meningioma/Pituitary).
        mode: "linear_probe" hoặc "full_finetune".
        pretrained: True tải trọng số từ Hugging Face; False khởi tạo ngẫu nhiên (dùng cho kiểm thử).
        backbone: truyền sẵn một `transformers.ViTModel` để thay thế (kiểm thử offline bằng cấu hình nhỏ).
        input_norm: "imagenet" (mặc định, khớp UnifiedBrainMRIDataset) hoặc "none" nếu đầu vào đã là (x-0.5)/0.5.
    """

    def __init__(self, num_classes: int, mode: str = "full_finetune", model_name: str = MODEL_NAME,
                 pretrained: bool = True, backbone: nn.Module | None = None, dropout: float = 0.0,
                 input_norm: str = "imagenet", local_files_only: bool = False):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}")
        if input_norm not in ("imagenet", "none"):
            raise ValueError("input_norm phải là 'imagenet' hoặc 'none'")
        from transformers import ViTConfig, ViTModel

        if backbone is None:
            if pretrained:
                backbone = ViTModel.from_pretrained(model_name, add_pooling_layer=False,
                                                    local_files_only=local_files_only)
            else:
                backbone = ViTModel(ViTConfig.from_pretrained(model_name, local_files_only=local_files_only),
                                    add_pooling_layer=False)
        self.backbone = backbone
        self.hidden = backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden, num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        self.num_classes = num_classes
        self.input_norm = input_norm
        self.register_buffer("_in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.mode = mode
        self.set_mode(mode)

    # ------------------------------------------------------------------ chế độ
    def set_mode(self, mode: str) -> "ViTBaseline":
        if mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}")
        self.mode = mode
        freeze = mode == "linear_probe"
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        for p in self.head.parameters():
            p.requires_grad_(True)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "linear_probe":
            self.backbone.eval()          # tắt dropout trong backbone đóng băng
        return self

    # ------------------------------------------------------------------ suy luận
    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "none":
            return x
        return ((x * self._in_std + self._in_mean) - 0.5) / 0.5

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Token [CLS] sau LayerNorm cuối của ViT, hình (B, hidden)."""
        x = self._prepare(x)
        if self.mode == "linear_probe":
            with torch.no_grad():
                return self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.backbone(pixel_values=x).last_hidden_state[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x)))

    # ------------------------------------------------------------------ tiện ích
    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / total}

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float = 0.01):
        """Linear probe chỉ có nhóm head; full fine-tune dùng lr nhỏ cho backbone, lr lớn cho head."""
        groups = [{"params": [p for p in self.head.parameters()], "lr": lr_head, "weight_decay": weight_decay}]
        if self.mode == "full_finetune":
            groups.append({"params": [p for p in self.backbone.parameters()], "lr": lr_backbone,
                           "weight_decay": weight_decay})
        return groups

    def trainable_state_dict(self):
        """Linear probe chỉ cần lưu đầu phân loại (vài nghìn tham số) thay vì cả backbone ~330 MB."""
        sd = self.state_dict()
        if self.mode == "linear_probe":
            return {k: v for k, v in sd.items() if k.startswith("head.")}
        return sd


def build_vit_baseline(num_classes: int, mode: str, **kw) -> ViTBaseline:
    return ViTBaseline(num_classes=num_classes, mode=mode, **kw)
