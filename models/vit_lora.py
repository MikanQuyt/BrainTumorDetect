"""Kiến trúc ViT-LoRA (Nhánh 1 — Tuần 3) kết hợp nhánh hồi quy hộp giới hạn ViTDet SFP.

Thành phần cốt lõi:
  1. Backbone `google/vit-base-patch16-224-in21k` được gắn các ma trận hạng thấp LoRA (A, B) vào
     các tầng `query` và `value` của toàn bộ 12 khối Self-Attention thông qua thư viện `peft`.
  2. Đầu phân loại (Classification Head) trên token `[CLS]` kết hợp hàm mất mát
     Weighted Cross-Entropy / Focal Loss có Label Smoothing (ε = 0.05).
  3. Nhánh hồi quy hộp giới hạn `ViTDetBoxHead` (Simple Feature Pyramid + MLP Box Regression Head)
     tái tạo bản đồ đặc trưng không gian 2D (14x14 -> 28x28) từ các token patch của ViT để dự đoán
     trực tiếp tọa độ hộp giới hạn chuẩn hóa [x_min, y_min, x_max, y_max] ∈ [0, 1]^4, huấn luyện
     bằng tổ hợp GIoU Loss + L1 Loss (dành cho Dataset 2, 3, 4 có nhãn Box chuẩn).
  4. Hỗ trợ trích xuất trọng số Attention của cả 12 tầng (`output_attentions=True`, `attn_implementation="eager"`)
     và đặc trưng/gradient của từng tầng ẩn để phục vụ bộ tự sinh lời nhắc Attention Rollout + LayerCAM.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.vit_baseline import IMAGENET_MEAN, IMAGENET_STD, MODEL_NAME

__all__ = [
    "ViTDetBoxHead",
    "ViTLoRA",
    "build_vit_lora",
    "pad_normalized_boxes",
]


_MODULE_ALIASES = {
    "query": ("query", "q_proj"),
    "q_proj": ("q_proj", "query"),
    "key": ("key", "k_proj"),
    "k_proj": ("k_proj", "key"),
    "value": ("value", "v_proj"),
    "v_proj": ("v_proj", "value"),
}


def _resolve_vit_target_modules(backbone: nn.Module, requested: Sequence[str]) -> tuple[str, ...]:
    """Tự động ánh xạ tên tầng Attention giữa transformers 4.x (`query`, `value`) và 5.x (`q_proj`, `v_proj`)."""
    existing_leaf_names = {name.split(".")[-1] for name, _ in backbone.named_modules() if name}
    resolved = []
    for req in requested:
        candidates = _MODULE_ALIASES.get(req, (req,))
        matched = next((c for c in candidates if c in existing_leaf_names), req)
        if matched not in resolved:
            resolved.append(matched)
    return tuple(resolved)


class ViTDetBoxHead(nn.Module):
    """Nhánh hồi quy hộp giới hạn theo kiến trúc ViTDet Simple Feature Pyramid (SFP).

    Từ dãy token patch của ViT hình (B, N_patches, D) với N_patches = H_p * W_p (14x14 cho ảnh 224, patch 16):
      1. Chuyển đổi sang lưới đặc trưng 2D (B, D, H_p, W_p).
      2. Qua nhánh SFP gồm tích chập 1x1 giảm chiều + tích chập chuyển vị (ConvTranspose2d 2x lên 28x28)
         + tích chập tinh chỉnh 3x3 để tăng độ phân giải không gian cho biên khối u.
      3. Kết hợp đặc trưng gộp đa tỷ lệ (Adaptive Avg + Max Pooling ở 14x14 và 28x28) cùng token toàn cục [CLS]
         đưa vào MLP dự đoán tâm và kích thước hộp (cx, cy, w, h) ∈ (0, 1)^4, sau đó quy đổi sang
         [x_min, y_min, x_max, y_max] ∈ [0, 1]^4 đảm bảo luôn hợp lệ (x_max >= x_min, y_max >= y_min).
    """

    def __init__(self, hidden_size: int = 768, sfp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.sfp_dim = sfp_dim
        num_groups = min(32, sfp_dim)
        while sfp_dim % num_groups != 0 and num_groups > 1:
            num_groups //= 2

        # Nhánh 14x14 (P4)
        self.lateral = nn.Sequential(
            nn.Conv2d(hidden_size, sfp_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, sfp_dim),
            nn.GELU(),
        )
        # Nhánh phóng đại 28x28 (P3 - Simple Feature Pyramid)
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(sfp_dim, sfp_dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(num_groups, sfp_dim),
            nn.GELU(),
            nn.Conv2d(sfp_dim, sfp_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, sfp_dim),
            nn.GELU(),
        )
        # Nhúng vị trí không gian 2D (Coordinate positional prior) vào bản đồ P3 để nhạy tọa độ (x, y)
        self.coord_conv = nn.Sequential(
            nn.Conv2d(sfp_dim + 2, sfp_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, sfp_dim),
            nn.GELU(),
        )
        # Trọng số chú ý không gian (Spatial Attention Pooling) trên bản đồ P3
        self.spatial_attn = nn.Conv2d(sfp_dim, 1, kernel_size=1)
        self.cls_proj = nn.Sequential(
            nn.Linear(hidden_size, sfp_dim),
            nn.LayerNorm(sfp_dim),
            nn.GELU(),
        )

        # Tổng hợp: [spatial_attn_pool(P3), max_pool(P3), avg_pool(P4), cls_proj] -> 4 * sfp_dim
        in_mlp = sfp_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_mlp, sfp_dim),
            nn.LayerNorm(sfp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sfp_dim, sfp_dim // 2),
            nn.GELU(),
            nn.Linear(sfp_dim // 2, 4),
        )
        self._init_weights()

    def _init_weights(self):
        # Khởi tạo lớp cuối sao cho sigmoid(bias) ~ (cx=0.5, cy=0.5, w=0.4, h=0.4) -> box khởi đầu [0.3, 0.3, 0.7, 0.7]
        last_linear = self.mlp[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            nn.init.constant_(last_linear.bias, 0.0)
            with torch.no_grad():
                last_linear.bias[2:].fill_(-0.4)

    @staticmethod
    def _make_coord_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return grid

    @staticmethod
    def cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
        cx, cy, bw, bh = cxcywh.unbind(dim=-1)
        x1 = (cx - 0.5 * bw).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * bh).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * bw).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * bh).clamp(0.0, 1.0)
        # Đảm bảo hộp không suy biến hoàn toàn (x2 > x1, y2 > y1 tối thiểu 1e-4)
        x2 = torch.maximum(x2, (x1 + 1e-4).clamp(max=1.0))
        y2 = torch.maximum(y2, (y1 + 1e-4).clamp(max=1.0))
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, patch_tokens: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """Args:
            patch_tokens: (B, N_patches, D)
            cls_token: (B, D)
        Returns:
            pred_boxes: (B, 4) chuẩn hóa [x_min, y_min, x_max, y_max] ∈ [0, 1].
        """
        b, n, d = patch_tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"Số lượng patch token ({n}) không phải là số chính phương (cần H_p * W_p).")

        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(b, d, side, side)
        p4 = self.lateral(feat_2d)                         # (B, sfp_dim, 14, 14)
        p3 = self.up_p3(p4)                                # (B, sfp_dim, 28, 28)

        coord_grid = self._make_coord_grid(b, p3.shape[2], p3.shape[3], p3.device, p3.dtype)
        p3_coord = self.coord_conv(torch.cat([p3, coord_grid], dim=1))

        # Spatial attention pooling + Differentiable Soft-Argmax trên P3
        attn_map = self.spatial_attn(p3_coord)                        # (B, 1, 28, 28)
        attn_logits = attn_map.flatten(2)                             # (B, 1, H*W)
        attn_weights = F.softmax(attn_logits, dim=-1)                 # (B, 1, H*W)
        p3_flat = p3_coord.flatten(2)                                 # (B, sfp_dim, H*W)
        p3_attn_pool = torch.bmm(p3_flat, attn_weights.transpose(1, 2)).squeeze(-1)  # (B, sfp_dim)

        # Kỳ vọng tọa độ tâm (mu_x, mu_y) khả vi từ bản đồ chú ý không gian P3
        grid_flat = coord_grid.flatten(2)                             # (B, 2, H*W)
        mu_xy = torch.bmm(grid_flat, attn_weights.transpose(1, 2)).squeeze(-1)       # (B, 2) ∈ [0, 1]^2

        p3_max_pool = F.adaptive_max_pool2d(p3_coord, 1).flatten(1)   # (B, sfp_dim)
        p4_avg_pool = F.adaptive_avg_pool2d(p4, 1).flatten(1)         # (B, sfp_dim)
        cls_emb = self.cls_proj(cls_token)                            # (B, sfp_dim)

        fused = torch.cat([p3_attn_pool, p3_max_pool, p4_avg_pool, cls_emb], dim=-1)
        raw_out = self.mlp(fused)
        delta_xy = 0.25 * torch.tanh(raw_out[:, :2])
        cxcy = (mu_xy + delta_xy).clamp(0.02, 0.98)
        wh = torch.sigmoid(raw_out[:, 2:]).clamp(0.04, 0.95)
        cxcywh = torch.cat([cxcy, wh], dim=-1)
        return self.cxcywh_to_xyxy(cxcywh)

    def spatial_heatmap(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Trích xuất bản đồ chú ý không gian 28x28 đã chuẩn hóa [0, 1] từ nhánh ViTDet SFP."""
        b, n, d = patch_tokens.shape
        side = int(math.isqrt(n))
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(b, d, side, side)
        p4 = self.lateral(feat_2d)
        p3 = self.up_p3(p4)
        coord_grid = self._make_coord_grid(b, p3.shape[2], p3.shape[3], p3.device, p3.dtype)
        p3_coord = self.coord_conv(torch.cat([p3, coord_grid], dim=1))
        attn_map = torch.sigmoid(self.spatial_attn(p3_coord).squeeze(1))  # (B, 28, 28)
        flat = attn_map.flatten(-2)
        mn = flat.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        mx = flat.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        return ((attn_map - mn) / (mx - mn + 1e-7)).clamp(0.0, 1.0)


class ViTLoRA(nn.Module):
    """Kiến trúc ViT-Base/16 tích hợp LoRA (PEFT) vào 12 tầng Attention (`query`, `value`) và nhánh `ViTDetBoxHead`.

    Args:
        num_classes: số lớp phân loại (2 cho nhị phân Có/Không u, 4 cho đa lớp).
        lora_r: hạng ma trận LoRA r (mặc định 8; hỗ trợ khảo sát r ∈ {4, 8, 16, 32}).
        lora_alpha: hệ số tỷ lệ LoRA α (mặc định 16).
        lora_dropout: xác suất dropout trên nhánh LoRA (mặc định 0.1).
        target_modules: danh sách tên module đích trong khối Attention (mặc định `("query", "value")`).
        use_box_head: bật nhánh `ViTDetBoxHead` để hồi quy hộp giới hạn chuẩn hóa [0, 1]^4.
        sfp_dim: số kênh ẩn của nhánh ViTDet SFP (mặc định 256).
        dropout: xác suất dropout trước đầu phân loại và trong MLP box head.
        model_name: tên checkpoint Hugging Face (mặc định `google/vit-base-patch16-224-in21k`).
        pretrained: True nạp trọng số tiền huấn luyện; False khởi tạo ngẫu nhiên từ config.
        backbone: cho phép truyền sẵn một `ViTModel` (dùng trong unit test offline nhanh).
        input_norm: `"imagenet"` (tự chuyển từ chuẩn hóa ImageNet sang [-1, 1] của ViT-in21k) hoặc `"none"`.
        local_files_only: chỉ đọc từ cache cục bộ của Hugging Face.
    """

    def __init__(
        self,
        num_classes: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Sequence[str] = ("query", "value"),
        use_box_head: bool = True,
        sfp_dim: int = 128,
        dropout: float = 0.1,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
        backbone: nn.Module | None = None,
        input_norm: str = "imagenet",
        local_files_only: bool = False,
    ):
        super().__init__()
        if lora_r <= 0:
            raise ValueError(f"lora_r phải > 0, nhận được {lora_r}")
        if input_norm not in ("imagenet", "none"):
            raise ValueError("input_norm phải là 'imagenet' hoặc 'none'")

        from peft import LoraConfig, get_peft_model
        from transformers import ViTConfig, ViTModel

        if backbone is None:
            if pretrained:
                backbone = ViTModel.from_pretrained(
                    model_name,
                    add_pooling_layer=False,
                    attn_implementation="eager",
                    local_files_only=local_files_only,
                )
            else:
                cfg = ViTConfig.from_pretrained(model_name, local_files_only=local_files_only)
                cfg._attn_implementation = "eager"
                backbone = ViTModel(cfg, add_pooling_layer=False)
        else:
            if hasattr(backbone, "config"):
                backbone.config._attn_implementation = "eager"
            if hasattr(backbone, "set_attn_implementation"):
                try:
                    backbone.set_attn_implementation("eager")
                except Exception:
                    pass

        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = tuple(target_modules)
        self.resolved_target_modules = _resolve_vit_target_modules(backbone, self.target_modules)

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.resolved_target_modules),
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.hidden = int(self.backbone.config.hidden_size)
        self.num_layers = int(self.backbone.config.num_hidden_layers)
        self.num_classes = int(num_classes)
        self.input_norm = input_norm
        self.use_box_head = bool(use_box_head)

        # Đầu phân loại (Classification Head)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden, self.num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        # Nhánh hồi quy hộp giới hạn (Box Regression Head / ViTDet SFP)
        self.box_head: ViTDetBoxHead | None = (
            ViTDetBoxHead(hidden_size=self.hidden, sfp_dim=sfp_dim, dropout=dropout)
            if self.use_box_head
            else None
        )

        self.register_buffer("_in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ kiểm tra & thống kê tham số
    def lora_module_names(self) -> list[str]:
        """Danh sách các tầng đã được gắn ma trận hạng thấp LoRA (lora_A & lora_B)."""
        names = []
        for name, module in self.backbone.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                names.append(name)
        return names

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        box_params = (
            sum(p.numel() for p in self.box_head.parameters() if p.requires_grad)
            if self.box_head is not None
            else 0
        )
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
            "backbone_total": backbone_total,
            "lora_params": lora_params,
            "lora_pct_of_backbone": 100.0 * lora_params / max(backbone_total, 1),
            "head_params": head_params,
            "box_head_params": box_params,
            "num_lora_modules": len(self.lora_module_names()),
        }

    def param_groups(
        self,
        lr_lora: float = 3e-4,
        lr_head: float = 1e-3,
        lr_box: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """Chia nhóm tốc độ học: nhánh LoRA (lr_lora), đầu phân loại (lr_head) và đầu hồi quy Box (lr_box)."""
        lora_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        groups = [
            {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay, "name": "lora"},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
        ]
        if self.box_head is not None:
            box_params = [p for p in self.box_head.parameters() if p.requires_grad]
            if box_params:
                groups.append({"params": box_params, "lr": lr_box, "weight_decay": weight_decay, "name": "box_head"})
        return groups

    # ------------------------------------------------------------------ tiền xử lý & suy luận
    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "none":
            return x
        return ((x * self._in_std + self._in_mean) - 0.5) / 0.5

    def encode(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Chạy qua backbone ViT-LoRA và trả về BaseModelOutput (last_hidden_state, attentions, hidden_states)."""
        x_prep = self._prepare(x)
        return self.backbone(
            pixel_values=x_prep,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_box: bool = False,
        return_dict: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        out = self.encode(
            x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        seq = out.last_hidden_state              # (B, 1 + 196, D)
        cls_token = seq[:, 0]                    # (B, D)
        patch_tokens = seq[:, 1:]                # (B, 196, D)
        logits = self.head(self.drop(cls_token))

        pred_boxes = None
        if (return_box or return_dict) and self.box_head is not None:
            pred_boxes = self.box_head(patch_tokens, cls_token)

        if return_dict:
            return {
                "logits": logits,
                "pred_boxes": pred_boxes,
                "cls_features": cls_token,
                "patch_tokens": patch_tokens,
                "attentions": out.attentions,
                "hidden_states": out.hidden_states,
            }
        if return_box:
            if pred_boxes is None:
                raise RuntimeError("Mô hình được khởi tạo với use_box_head=False nên không thể trả về pred_boxes.")
            return logits, pred_boxes
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.35,
        no_tumor_class: int = 0,
        box_padding: float = 0.08,
    ) -> dict[str, torch.Tensor]:
        """Suy luận nhanh phân loại có điều chỉnh ngưỡng τ (mặc định 0.35) + tọa độ Box (chuẩn hóa & hệ 1024)."""
        was_training = self.training
        self.eval()
        try:
            res = self.forward(x, return_dict=True)
            logits = res["logits"]
            probs = F.softmax(logits.float(), dim=-1)
            if self.num_classes == 2:
                tumor_prob = probs[:, 1]
            else:
                tumor_prob = 1.0 - probs[:, no_tumor_class]
            has_tumor_pred = tumor_prob >= threshold
            pred_class = probs.argmax(dim=-1)

            boxes_norm = res["pred_boxes"]
            if boxes_norm is not None and box_padding > 0.0:
                boxes_norm = pad_normalized_boxes(boxes_norm, padding_ratio=box_padding)
            boxes_1024 = boxes_norm * 1024.0 if boxes_norm is not None else None
            return {
                "logits": logits,
                "probs": probs,
                "tumor_prob": tumor_prob,
                "has_tumor_pred": has_tumor_pred,
                "pred_class": pred_class,
                "pred_boxes_norm": boxes_norm,
                "pred_boxes_1024": boxes_1024,
            }
        finally:
            self.train(was_training)

    # ------------------------------------------------------------------ lưu & nạp trọng số nhẹ (LoRA + Head + BoxHead)
    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Chỉ trích xuất các tensor có `requires_grad=True` (lora_A, lora_B, head, box_head) — dung lượng ~2–5 MB."""
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        # Bao gồm thêm các buffer của GroupNorm/LayerNorm nếu có
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k in trainable_keys or k.startswith(("head.", "box_head."))
        }

    def save_lora_checkpoint(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.trainable_state_dict(),
            "config": {
                "num_classes": self.num_classes,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": list(self.target_modules),
                "use_box_head": self.use_box_head,
                "input_norm": self.input_norm,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def load_lora_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Các khóa không hợp lệ khi nạp LoRA checkpoint: {unexpected}")
        return ckpt


def pad_normalized_boxes(boxes: torch.Tensor, padding_ratio: float = 0.08) -> torch.Tensor:
    """Mở rộng biên hộp giới hạn chuẩn hóa [x_min, y_min, x_max, y_max] thêm δ% (mặc định +8%) quanh tâm."""
    if padding_ratio == 0.0:
        return boxes
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    dx = 0.5 * padding_ratio * w
    dy = 0.5 * padding_ratio * h
    nx1 = (x1 - dx).clamp(0.0, 1.0)
    ny1 = (y1 - dy).clamp(0.0, 1.0)
    nx2 = (x2 + dx).clamp(0.0, 1.0)
    ny2 = (y2 + dy).clamp(0.0, 1.0)
    return torch.stack([nx1, ny1, nx2, ny2], dim=-1)


def build_vit_lora(num_classes: int = 4, lora_r: int = 8, lora_alpha: int = 16, **kwargs) -> ViTLoRA:
    return ViTLoRA(num_classes=num_classes, lora_r=lora_r, lora_alpha=lora_alpha, **kwargs)
