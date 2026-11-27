"""MedSAM-LoRA (Tuần 4): đóng băng Image Encoder (ViT-B), chèn LoRA (r = 8) vào các khối Attention,
huấn luyện đầy đủ Mask Decoder.

Chiến lược tinh chỉnh
---------------------
  * `image_encoder` : đóng băng toàn bộ trọng số gốc; chèn LoRA vào phép chiếu `qkv` của CẢ 12 khối Attention.
                      LoRA chỉ cộng thêm vào nhánh Query và Value (ΔW_q, ΔW_v), nhánh Key giữ nguyên (chuẩn LoRA).
  * `prompt_encoder`: đóng băng (hộp giới hạn 1024x1024 -> embedding).
  * `mask_decoder`  : huấn luyện đầy đủ (Full fine-tuning).

LoRA được cài đặt trực tiếp (không phụ thuộc `peft`) vì `Attention.qkv` của SAM là một `nn.Linear` gộp (3·dim).
    h = W x + (α / r) · [B_q A_q x ; 0 ; B_v A_v x],   A ~ Kaiming-uniform, B = 0  (ΔW = 0 lúc khởi tạo)

Đầu vào/ra dùng cùng quy ước với `UnifiedBrainMRIDataset` (ảnh 1024x1024 giá trị [0, 1], hộp pixel hệ 1024):

    model = MedSamLoRA.from_checkpoint("checkpoints/medsam_vit_b.pth", lora_r=8)
    logits = model(image_1024, boxes_1024)           # (B, 1, 1024, 1024) logit (đã nội suy từ 256x256)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LoRAQKV", "MedSamLoRA", "jitter_boxes", "extract_peak_points", "inject_lora_into_image_encoder"]


class LoRAQKV(nn.Module):
    """Bọc `nn.Linear(dim, 3*dim)` (qkv của SAM) và cộng ΔW cho Query + Value với hạng r."""

    def __init__(self, qkv: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r phải > 0")
        self.qkv = qkv
        self.dim = qkv.in_features
        if qkv.out_features != 3 * self.dim:
            raise ValueError("qkv phải có out_features = 3 * in_features")
        self.r, self.scaling = r, float(alpha) / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A_q = nn.Linear(self.dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, self.dim, bias=False)
        self.lora_A_v = nn.Linear(self.dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, self.dim, bias=False)
        for a in (self.lora_A_q, self.lora_A_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.lora_B_q, self.lora_B_v):
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x)
        xd = self.drop(x)
        dq = self.lora_B_q(self.lora_A_q(xd)) * self.scaling
        dv = self.lora_B_v(self.lora_A_v(xd)) * self.scaling
        return out + torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)


def inject_lora_into_image_encoder(image_encoder: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> int:
    """Thay `blk.attn.qkv` của mọi khối bằng `LoRAQKV`; trả về số khối đã chèn."""
    n = 0
    for blk in image_encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
        n += 1
    return n


def jitter_boxes(
    boxes: torch.Tensor,
    max_px: float = 10.0,
    img_size: int = 1024,
    min_size: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Box Jittering: cộng nhiễu đều U(-max_px, +max_px) pixel cho từng tọa độ của hộp (B, 4) hệ `img_size`.

    Hộp được cắt trong [0, img_size] và đảm bảo cạnh tối thiểu `min_size` pixel (không suy biến).
    Mô phỏng sai số định vị của Auto-Prompt Generator (Nhánh 1) khi huấn luyện MedSAM-LoRA.
    """
    if max_px <= 0:
        return boxes
    b = boxes.float()
    noise = (torch.rand(b.shape, generator=generator, device=b.device) * 2.0 - 1.0) * max_px
    j = (b + noise).clamp(0.0, float(img_size))
    x0 = torch.minimum(j[:, 0], j[:, 2])
    x1 = torch.maximum(j[:, 0], j[:, 2])
    y0 = torch.minimum(j[:, 1], j[:, 3])
    y1 = torch.maximum(j[:, 1], j[:, 3])
    x1 = torch.maximum(x1, (x0 + min_size).clamp(max=float(img_size)))
    y1 = torch.maximum(y1, (y0 + min_size).clamp(max=float(img_size)))
    x0 = torch.minimum(x0, x1 - min_size).clamp(min=0.0)
    y0 = torch.minimum(y0, y1 - min_size).clamp(min=0.0)
    return torch.stack([x0, y0, x1, y1], dim=1).to(boxes.dtype)


@torch.no_grad()
def extract_peak_points(
    images_1024: torch.Tensor,
    boxes_1024: torch.Tensor,
    inner_margin: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trích xuất tâm điểm dương cực đại (Peak Positive Point) bên trong hộp mà KHÔNG dùng gt_mask.

    Kết hợp:
      1. Tương phản cường độ cục bộ so với vành nhu mô bao quanh hộp (bắt cả khối u tăng sáng lẫn giảm sáng).
      2. Bất đối xứng bán cầu não trái-phải.
      3. Trọng số ưu tiên vùng lõi trung tâm hộp (giảm thiểu chọn nhầm mép vỏ xương/ranh giới hộp).

    Returns:
        coords: Tensor hình (B, 1, 2) tọa độ [x, y] theo pixel ảnh gốc.
        labels: Tensor hình (B, 1) toàn giá trị 1 (điểm dương foreground).
    """
    b, _, h, w = images_1024.shape
    device = images_1024.device
    gray = images_1024.detach().float().mean(dim=1, keepdim=True)
    gh, gw = min(h, 256), min(w, 256)
    if (gh, gw) != (h, w):
        work = F.interpolate(gray, size=(gh, gw), mode="bilinear", align_corners=False)
    else:
        work = gray
    k = 5 if min(gh, gw) >= 32 else 3
    smooth = F.avg_pool2d(work, kernel_size=k, stride=1, padding=k // 2)
    asym = (smooth - torch.flip(smooth, dims=[-1])).abs()

    sx, sy = gw / float(w), gh / float(h)
    coords = torch.zeros((b, 1, 2), dtype=torch.float32, device=device)
    labels = torch.ones((b, 1), dtype=torch.long, device=device)

    for i in range(b):
        bx0, by0, bx1, by1 = boxes_1024[i].detach().float().tolist()
        x0, x1 = max(0.0, min(bx0, bx1)), min(float(w), max(bx0, bx1))
        y0, y1 = max(0.0, min(by0, by1)), min(float(h), max(by0, by1))
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)

        # Thu hẹp biên hộp (inner_margin) để điểm dương nằm trong lõi khối u
        mx = inner_margin * bw
        my = inner_margin * bh
        ix0_px = min(max(x0 + mx, x0), x1)
        ix1_px = max(min(x1 - mx, x1), ix0_px)
        iy0_px = min(max(y0 + my, y0), y1)
        iy1_px = max(min(y1 - my, y1), iy0_px)

        gx0 = min(max(int(math.floor(ix0_px * sx)), 0), gw - 1)
        gx1 = max(min(int(math.ceil(ix1_px * sx)), gw), gx0 + 1)
        gy0 = min(max(int(math.floor(iy0_px * sy)), 0), gh - 1)
        gy1 = max(min(int(math.ceil(iy1_px * sy)), gh), gy0 + 1)

        # Vành ngoài mở rộng +25% hộp để ước lượng nền nhu mô lân cận (tính |I - bg_mean|)
        ox0 = max(int(math.floor((x0 - 0.25 * bw) * sx)), 0)
        ox1 = min(int(math.ceil((x1 + 0.25 * bw) * sx)), gw)
        oy0 = max(int(math.floor((y0 - 0.25 * bh) * sy)), 0)
        oy1 = min(int(math.ceil((y1 + 0.25 * bh) * sy)), gh)
        outer_patch = smooth[i, 0, oy0:oy1, ox0:ox1]
        bg_ref = outer_patch.mean() if outer_patch.numel() > 0 else smooth[i, 0].mean()

        crop_s = smooth[i, 0, gy0:gy1, gx0:gx1]
        crop_a = asym[i, 0, gy0:gy1, gx0:gx1]
        contrast = (crop_s - bg_ref).abs()

        def _norm01(t: torch.Tensor) -> torch.Tensor:
            d = t.max() - t.min()
            return (t - t.min()) / d if float(d) > 1e-6 else torch.zeros_like(t)

        sal = 0.65 * _norm01(contrast) + 0.35 * _norm01(crop_a)

        ch, cw = sal.shape
        cy = torch.linspace(-1.0, 1.0, ch, device=device).view(ch, 1)
        cx = torch.linspace(-1.0, 1.0, cw, device=device).view(1, cw)
        center_prior = (1.0 - 0.30 * (cx * cx + cy * cy)).clamp(min=0.40)
        score = (sal + 0.25) * center_prior

        flat_idx = int(torch.argmax(score).item())
        py_idx, px_idx = divmod(flat_idx, cw)
        px = ((gx0 + px_idx + 0.5) / sx)
        py = ((gy0 + py_idx + 0.5) / sy)
        coords[i, 0, 0] = min(max(px, x0), x1)
        coords[i, 0, 1] = min(max(py, y0), y1)

    return coords, labels


class MedSamLoRA(nn.Module):
    """MedSAM (ViT-B) + LoRA(r) ở Image Encoder (đóng băng) + Mask Decoder huấn luyện đầy đủ."""

    def __init__(self, sam_model: nn.Module, lora_r: int = 8, lora_alpha: float | None = None, lora_dropout: float = 0.0):
        super().__init__()
        self.sam = sam_model
        self.lora_r = lora_r
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else 2 * lora_r)
        for p in self.sam.parameters():
            p.requires_grad = False                                   # đóng băng Image Encoder + Prompt Encoder
        self.n_lora_blocks = inject_lora_into_image_encoder(
            self.sam.image_encoder, r=lora_r, alpha=self.lora_alpha, dropout=lora_dropout)
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True                                    # huấn luyện đầy đủ Mask Decoder

    # ------------------------------------------------------------------ khởi tạo
    @classmethod
    def from_checkpoint(cls, ckpt_path, lora_r: int = 8, lora_alpha: float | None = None,
                        lora_dropout: float = 0.0, device: str = "cpu", verify: bool = True):
        """Nạp `medsam_vit_b.pth` (kiểm tra MD5 như Tuần 2) rồi bọc LoRA."""
        from segment_anything import sam_model_registry

        from utils.checkpoints import CHECKPOINTS, md5sum

        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}. Chạy: python -m utils.checkpoints --only medsam")
        if verify and md5sum(path) != CHECKPOINTS["medsam"]["md5"]:
            raise RuntimeError(f"MD5 của {path.name} không khớp bản chuẩn MedSAM.")
        sam = sam_model_registry["vit_b"](checkpoint=None)
        sam.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(sam, lora_r, lora_alpha, lora_dropout).to(device)

    # ------------------------------------------------------------------ tham số
    def lora_parameters(self):
        return [p for n, p in self.sam.image_encoder.named_parameters() if "lora_" in n]

    def decoder_parameters(self):
        return list(self.sam.mask_decoder.parameters())

    def param_groups(self, lr_lora: float = 1e-4, lr_decoder: float = 1e-4, weight_decay: float = 0.01):
        return [
            {"params": self.lora_parameters(), "lr": lr_lora, "weight_decay": 0.0},
            {"params": self.decoder_parameters(), "lr": lr_decoder, "weight_decay": weight_decay},
        ]

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        lora = sum(p.numel() for p in self.lora_parameters())
        dec = sum(p.numel() for p in self.decoder_parameters())
        enc_frozen = sum(p.numel() for n, p in self.sam.image_encoder.named_parameters() if "lora_" not in n)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / max(total, 1),
                "lora_params": lora, "mask_decoder_params": dec, "image_encoder_frozen_params": enc_frozen}

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam.image_encoder.eval()        # Encoder đóng băng: giữ ở eval
        self.sam.prompt_encoder.eval()
        for m in self.sam.image_encoder.modules():
            if isinstance(m, LoRAQKV):
                m.drop.train(mode)           # Giữ LoRA dropout theo mode train/eval
        return self

    # ------------------------------------------------------------------ suy luận
    def forward(
        self,
        images_1024: torch.Tensor,
        boxes_1024: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        upsample: bool = True,
    ) -> torch.Tensor:
        """images_1024: (B, 3, 1024, 1024) [0,1]; boxes_1024: (B, 4) pixel hệ 1024. Trả logit (B, 1, H, W).

        Chú ý: `MaskDecoder` của `segment-anything==1.0` dùng `repeat_interleave(image_embeddings, B)` khi B > 1,
        nên giải mã từng mẫu trong batch rồi ghép lại (`torch.cat`) để hỗ trợ an toàn mọi batch_size >= 1.
        """
        embed = self.sam.image_encoder(images_1024)
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        bs = images_1024.shape[0]

        pt_coords = pt_labels = None
        if points is not None:
            pt_coords = points[0].float()
            pt_labels = points[1].long()
            if pt_coords.ndim == 2:
                pt_coords = pt_coords.unsqueeze(1)
            if pt_labels.ndim == 1:
                pt_labels = pt_labels.unsqueeze(1)

        low_res_list = []
        for i in range(bs):
            box_i = boxes_1024[i : i + 1].float()[:, None, :]
            pts_i = (pt_coords[i : i + 1], pt_labels[i : i + 1]) if pt_coords is not None else None
            with torch.no_grad():
                sparse_i, dense_i = self.sam.prompt_encoder(points=pts_i, boxes=box_i, masks=None)
            low_i, _ = self.sam.mask_decoder(
                image_embeddings=embed[i : i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_i,
                dense_prompt_embeddings=dense_i,
                multimask_output=False,
            )
            low_res_list.append(low_i)

        low_res = torch.cat(low_res_list, dim=0)
        if not upsample:
            return low_res
        return F.interpolate(low_res, size=images_1024.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict_masks(
        self,
        images_1024: torch.Tensor,
        boxes_1024: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(images_1024, boxes_1024, points=points)) > threshold

    # ------------------------------------------------------------------ lưu / nạp
    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if "lora_" in k or "mask_decoder" in k}

    def save_checkpoint(self, path, extra: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.trainable_state_dict(), "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha, **(extra or {})}, path)

    def load_checkpoint(self, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        missing, unexpected = self.load_state_dict(ck["state_dict"], strict=False)
        bad = [k for k in unexpected]
        if bad:
            raise RuntimeError(f"Checkpoint không khớp kiến trúc (khóa thừa): {bad[:3]}")
        return ck
