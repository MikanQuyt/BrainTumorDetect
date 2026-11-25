"""Unit tests cho Tuần 4: MedSAM-LoRA (r=8 + Mask Decoder), SegmentationLoss (0.5 Dice + 0.5 Focal/BCE) và Box Jittering ±10px.

Chạy:
    python -m unittest tests/test_medsam_lora.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.train_medsam_lora import run_epoch  # noqa: E402
from models.medsam_lora import LoRAQKV, MedSamLoRA, extract_peak_points, jitter_boxes  # noqa: E402
from utils.losses import BinaryFocalLoss, DiceLoss, SegmentationLoss  # noqa: E402


# ------------------------------------------------------------------ Mô hình SAM thu gọn cho kiểm thử nhanh (12 khối Attention)
class _DummyAttn(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v)


class _DummyBlock(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.attn = _DummyAttn(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.attn(x))


class _DummyImageEncoder(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 12, out_ch: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList([_DummyBlock(dim) for _ in range(depth)])
        self.neck = nn.Conv2d(dim, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x).permute(0, 2, 3, 1)  # (B, H/4, W/4, dim)
        for blk in self.blocks:
            h = blk(h)
        return self.neck(h.permute(0, 3, 1, 2))


class _DummyPromptEncoder(nn.Module):
    def __init__(self, out_ch: int = 16, grid: int = 8):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, out_ch, grid, grid))
        self.box_proj = nn.Linear(4, out_ch)
        self.pt_proj = nn.Linear(2, out_ch)
        self.seen_points_count = 0

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe

    def forward(self, points=None, boxes=None, masks=None):
        bs = boxes.shape[0]
        sparse = self.box_proj(boxes)  # (B, 1, out_ch)
        if points is not None:
            coords, _labels = points
            self.seen_points_count += int(coords.shape[1])
            pt_emb = self.pt_proj(coords.float())
            sparse = torch.cat([pt_emb, sparse], dim=1)
        dense = torch.zeros(bs, self.pe.shape[1], self.pe.shape[2], self.pe.shape[3], device=boxes.device)
        return sparse, dense


class _DummyMaskDecoder(nn.Module):
    """Mô phỏng hành vi `MaskDecoder` của segment-anything: chỉ nhận batch=1 mỗi lần gọi."""

    def __init__(self, in_ch: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1),
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=False):
        if image_embeddings.shape[0] != 1:
            raise RuntimeError("MaskDecoder của segment-anything yêu cầu batch=1 mỗi lần gọi")
        feat = image_embeddings + dense_prompt_embeddings + sparse_prompt_embeddings.mean(dim=1)[:, :, None, None]
        return self.conv(feat), torch.ones(1, 1, device=feat.device)


class _DummySam(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 12, out_ch: int = 16, grid: int = 8):
        super().__init__()
        self.image_encoder = _DummyImageEncoder(dim=dim, depth=depth, out_ch=out_ch)
        self.prompt_encoder = _DummyPromptEncoder(out_ch=out_ch, grid=grid)
        self.mask_decoder = _DummyMaskDecoder(in_ch=out_ch)


class TestMedSamLoRA(unittest.TestCase):
    def test_freeze_policy_and_lora_injection_across_12_blocks(self):
        sam = _DummySam(dim=32, depth=12)
        model = MedSamLoRA(sam, lora_r=8, lora_alpha=16.0, lora_dropout=0.05)

        self.assertEqual(model.n_lora_blocks, 12)
        for blk in model.sam.image_encoder.blocks:
            self.assertIsInstance(blk.attn.qkv, LoRAQKV)
            self.assertEqual(blk.attn.qkv.r, 8)

        # Kiểm tra chính sách đóng băng & mở khóa tham số
        for name, p in model.sam.image_encoder.named_parameters():
            if "lora_" in name:
                self.assertTrue(p.requires_grad, f"{name} phải requires_grad=True")
            else:
                self.assertFalse(p.requires_grad, f"{name} phải bị đóng băng")

        for name, p in model.sam.prompt_encoder.named_parameters():
            self.assertFalse(p.requires_grad, f"Prompt encoder {name} phải bị đóng băng")

        for name, p in model.sam.mask_decoder.named_parameters():
            self.assertTrue(p.requires_grad, f"Mask decoder {name} phải huấn luyện đầy đủ")

        counts = model.param_counts()
        # Mỗi khối dim=32, r=8 có A_q(32*8) + B_q(8*32) + A_v(32*8) + B_v(8*32) = 1024; 12 khối = 12288
        self.assertEqual(counts["lora_params"], 12 * 4 * 32 * 8)
        self.assertEqual(counts["trainable"], counts["lora_params"] + counts["mask_decoder_params"])

    def test_lora_zero_init_identity(self):
        torch.manual_seed(0)
        qkv = nn.Linear(32, 96)
        x = torch.randn(2, 10, 32)
        ref = qkv(x)
        lora_qkv = LoRAQKV(qkv, r=8, alpha=16.0, dropout=0.0)
        out = lora_qkv(x)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_batched_forward_backward_and_only_lora_plus_decoder_update(self):
        torch.manual_seed(42)
        sam = _DummySam(dim=32, depth=12, out_ch=16, grid=8)
        model = MedSamLoRA(sam, lora_r=8, lora_alpha=16.0)
        crit = SegmentationLoss(mode="focal", dice_weight=0.5, pixel_weight=0.5)
        opt = torch.optim.AdamW(model.param_groups(lr_lora=1e-2, lr_decoder=1e-2))

        frozen_before = {
            k: v.detach().clone()
            for k, v in model.named_parameters()
            if not v.requires_grad
        }
        trainable_before = {
            k: v.detach().clone()
            for k, v in model.named_parameters()
            if v.requires_grad
        }

        images = torch.rand(3, 3, 32, 32)
        boxes = torch.tensor([[2.0, 2.0, 20.0, 20.0], [4.0, 6.0, 28.0, 26.0], [1.0, 1.0, 15.0, 15.0]])
        masks = torch.zeros(3, 1, 32, 32)
        masks[:, :, 6:18, 6:18] = 1.0

        logits = model(images, boxes, upsample=True)
        self.assertEqual(tuple(logits.shape), (3, 1, 32, 32))

        loss_dict = crit(logits, masks)
        opt.zero_grad()
        loss_dict["loss"].backward()
        opt.step()

        # Trọng số đóng băng KHÔNG đổi 1 bit nào
        for k, v0 in frozen_before.items():
            self.assertTrue(torch.equal(dict(model.named_parameters())[k].detach(), v0), f"Tham số đóng băng {k} bị đổi!")

        # Ít nhất 1 tham số LoRA và 1 tham số Mask Decoder đã được cập nhật
        lora_changed = any(
            not torch.equal(dict(model.named_parameters())[k].detach(), v0)
            for k, v0 in trainable_before.items()
            if "lora_" in k
        )
        dec_changed = any(
            not torch.equal(dict(model.named_parameters())[k].detach(), v0)
            for k, v0 in trainable_before.items()
            if "mask_decoder" in k
        )
        self.assertTrue(lora_changed, "Tham số LoRA phải được cập nhật qua backprop")
        self.assertTrue(dec_changed, "Tham số Mask Decoder phải được cập nhật qua backprop")

    def test_checkpoint_save_and_load_roundtrip(self):
        torch.manual_seed(7)
        sam1 = _DummySam(dim=32, depth=12, out_ch=16, grid=8)
        m1 = MedSamLoRA(sam1, lora_r=8, lora_alpha=16.0)
        for p in m1.lora_parameters():
            nn.init.normal_(p, std=0.05)

        x = torch.rand(2, 3, 32, 32)
        boxes = torch.tensor([[2.0, 3.0, 22.0, 24.0], [5.0, 5.0, 25.0, 25.0]])
        m1.eval()
        with torch.no_grad():
            ref = m1(x, boxes)

        with tempfile.TemporaryDirectory() as tmp:
            ck_path = Path(tmp) / "medsam_lora.pt"
            m1.save_checkpoint(ck_path, extra={"epoch": 3})
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            # Chỉ lưu LoRA và Mask Decoder, không lưu trọng số Image Encoder gốc
            for k in ck["state_dict"]:
                self.assertTrue("lora_" in k or "mask_decoder" in k, f"Khóa thừa trong checkpoint: {k}")

            sam2 = _DummySam(dim=32, depth=12, out_ch=16, grid=8)
            sam2.load_state_dict(
                {k: v for k, v in sam1.state_dict().items() if "lora_" not in k},
                strict=False,
            )
            # Đồng bộ trọng số gốc của qkv (vì sam1 đã bọc LoRAQKV nên có khóa .qkv.qkv.*)
            for b1, b2 in zip(sam1.image_encoder.blocks, sam2.image_encoder.blocks):
                b2.attn.qkv.load_state_dict(b1.attn.qkv.qkv.state_dict())

            m2 = MedSamLoRA(sam2, lora_r=8, lora_alpha=16.0)
            m2.load_checkpoint(ck_path)
            m2.eval()
            with torch.no_grad():
                out2 = m2(x, boxes)
            self.assertTrue(torch.allclose(ref, out2, atol=1e-6))


class TestSegmentationLossesAndBoxJittering(unittest.TestCase):
    def test_dice_focal_and_bce_combination(self):
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:12, 4:12] = 1.0

        # Logits trùng khớp hoàn hảo -> loss gần 0
        perfect_logits = torch.where(targets > 0.5, torch.full_like(targets, 10.0), torch.full_like(targets, -10.0))
        bad_logits = -perfect_logits

        for mode in ("focal", "bce"):
            crit = SegmentationLoss(mode=mode, dice_weight=0.5, pixel_weight=0.5)
            good = crit(perfect_logits, targets)
            bad = crit(bad_logits, targets)
            self.assertLess(float(good["loss"]), 0.02)
            self.assertGreater(float(bad["loss"]), 0.5)
            expected = 0.5 * good["dice_loss"] + 0.5 * good["pixel_loss"]
            self.assertAlmostEqual(float(good["loss"]), float(expected), places=6)

        # Kiểm tra DiceLoss & BinaryFocalLoss độc lập
        d_loss = DiceLoss()(perfect_logits, targets)
        f_loss = BinaryFocalLoss(gamma=2.0, alpha=0.25)(perfect_logits, targets)
        self.assertLess(float(d_loss), 1e-3)
        self.assertLess(float(f_loss), 1e-4)

    def test_box_jittering_bounds_and_non_degenerate(self):
        # Gồm cả hộp bình thường, hộp sát góc (0, 0), và hộp rất nhỏ sát biên 1024
        boxes = torch.tensor(
            [
                [100.0, 200.0, 400.0, 500.0],
                [0.0, 0.0, 5.0, 5.0],
                [1020.0, 1020.0, 1024.0, 1024.0],
            ]
        )
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        j1 = jitter_boxes(boxes, max_px=10.0, img_size=1024, min_size=2.0, generator=g1)
        j2 = jitter_boxes(boxes, max_px=10.0, img_size=1024, min_size=2.0, generator=g2)

        self.assertTrue(torch.equal(j1, j2))
        self.assertTrue((j1 >= 0.0).all() and (j1 <= 1024.0).all())
        self.assertTrue(((j1[:, 2] - j1[:, 0]) >= 2.0 - 1e-5).all())
        self.assertTrue(((j1[:, 3] - j1[:, 1]) >= 2.0 - 1e-5).all())

        # Với hộp ở giữa ảnh, độ lệch từng tọa độ không vượt quá ±10px
        diff_mid = (j1[0] - boxes[0]).abs()
        self.assertTrue((diff_mid <= 10.0 + 1e-5).all())
        self.assertGreater(float(diff_mid.sum()), 0.0)

        # max_px=0 trả nguyên hộp
        self.assertTrue(torch.equal(jitter_boxes(boxes, max_px=0.0), boxes))


class TestHybridPromptAndPerClassEval(unittest.TestCase):
    def test_extract_peak_points_bright_dark_and_constant(self):
        # 1. Ảnh có khối u tăng sáng (bright) ở vùng phải-trên bên trong box
        img_bright = torch.full((1, 3, 64, 64), 0.25)
        img_bright[:, :, 18:28, 36:46] = 0.95
        box_b = torch.tensor([[12.0, 12.0, 52.0, 52.0]])

        # 2. Ảnh có khối u giảm sáng (hypointense) ở vùng trái-dưới bên trong box
        img_dark = torch.full((1, 3, 64, 64), 0.70)
        img_dark[:, :, 34:46, 18:30] = 0.05

        # 3. Ảnh đồng nhất (không nhiễu): trả về tâm hộp
        img_const = torch.zeros((1, 3, 64, 64))

        imgs = torch.cat([img_bright, img_dark, img_const], dim=0)
        boxes = box_b.repeat(3, 1)

        coords, labels = extract_peak_points(imgs, boxes, inner_margin=0.10)
        self.assertEqual(tuple(coords.shape), (3, 1, 2))
        self.assertEqual(tuple(labels.shape), (3, 1))
        self.assertTrue((labels == 1).all())
        self.assertFalse(torch.isnan(coords).any())

        # Nằm gọn trong hộp
        for i in range(3):
            self.assertGreaterEqual(float(coords[i, 0, 0]), 12.0)
            self.assertLessEqual(float(coords[i, 0, 0]), 52.0)
            self.assertGreaterEqual(float(coords[i, 0, 1]), 12.0)
            self.assertLessEqual(float(coords[i, 0, 1]), 52.0)

        # Kiểm tra đúng vị trí lõi u sáng & lõi u tối
        self.assertGreater(float(coords[0, 0, 0]), 32.0)
        self.assertLess(float(coords[0, 0, 1]), 32.0)
        self.assertLess(float(coords[1, 0, 0]), 32.0)
        self.assertGreater(float(coords[1, 0, 1]), 32.0)

    def test_hybrid_prompt_forward_and_per_class_eval_breakdown(self):
        from torch.utils.data import DataLoader, Dataset

        class _MockSegDataset(Dataset):
            def __len__(self):
                return 6

            def __getitem__(self, idx):
                sub = [1, 1, 2, 2, 3, 3][idx]  # glioma, meningioma, pituitary
                img = torch.rand(3, 32, 32)
                msk = torch.zeros(32, 32, dtype=torch.long)
                msk[8:24, 8:24] = 1
                return {
                    "image_1024": img,
                    "gt_box": torch.tensor([6.0, 6.0, 26.0, 26.0]),
                    "gt_mask": msk,
                    "has_box": torch.tensor(True),
                    "has_mask": torch.tensor(True),
                    "sub_class_label": torch.tensor(sub, dtype=torch.long),
                }

        loader = DataLoader(_MockSegDataset(), batch_size=2, shuffle=False)
        sam = _DummySam(dim=32, depth=12, out_ch=16, grid=8)
        model = MedSamLoRA(sam, lora_r=8, lora_alpha=16.0)
        crit = SegmentationLoss(mode="focal", dice_weight=0.5, pixel_weight=0.5)

        # 1. Box-only eval
        res_box = run_epoch(
            model, loader, crit, "cpu", prompt_mode="box", compute_hd95=True, compute_per_class=True, desc="test-box"
        )
        self.assertEqual(sam.prompt_encoder.seen_points_count, 0)
        self.assertEqual(res_box["prompt_mode"], "box")
        self.assertEqual(set(res_box["per_class"].keys()), {"glioma", "meningioma", "pituitary"})
        self.assertEqual(res_box["per_class"]["glioma"]["n"], 2)

        # 2. Hybrid (Box + Peak Point) eval & train
        res_hyb = run_epoch(
            model, loader, crit, "cpu", prompt_mode="hybrid", compute_hd95=True, compute_per_class=True, desc="test-hybrid"
        )
        self.assertEqual(sam.prompt_encoder.seen_points_count, 6)
        self.assertEqual(res_hyb["prompt_mode"], "hybrid")
        self.assertEqual(set(res_hyb["per_class"].keys()), {"glioma", "meningioma", "pituitary"})

        # Hybrid training step updates LoRA + Decoder without error
        opt = torch.optim.AdamW(model.param_groups(lr_lora=1e-3, lr_decoder=1e-3))
        res_tr = run_epoch(
            model, loader, crit, "cpu", optimizer=opt, box_jitter_px=4.0, prompt_mode="hybrid", desc="train-hybrid"
        )
        self.assertEqual(res_tr["n"], 6)
        self.assertGreater(res_tr["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
