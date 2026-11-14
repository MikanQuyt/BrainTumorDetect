"""Unit tests kiểm định toàn diện kiến trúc ViT-LoRA, Hàm mất mát và Bộ Tự sinh Lời nhắc (Tuần 3)."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from transformers import ViTConfig, ViTModel

from experiments.run_lora_rank_ablation import summarize_rank_parameters
from models.auto_prompt import (
    AutoPromptGenerator,
    compute_attention_rollout,
    compute_layercam,
    extract_brain_parenchyma_mask,
    fuse_rollout_and_layercam,
    heatmap_to_box_and_points,
)
from models.vit_lora import ViTDetBoxHead, ViTLoRA, pad_normalized_boxes
from utils.losses import (
    BoxRegressionLoss,
    FocalLossWithLabelSmoothing,
    GIoULoss,
    WeightedLabelSmoothingCrossEntropy,
    box_iou_tensor,
    build_classification_loss,
    compute_class_weights,
    generalized_box_iou_tensor,
)
from utils.metrics import (
    BoxLocalizationMeter,
    box_iou_score,
    optimize_detection_threshold,
    pointing_game_hit,
)


def _make_tiny_vit_12layers(hidden_size: int = 64, num_heads: int = 4) -> ViTModel:
    cfg = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 2,
        image_size=224,
        patch_size=16,
    )
    cfg._attn_implementation = "eager"
    return ViTModel(cfg, add_pooling_layer=False)


class TestViTLoRAArchitecture(unittest.TestCase):
    def test_lora_injected_into_12_attention_layers_query_and_value(self):
        backbone = _make_tiny_vit_12layers(hidden_size=64, num_heads=4)
        model = ViTLoRA(
            num_classes=4,
            lora_r=8,
            lora_alpha=16,
            target_modules=("query", "value"),
            use_box_head=True,
            sfp_dim=32,
            pretrained=False,
            backbone=backbone,
        )
        lora_mods = model.lora_module_names()
        # 12 tầng * 2 module (query/q_proj, value/v_proj) = 24 module gắn LoRA
        self.assertEqual(len(lora_mods), 24)
        self.assertTrue(all(name.endswith((".query", ".value", ".q_proj", ".v_proj")) for name in lora_mods))

        counts = model.param_counts()
        self.assertEqual(counts["num_lora_modules"], 24)
        self.assertGreater(counts["lora_params"], 0)
        self.assertGreater(counts["box_head_params"], 0)

        # Kiểm tra các trọng số gốc của ViT đã đóng băng, còn lora_A / lora_B mở khóa
        for name, param in model.backbone.named_parameters():
            if "lora_" in name:
                self.assertTrue(param.requires_grad, f"{name} phải có requires_grad=True")
            else:
                self.assertFalse(param.requires_grad, f"{name} phải đóng băng (requires_grad=False)")

    def test_forward_and_box_head_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        backbone = _make_tiny_vit_12layers(hidden_size=64, num_heads=4)
        model = ViTLoRA(
            num_classes=4,
            lora_r=4,
            lora_alpha=8,
            use_box_head=True,
            sfp_dim=32,
            pretrained=False,
            backbone=backbone,
        )
        x = torch.randn(2, 3, 224, 224)
        logits, pred_boxes = model(x, return_box=True)
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(pred_boxes.shape, (2, 4))
        self.assertTrue(((pred_boxes >= 0.0) & (pred_boxes <= 1.0)).all())
        self.assertTrue((pred_boxes[:, 2] >= pred_boxes[:, 0]).all())
        self.assertTrue((pred_boxes[:, 3] >= pred_boxes[:, 1]).all())

        out_dict = model(x, return_dict=True, output_attentions=True)
        self.assertEqual(len(out_dict["attentions"]), 12)
        self.assertEqual(out_dict["attentions"][0].shape, (2, 4, 197, 197))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "lora.pt"
            model.save_lora_checkpoint(ckpt_path, extra={"epoch": 3})

            backbone2 = _make_tiny_vit_12layers(hidden_size=64, num_heads=4)
            # Copy base weights so only LoRA/head load needs to match
            backbone2.load_state_dict(backbone.base_model.state_dict() if hasattr(backbone, "base_model") else backbone.state_dict(), strict=False)
            model2 = ViTLoRA(
                num_classes=4,
                lora_r=4,
                lora_alpha=8,
                use_box_head=True,
                sfp_dim=32,
                pretrained=False,
                backbone=backbone2,
            )
            # Copy frozen base weights then overwrite trainable state_dict
            model2.load_state_dict(model.state_dict(), strict=True)
            for p in model2.head.parameters():
                p.data.zero_()
            model2.load_lora_checkpoint(ckpt_path)
            model.eval()
            model2.eval()
            with torch.no_grad():
                l1, b1 = model(x, return_box=True)
                l2, b2 = model2(x, return_box=True)
            self.assertTrue(torch.allclose(l1, l2, atol=1e-6))
            self.assertTrue(torch.allclose(b1, b2, atol=1e-6))

    def test_ablation_rank_parameter_summary(self):
        rows = summarize_rank_parameters(ranks=(4, 8, 16, 32), num_classes=4, use_box_head=False)
        by_cfg = {r["config"]: r for r in rows}
        self.assertLess(by_cfg["vit_lora_r8"]["lora_pct_of_backbone"], 1.0)
        self.assertLess(by_cfg["vit_lora_r4"]["lora_params"], by_cfg["vit_lora_r8"]["lora_params"])
        self.assertLess(by_cfg["vit_lora_r8"]["lora_params"], by_cfg["vit_lora_r16"]["lora_params"])
        self.assertLess(by_cfg["vit_lora_r16"]["lora_params"], by_cfg["vit_lora_r32"]["lora_params"])


class TestLossesAndBoxRegression(unittest.TestCase):
    def test_weighted_ce_and_focal_loss_with_label_smoothing(self):
        labels = [0, 0, 0, 1, 2, 3]
        w = compute_class_weights(labels, num_classes=4)
        self.assertEqual(w.shape, (4,))
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        self.assertLess(float(w[0]), float(w[1]))

        logits = torch.randn(6, 4, requires_grad=True)
        targets = torch.tensor(labels, dtype=torch.long)

        ce_fn = build_classification_loss("weighted_ce", class_weights=w, label_smoothing=0.05)
        loss_ce = ce_fn(logits, targets)
        self.assertTrue(torch.isfinite(loss_ce))
        loss_ce.backward()
        self.assertIsNotNone(logits.grad)

        logits2 = torch.randn(6, 4, requires_grad=True)
        focal_fn = build_classification_loss("focal", class_weights=w, gamma=2.0, label_smoothing=0.05)
        loss_focal = focal_fn(logits2, targets)
        self.assertTrue(torch.isfinite(loss_focal))
        loss_focal.backward()
        self.assertIsNotNone(logits2.grad)

    def test_giou_and_box_regression_loss_with_masking(self):
        b1 = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]], dtype=torch.float32, requires_grad=True)
        b2 = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.4, 0.4, 0.9, 0.9]], dtype=torch.float32)

        ious = box_iou_tensor(b1, b2)
        gious = generalized_box_iou_tensor(b1, b2)
        self.assertAlmostEqual(float(ious[0].detach()), 1.0, places=5)
        self.assertAlmostEqual(float(gious[0].detach()), 1.0, places=5)
        self.assertLess(float(gious[1].detach()), float(ious[1].detach()) + 1e-5)

        criterion = BoxRegressionLoss(giou_weight=1.0, l1_weight=1.0)
        has_box = torch.tensor([True, False])
        res = criterion(b1, b2, has_box=has_box)
        self.assertAlmostEqual(float(res["loss"].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(res["mean_iou"]), 1.0, places=5)

        # Trường hợp cả batch không có box (ví dụ toàn ảnh No Tumor hoặc Dataset 1)
        res_empty = criterion(b1, b2, has_box=torch.tensor([False, False]))
        self.assertEqual(float(res_empty["loss"].detach()), 0.0)
        self.assertEqual(int(res_empty["n_boxes"]), 0)

    def test_box_head_optimization_converges_on_synthetic_target(self):
        torch.manual_seed(0)
        head = ViTDetBoxHead(hidden_size=32, sfp_dim=16, dropout=0.0)
        patch_tokens = torch.randn(4, 196, 32)
        cls_token = torch.randn(4, 32)
        target_box = torch.tensor([[0.2, 0.25, 0.65, 0.75]] * 4, dtype=torch.float32)

        criterion = BoxRegressionLoss(giou_weight=1.0, l1_weight=1.0)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2)

        init_res = criterion(head(patch_tokens, cls_token), target_box)
        for _ in range(25):
            opt.zero_grad()
            pred = head(patch_tokens, cls_token)
            out = criterion(pred, target_box)
            out["loss"].backward()
            opt.step()
        final_res = criterion(head(patch_tokens, cls_token), target_box)
        self.assertLess(float(final_res["loss"].detach()), float(init_res["loss"].detach()))
        self.assertGreater(float(final_res["mean_iou"]), 0.75)


class TestAutoPromptGenerator(unittest.TestCase):
    def test_attention_rollout_and_layercam_and_auto_prompt_modes(self):
        torch.manual_seed(7)
        backbone = _make_tiny_vit_12layers(hidden_size=64, num_heads=4)
        model = ViTLoRA(
            num_classes=4,
            lora_r=4,
            lora_alpha=8,
            use_box_head=True,
            sfp_dim=32,
            pretrained=False,
            backbone=backbone,
        )
        x = torch.randn(2, 3, 224, 224)

        cam_out = compute_layercam(model, x)
        self.assertEqual(cam_out["layercam_grid"].shape, (2, 14, 14))
        self.assertEqual(len(cam_out["attentions"]), 12)

        rollout = compute_attention_rollout(cam_out["attentions"])
        self.assertEqual(rollout.shape, (2, 14, 14))
        self.assertTrue(((rollout >= 0.0) & (rollout <= 1.0)).all())

        pmask = extract_brain_parenchyma_mask(x, out_size=(256, 256))
        self.assertEqual(pmask.shape, (2, 256, 256))

        fused = fuse_rollout_and_layercam(rollout, cam_out["layercam_grid"], parenchyma_mask=pmask, out_size=(256, 256))
        self.assertEqual(fused["fused_heatmap"].shape, (2, 256, 256))

        gen = AutoPromptGenerator(model, mode="auto", threshold=0.35, box_padding=0.08, out_size=(1024, 1024))
        for mode in ("weakly_supervised", "box_head", "hybrid_box", "auto"):
            prompts = gen.generate(x, mode=mode)
            self.assertEqual(prompts["boxes_1024"].shape, (2, 4))
            self.assertEqual(prompts["point_coords_1024"].shape, (2, 2, 2))
            self.assertEqual(prompts["point_labels"].shape, (2, 2))
            self.assertEqual(prompts["fused_heatmap"].shape, (2, 1024, 1024))
            # Điểm dương (index 0) luôn nằm trong hộp giới hạn cuối cùng
            bx = prompts["boxes_1024"]
            px = prompts["point_coords_1024"][:, 0, 0]
            py = prompts["point_coords_1024"][:, 0, 1]
            self.assertTrue(((px >= bx[:, 0] - 1e-4) & (px <= bx[:, 2] + 1e-4)).all())
            self.assertTrue(((py >= bx[:, 1] - 1e-4) & (py <= bx[:, 3] + 1e-4)).all())

    def test_heatmap_to_box_and_pointing_game_accuracy(self):
        # Tạo bản đồ nhiệt giả lập có khối u tại vùng [300..500, 400..600] trên ảnh 1024x1024
        hm = np.zeros((1024, 1024), dtype=np.float32)
        gt_mask = np.zeros((1024, 1024), dtype=np.uint8)
        gt_mask[300:500, 400:600] = 1
        yy, xx = np.ogrid[:1024, :1024]
        hm = np.exp(-((xx - 500.0) ** 2 + (yy - 400.0) ** 2) / (2.0 * 60.0 ** 2)).astype(np.float32)

        res = heatmap_to_box_and_points(hm, threshold_ratio=0.35, box_padding=0.08)
        gt_box = np.array([400.0, 300.0, 600.0, 500.0], dtype=np.float32)

        iou = box_iou_score(res["box_px"], gt_box)
        hit = pointing_game_hit(res["peak_pos_xy"], gt_mask=gt_mask, gt_box=gt_box)
        self.assertGreater(iou, 0.65)
        self.assertTrue(hit)

        meter = BoxLocalizationMeter()
        meter.update(pred_box=res["box_px"], gt_box=gt_box, peak_point_xy=res["peak_pos_xy"], gt_mask=gt_mask)
        summary = meter.summary()
        self.assertEqual(summary["pointing_game_acc"], 1.0)
        self.assertGreater(summary["mean_box_iou"], 0.65)

    def test_optimize_detection_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # Một mẫu có u có xác suất 0.38 (bị sót ở ngưỡng 0.50 nhưng bắt được ở ngưỡng 0.35)
        y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.38, 0.75, 0.82, 0.90, 0.92, 0.95])
        opt = optimize_detection_threshold(y_true, y_prob, target_recall=0.95)
        self.assertGreaterEqual(opt["best"]["recall"], 0.95)
        self.assertLessEqual(opt["best"]["threshold"], 0.40)


if __name__ == "__main__":
    unittest.main()
