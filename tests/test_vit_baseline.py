"""Kiểm thử ViTBaseline bằng backbone cấu hình nhỏ (không cần tải checkpoint). Chạy: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vit_baseline import IMAGENET_MEAN, IMAGENET_STD, ViTBaseline  # noqa: E402


def tiny_backbone():
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                    image_size=224, patch_size=16, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    return ViTModel(cfg, add_pooling_layer=False)


def make(mode, **kw):
    torch.manual_seed(0)
    return ViTBaseline(4, mode, backbone=tiny_backbone(), **kw)


class ViTBaselineTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2, 3, 224, 224)
        self.y = torch.tensor([0, 3])

    def test_output_shape(self):
        for mode in ("linear_probe", "full_finetune"):
            self.assertEqual(tuple(make(mode)(self.x).shape), (2, 4))

    def test_linear_probe_freezes_backbone(self):
        m = make("linear_probe")
        c = m.param_counts()
        head = sum(p.numel() for p in m.head.parameters())
        self.assertEqual(c["trainable"], head)
        self.assertTrue(all(not p.requires_grad for p in m.backbone.parameters()))
        torch.nn.functional.cross_entropy(m(self.x), self.y).backward()
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertTrue(all(p.grad is not None for p in m.head.parameters()))

    def test_full_finetune_trains_everything(self):
        m = make("full_finetune")
        c = m.param_counts()
        self.assertEqual(c["trainable"], c["total"])
        torch.nn.functional.cross_entropy(m(self.x), self.y).backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_linear_probe_backbone_stays_eval_in_train_mode(self):
        m = make("linear_probe")
        m.train()
        self.assertFalse(m.backbone.training)       # dropout của backbone đóng băng phải tắt
        self.assertTrue(m.head.training)
        a, b = m(self.x), m(self.x)
        self.assertTrue(torch.allclose(a, b))         # xác định vì backbone không dropout

    def test_full_finetune_backbone_trains_in_train_mode(self):
        m = make("full_finetune")
        m.train()
        self.assertTrue(m.backbone.training)

    def test_linear_probe_weights_unchanged_after_step(self):
        m = make("linear_probe")
        before = {k: v.clone() for k, v in m.backbone.state_dict().items()}
        opt = torch.optim.AdamW(m.param_groups(1e-3, 1e-3))
        m.train()
        torch.nn.functional.cross_entropy(m(self.x), self.y).backward()
        opt.step()
        for k, v in m.backbone.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)

    def test_mode_switch(self):
        m = make("linear_probe")
        m.set_mode("full_finetune")
        self.assertEqual(m.param_counts()["trainable"], m.param_counts()["total"])
        m.set_mode("linear_probe")
        self.assertLess(m.param_counts()["trainable"], m.param_counts()["total"])
        with self.assertRaises(ValueError):
            m.set_mode("lora")

    def test_param_groups(self):
        self.assertEqual(len(make("linear_probe").param_groups(2e-5, 1e-3)), 1)
        g = make("full_finetune").param_groups(2e-5, 1e-3)
        self.assertEqual([x["lr"] for x in g], [1e-3, 2e-5])

    def test_trainable_state_dict(self):
        lp = make("linear_probe").trainable_state_dict()
        self.assertTrue(all(k.startswith("head.") for k in lp))
        self.assertGreater(len(make("full_finetune").trainable_state_dict()), len(lp))

    def test_imagenet_to_half_renorm(self):
        # Ảnh giả [0,1] -> chuẩn hóa ImageNet; mô hình phải cho kết quả bằng đường (x - 0.5) / 0.5 trực tiếp
        raw = torch.rand(2, 3, 224, 224)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        a = make("full_finetune", input_norm="imagenet").eval()
        b = make("full_finetune", input_norm="none").eval()
        with torch.no_grad():
            out_a = a((raw - mean) / std)
            out_b = b((raw - 0.5) / 0.5)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5))

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            ViTBaseline(4, "nope", backbone=tiny_backbone())
        with self.assertRaises(ValueError):
            ViTBaseline(4, "linear_probe", backbone=tiny_backbone(), input_norm="x")


if __name__ == "__main__":
    unittest.main()
