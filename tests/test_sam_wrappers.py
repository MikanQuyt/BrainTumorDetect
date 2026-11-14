"""Kiểm thử bộ bọc SAM/MedSAM bằng mô hình giả (không cần trọng số). Chạy: python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sam_wrappers import MedSamBoxSegmenter, load_segmenter  # noqa: E402
from utils import checkpoints  # noqa: E402


class FakeMedSam:
    """Giả lập giao diện MedSAM: prompt_encoder, mask_decoder, image_encoder."""

    def __init__(self):
        self.seen_box = None
        self.prompt_encoder = self
        self.mask_decoder = self.decode
        self.image_encoder = lambda x: torch.zeros(1, 256, 64, 64)

    def __call__(self, points=None, boxes=None, masks=None):
        self.seen_box = boxes.clone()
        return torch.zeros(1, 2, 256), torch.zeros(1, 256, 64, 64)

    def get_dense_pe(self):
        return torch.zeros(1, 256, 64, 64)

    def decode(self, **kw):
        logits = torch.full((1, 1, 256, 256), -10.0)
        logits[..., 64:192, 64:192] = 10.0               # vùng dương ở giữa
        return logits, None


class MedSamWrapperTest(unittest.TestCase):
    def test_preprocess_range_and_shape(self):
        img = np.random.RandomState(0).randint(0, 255, (50, 70, 3), np.uint8)
        t = MedSamBoxSegmenter.preprocess(img)
        self.assertEqual(tuple(t.shape), (1, 3, 1024, 1024))
        self.assertAlmostEqual(float(t.min()), 0.0, places=5)
        self.assertAlmostEqual(float(t.max()), 1.0, places=5)

    def test_preprocess_constant_image_has_no_nan(self):
        t = MedSamBoxSegmenter.preprocess(np.full((20, 20, 3), 7, np.uint8))
        self.assertFalse(torch.isnan(t).any())

    def test_box_scaling_to_1024(self):
        b = MedSamBoxSegmenter.box_to_1024([10, 20, 50, 40], h=100, w=200)
        np.testing.assert_allclose(b, [[10 / 200 * 1024, 20 / 100 * 1024, 50 / 200 * 1024, 40 / 100 * 1024]], rtol=1e-5)

    def test_box_is_clipped_to_image(self):
        b = MedSamBoxSegmenter.box_to_1024([-10, -5, 500, 500], h=100, w=200)
        np.testing.assert_allclose(b, [[0, 0, 1024, 1024]], atol=1e-4)

    def test_segment_output_shape_and_threshold(self):
        fake = FakeMedSam()
        seg = MedSamBoxSegmenter(fake)
        mask = seg.segment(np.zeros((120, 80, 3), np.uint8), [10, 10, 60, 100])
        self.assertEqual(mask.shape, (120, 80))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[60, 40])                    # tâm: dương
        self.assertFalse(mask[2, 2])                     # góc: âm
        self.assertEqual(tuple(fake.seen_box.shape), (1, 1, 4))

    def test_unknown_model_name(self):
        with self.assertRaises(ValueError):
            load_segmenter("lora-sam")

    def test_missing_checkpoint_message(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError) as cm:
            load_segmenter("sam", ckpt_dir=d)
        self.assertIn("utils.checkpoints", str(cm.exception))


class CheckpointUtilsTest(unittest.TestCase):
    def test_md5_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / checkpoints.CHECKPOINTS["sam"]["file"]
            p.write_bytes(b"abc")
            self.assertEqual(checkpoints.md5sum(p), "900150983cd24fb0d6963f7d28e17f72")
            self.assertFalse(checkpoints.verify_checkpoint("sam", d))      # sai MD5 chuẩn
            with mock.patch.dict(checkpoints.CHECKPOINTS["sam"], {"md5": "900150983cd24fb0d6963f7d28e17f72"}):
                self.assertTrue(checkpoints.verify_checkpoint("sam", d))
            self.assertFalse(checkpoints.verify_checkpoint("medsam", d))   # file không tồn tại

    def test_registered_checksums_match_week2_runs(self):
        self.assertEqual(checkpoints.CHECKPOINTS["sam"]["md5"], "01ec64d29a2fca3f0661936605ae66f8")
        self.assertEqual(checkpoints.CHECKPOINTS["medsam"]["md5"], "3bb6db55bd0c9ca30b61248bca72f8d6")


if __name__ == "__main__":
    unittest.main()
