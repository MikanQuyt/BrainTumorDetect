"""Kiểm thử UnifiedBrainMRIDataset trên dữ liệu giả lập. Chạy: python -m unittest discover tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import UnifiedBrainMRIDataset, build_train_transform  # noqa: E402


def make_fixture(root):
    img = np.full((160, 200, 3), 40, np.uint8)
    mask = np.zeros((160, 200), np.uint8)
    mask[50:90, 120:170] = 255
    img[mask > 0] = 220
    cv2.imwrite(str(root / "a.png"), img)
    cv2.imwrite(str(root / "a_mask.png"), mask)
    records = [
        dict(dataset="indk214", img_path="a.png", mask_path="a_mask.png", has_mask=True, has_tumor=1,
             sub_class_label=1, gt_box=None, split="train", fold=None),
        dict(dataset="indk214", img_path="a.png", mask_path=None, has_mask=False, has_tumor=0,
             sub_class_label=0, gt_box=None, split="val", fold=None),
        dict(dataset="navoneel", img_path="a.png", mask_path=None, has_mask=False, has_tumor=1,
             sub_class_label=-1, gt_box=None, split="external_test", fold=2),
    ]
    (root / "ann.json").write_text(json.dumps(records), encoding="utf-8")


class UnifiedDatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_fixture(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def ds(self, **kw):
        return UnifiedBrainMRIDataset(self.root / "ann.json", root=self.root, **kw)

    def test_output_contract(self):
        item = self.ds(train=False, split="train")[0]
        self.assertEqual(tuple(item["image_224"].shape), (3, 224, 224))
        self.assertEqual(tuple(item["image_1024"].shape), (3, 1024, 1024))
        self.assertEqual(tuple(item["gt_mask"].shape), (1024, 1024))
        self.assertTrue(item["has_mask"] and item["has_box"])
        self.assertEqual(item["gt_box"].shape[0], 4)

    def test_has_mask_flag_per_dataset(self):
        self.assertTrue(self.ds(train=False, datasets="indk214", split="train")[0]["has_mask"])
        self.assertFalse(self.ds(train=False, datasets="navoneel")[0]["has_mask"])

    def test_filters(self):
        self.assertEqual(len(self.ds()), 3)
        self.assertEqual(len(self.ds(split="val")), 1)
        self.assertEqual(len(self.ds(fold=2)), 1)
        self.assertEqual(len(self.ds(exclude_fold=2)), 2)

    def test_eval_box_matches_mask(self):
        item = self.ds(train=False, split="train")[0]
        ys, xs = np.where(item["gt_mask"].numpy() > 0)
        got = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
        np.testing.assert_allclose(item["gt_box"].numpy(), got, atol=8.0)  # sai số do resize nearest

    def test_box_follows_mask_under_augmentation(self):
        aug = A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
                        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))
        item = self.ds(transform=aug, split="train")[0]
        ys, xs = np.where(item["gt_mask"].numpy() > 0)
        got = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
        np.testing.assert_allclose(item["gt_box"].numpy(), got, atol=8.0)

    def test_random_train_augmentation_runs(self):
        ds = self.ds(transform=build_train_transform(), split="train")
        for _ in range(20):
            item = ds[0]
            if item["has_box"]:
                b = item["gt_box_norm"]
                self.assertTrue((b >= 0).all() and (b <= 1).all() and b[2] > b[0] and b[3] > b[1])

    def test_no_tumor_has_empty_box(self):
        item = self.ds(train=False, split="val")[0]
        self.assertFalse(item["has_box"])
        self.assertEqual(float(item["gt_box"].abs().sum()), 0.0)

    def test_all_four_real_annotation_files_harmonized(self):
        ds_train = UnifiedBrainMRIDataset(datasets="all", split="train", train=False)
        ds_val = UnifiedBrainMRIDataset(datasets="all", split="val", train=False)
        ds_test = UnifiedBrainMRIDataset(datasets="all", split="test", train=False)
        for ds in (ds_train, ds_val, ds_test):
            present = {r["dataset"] for r in ds.records}
            self.assertEqual(present, {"navoneel", "indk214", "brisc2025", "brats2020"})
        # DS1 5-Fold CV check
        ds1_fold0 = UnifiedBrainMRIDataset(datasets="navoneel", fold=0, train=False)
        ds1_ex0 = UnifiedBrainMRIDataset(datasets="navoneel", exclude_fold=0, train=False)
        self.assertEqual(len(ds1_fold0) + len(ds1_ex0), 253)
        # BraTS2020 multiclass label harmonized (0 = no_tumor, 1 = glioma)
        brats_labels = {int(r["sub_class_label"]) for r in ds_train.records if r["dataset"] == "brats2020"}
        self.assertEqual(brats_labels, {0, 1})


if __name__ == "__main__":
    unittest.main()

