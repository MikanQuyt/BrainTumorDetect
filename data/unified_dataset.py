"""UnifiedBrainMRIDataset: đọc đồng nhất ảnh, nhãn phân loại và mặt nạ từ cả 4 bộ dữ liệu.

Mỗi record trong file JSON chú thích có dạng:
    {
      "dataset": "navoneel" | "indk214" | "brisc2025" | "brats2020",
      "img_path": "đường dẫn ảnh (tương đối so với root hoặc tuyệt đối)",
      "mask_path": "đường dẫn mask nhị phân" | null,
      "has_mask": true | false,            # True cho DS2, DS3, DS4; False cho DS1
      "has_tumor": 0 | 1,
      "sub_class_label": 0 No Tumor | 1 Glioma | 2 Meningioma | 3 Pituitary | -1 có u nhưng không rõ loại,
      "gt_box": [xmin, ymin, xmax, ymax] | null,   # pixel, ảnh gốc; nếu thiếu thì suy từ mask
      "split": "train" | "val" | "test" | "external_test",
      "fold": int | null                   # fold CV của DS1
    }
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import build_eval_transform, build_train_transform

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path, flag):
    # np.fromfile + imdecode để đọc được đường dẫn có ký tự Unicode trên Windows
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flag) if buf.size else None


class UnifiedBrainMRIDataset(Dataset):
    """Trả về từ điển đồng nhất cho cả nhánh ViT (224) và nhánh MedSAM (1024).

    Args:
        annotation_files: một hoặc nhiều file JSON chú thích.
        train: True dùng pipeline tăng cường; False dùng pipeline đánh giá. Bị ghi đè bởi `transform`.
        transform: Albumentations Compose tùy chỉnh (phải có bbox_params với label_fields=['class_labels']).
        datasets / split / fold / exclude_fold: lọc record theo bộ dữ liệu, tập chia, fold CV của DS1.
        root: thư mục gốc để nối với đường dẫn tương đối (mặc định: gốc dự án).
    """

    def __init__(self, annotation_files, train=True, transform=None, datasets=None,
                 split=None, fold=None, exclude_fold=None, root=None):
        if isinstance(annotation_files, (str, Path)):
            annotation_files = [annotation_files]
        self.root = Path(root) if root else PROJECT_ROOT

        records = []
        for f in annotation_files:
            path = Path(f)
            path = path if path.is_absolute() else self.root / path
            with open(path, "r", encoding="utf-8") as fh:
                records.extend(json.load(fh))

        ds_set = set([datasets] if isinstance(datasets, str) else datasets or [])
        self.records = [
            r for r in records
            if (not ds_set or r["dataset"] in ds_set)
            and (split is None or r.get("split") == split)
            and (fold is None or r.get("fold") == fold)
            and (exclude_fold is None or r.get("fold") != exclude_fold)
        ]
        self.transform = transform or (build_train_transform() if train else build_eval_transform())

    def __len__(self):
        return len(self.records)

    def _path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = _read(self._path(rec["img_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {rec['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        has_mask = bool(rec.get("has_mask")) and bool(rec.get("mask_path"))
        if has_mask:
            mask = _read(self._path(rec["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Không đọc được mask: {rec['mask_path']}")
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        has_tumor = int(rec["has_tumor"])
        sub_label = int(rec["sub_class_label"])

        box = rec.get("gt_box")
        if box is None and has_mask and mask.any():
            ys, xs = np.where(mask > 0)
            box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        bboxes = []
        if has_tumor and box is not None:
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = np.clip([x0, x1], 0, w)
            y0, y1 = np.clip([y0, y1], 0, h)
            if x1 > x0 and y1 > y0:  # box suy biến làm Albumentations báo lỗi
                bboxes = [[float(x0), float(y0), float(x1), float(y1)]]

        out = self.transform(image=image, mask=mask, bboxes=bboxes, class_labels=[sub_label] * len(bboxes))
        image, mask = out["image"], out["mask"]
        h, w = image.shape[:2]

        # Box mất sau augmentation -> box rỗng + has_box=False (không dùng lại box cũ sai vị trí)
        has_box = len(out["bboxes"]) > 0
        box_px = np.array(out["bboxes"][0][:4], dtype=np.float32) if has_box else np.zeros(4, np.float32)
        box_norm = box_px / np.array([w, h, w, h], dtype=np.float32)

        # Nhánh ViT: 224x224, chuẩn hóa ImageNet
        img224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img224 = (img224 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        # Nhánh MedSAM: 1024x1024, giá trị [0, 1]
        img1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        return {
            "image_224": torch.from_numpy(img224).permute(2, 0, 1).contiguous(),
            "image_1024": torch.from_numpy(img1024).permute(2, 0, 1).contiguous(),
            "binary_label": torch.tensor(has_tumor, dtype=torch.long),
            "sub_class_label": torch.tensor(sub_label, dtype=torch.long),
            "gt_box": torch.from_numpy(box_norm * 1024.0),   # pixel hệ 1024 (prompt MedSAM)
            "gt_box_norm": torch.from_numpy(box_norm),       # [0, 1] (nhánh hồi quy box của ViT)
            "has_box": torch.tensor(has_box, dtype=torch.bool),
            "gt_mask": torch.from_numpy(mask1024).long(),
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "dataset": rec["dataset"],
        }

