import albumentations as A
import torch
from pathlib import Path
from torch.utils.data import Dataset

from datasets.libs import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform():
    """Tăng cường đồng bộ ảnh, mask và bbox."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3,
            clip=True,
        ),
    )


def build_val_transform():
    """Không tăng cường, giữ định dạng đầu ra đồng nhất."""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
    )


class UnifiedBrainMRIDataset(Dataset):
    """Dataset thống nhất cho 4 bộ MRI não. Record JSON: img_path, mask_path, has_mask,
    has_tumor, sub_class_label, gt_box [xmin, ymin, xmax, ymax] (nếu thiếu thì suy từ mask)."""

    def __init__(self, json_annotation_files, transform=None, train=True, split=None, fold=None,
                 root=None):
        """split: lọc theo 'train'/'val'/'test'/'external_test'; fold: lọc fold CV của DS1;
        root: thư mục gốc để nối với đường dẫn tương đối trong JSON (mặc định: gốc dự án)."""
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]
        if isinstance(json_annotation_files, (str, bytes)):
            json_annotation_files = [json_annotation_files]

        self.data_records = []
        for json_file in json_annotation_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data_records.extend(json.load(f))
        if split is not None:
            self.data_records = [r for r in self.data_records if r.get('split') == split]
        if fold is not None:
            self.data_records = [r for r in self.data_records if r.get('fold') == fold]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_train_transform() if train else build_val_transform()

    def __len__(self):
        return len(self.data_records)

    @staticmethod
    def _read_image(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            raise FileNotFoundError(f'Không đọc được ảnh: {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _box_from_mask(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

    @staticmethod
    def _read_gray(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None

    def __getitem__(self, idx):
        record = self.data_records[idx]
        image = self._read_image(self.root / record['img_path'])
        h, w = image.shape[:2]

        if record.get('has_mask') and record.get('mask_path'):
            mask = self._read_gray(self.root / record['mask_path'])
            if mask is None:
                raise FileNotFoundError(f"Không đọc được mask: {record['mask_path']}")
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
            has_mask = True
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            has_mask = False

        has_tumor = int(record['has_tumor'])
        sub_class_label = int(record['sub_class_label'])

        gt_box = record.get('gt_box')
        if gt_box is None and has_mask:
            gt_box = self._box_from_mask(mask)

        bboxes = [list(gt_box)] if (gt_box is not None and has_tumor) else []
        labels = [sub_class_label] * len(bboxes)

        out = self.transform(image=image, mask=mask, bboxes=bboxes, class_labels=labels)
        image, mask = out['image'], out['mask']
        h, w = image.shape[:2]

        if out['bboxes']:
            box = np.array(out['bboxes'][0][:4], dtype=np.float32)
            has_box = True
        else:
            box = np.zeros(4, dtype=np.float32)
            has_box = False
        box_norm = box / np.array([w, h, w, h], dtype=np.float32)  # [0,1]

        # ViT: 224, chuẩn hóa ImageNet
        img224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img224 = (img224 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)

        # MedSAM: 1024, [0,1]
        img1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        return {
            'image_224': torch.from_numpy(img224).permute(2, 0, 1).contiguous(),
            'image_1024': torch.from_numpy(img1024).permute(2, 0, 1).contiguous(),
            'binary_label': torch.tensor(has_tumor, dtype=torch.long),
            'sub_class_label': torch.tensor(sub_class_label, dtype=torch.long),
            'gt_box': torch.from_numpy(box_norm * 1024.0),
            'gt_box_norm': torch.from_numpy(box_norm),
            'has_box': torch.tensor(has_box, dtype=torch.bool),
            'gt_mask': torch.from_numpy(mask1024).long(),
            'has_mask': torch.tensor(has_mask, dtype=torch.bool),
        }
