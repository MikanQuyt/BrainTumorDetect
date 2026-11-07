# BrainTumorDetect

Hệ thống phát hiện & phân vùng u não (ViT + MedSAM, LoRA) trên 4 bộ dữ liệu MRI.

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `configs/` | Cấu hình (đường dẫn dữ liệu, siêu tham số, tăng cường dữ liệu) |
| `data/` | `UnifiedBrainMRIDataset` và pipeline Albumentations |
| `models/` | Kiến trúc mô hình (ViT, SAM/MedSAM) |
| `engine/` | Vòng lặp huấn luyện và đánh giá |
| `utils/` | Hàm tiện ích, chỉ số đo lường |
| `app/` | Ứng dụng demo |
| `tests/` | Kiểm thử |

## UnifiedBrainMRIDataset

```python
from data import UnifiedBrainMRIDataset

ds = UnifiedBrainMRIDataset(
    ["dataset/unified_2d_annotations.json", "dataset/brats2020_annotations.json"],
    train=True, datasets=["indk214", "brisc2025"], split="train")
sample = ds[0]
```

Mỗi mẫu là từ điển: `image_224`, `image_1024`, `binary_label`, `sub_class_label`, `gt_box`,
`gt_box_norm`, `has_box`, `gt_mask`, `has_mask` (`True` cho DS2, DS3, DS4; `False` cho DS1) và `dataset`.
Ảnh, mask và bounding box đi qua cùng một lần biến đổi Albumentations nên luôn khớp nhau.
Schema JSON chú thích được mô tả ở đầu file `data/unified_dataset.py`.

## Kiểm thử

```bash
python -m unittest discover tests
```

