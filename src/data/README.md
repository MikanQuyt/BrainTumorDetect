# Pipeline dữ liệu BraTS2020 (cắt lát 2D)

## Yêu cầu
Python 3.10+; `pip install nibabel numpy pandas matplotlib tqdm`

## Chạy
1. Khảo sát: `python src/data/eda_brats.py --data_dir <BraTS2020_TrainingData> --out_dir reports`
2. Tạo dataset: `python src/data/build_dataset.py --data_dir <BraTS2020_TrainingData> --out_dir <thư mục lưu> --neg_ratio 1.5`
3. Kiểm tra: `python src/data/check_dataset.py --out_dir <thư mục lưu>`

## Các bước xử lý
Đọc FLAIR/T1ce/T2 + seg -> Z-score theo từng volume và modality (voxel khác 0, nền = 0)
-> cắt lát ngang theo trục z -> lọc lát rỗng -> mask Whole Tumor (seg > 0)
-> bounding box (một box bao toàn bộ u) -> ghép 3 kênh [FLAIR, T1ce, T2]
-> chia train/val/test theo ca -> cân bằng u : lành (chỉ tập train, trần 1.5 lát lành / 1 lát u cho mỗi ca).

## Kết quả (seed 42, 369 ca)
| Tập | Số ca | Số lát | Lát u | Lát lành | Tỷ lệ u |
|---|---|---|---|---|---|
| train | [..] | [..] | [..] | [..] | [..] |
| val | [..] | [..] | [..] | [..] | [..] |
| test | [..] | [..] | [..] | [..] | [..] |

Val và test giữ phân bố tự nhiên (không cân bằng).

## Đầu ra
- `train|val|test/<ca>_z<zzz>.npz`: `image` (240,240,3) float16, `mask` (240,240) uint8.
- `metadata.csv`: bbox dạng pixel (`x_min,y_min,x_max,y_max`) và dạng YOLO (`yolo_cx,yolo_cy,yolo_w,yolo_h`).
- `splits.csv`, `case_summary.csv`, `all_slices_areas.csv`, `config.json`.

## Tham số chính
`--seed 42`, `--val_ratio 0.15`, `--test_ratio 0.15`, `--min_brain_area 500`,
`--min_tumor_area 1`, `--neg_ratio 1.5`, `--balance_splits train`, `--dtype float16`.

## Đọc một lát
```python
import numpy as np
d = np.load("train/BraTS20_Training_001_z075.npz")
img, mask = d["image"].astype("float32"), d["mask"]
```