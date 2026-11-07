# Mô-đun `data/` — Tiền xử lý 4 Bộ Dữ liệu MRI & `UnifiedBrainMRIDataset`

Thư mục `data/` tập hợp toàn bộ mã nguồn khảo sát thống kê (EDA), tiền xử lý chuyên biệt (Extreme-Point Skull Cropping, CLAHE, Z-score 3D, Axial Slicing), trích xuất hộp giới hạn chuẩn `gt_box = [x_min, y_min, x_max, y_max]`, chia tập chống rò rỉ bệnh nhân và bộ nạp dữ liệu đồng nhất `UnifiedBrainMRIDataset` cho cả 4 bộ dữ liệu của đề tài **ViTSAM + LoRA**.

---

## 1. Danh sách các tệp mã nguồn trong `data/`

| Nhóm | Tệp mã nguồn | Chức năng chính |
| :--- | :--- | :--- |
| **Core DataLoader & Augmentation** *(Huỳnh Nguyễn Nhân Quyết)* | `unified_dataset.py` | Lớp `UnifiedBrainMRIDataset(torch.utils.data.Dataset)` đọc đồng nhất cả 4 bộ dữ liệu, xuất `image_224` (cho ViT), `image_1024`, `gt_mask` (`1024×1024`), `gt_box` và `gt_box_norm` (cho MedSAM & Box Head). |
| | `transforms.py` | Cấu hình pipeline tăng cường dữ liệu Albumentations (`HorizontalFlip`, `VerticalFlip`, `Rotate`, `RandomBrightnessContrast`) với `BboxParams(format="pascal_voc")` bảo toàn đồng bộ tọa độ Bounding Box và Mask. |
| | `skull_crop.py` | Các hàm xử lý ảnh và mặt nạ dùng chung (`skull_crop_box`, `crop_with_mask`, `apply_clahe`, `binarize_mask(thresh=127)`, `mask_to_box`). |
| **Dataset 1 & 2 (`navoneel`, `indk214`)** *(Thái Văn Huy)* | `ds12_index.py` | Lập chỉ mục chuẩn hóa toàn bộ mẫu của Dataset 1 (`253` ảnh) và Dataset 2 (`2.192` cặp Segmentation + `7.023` ảnh Classification). |
| | `eda_ds1_ds2.py` | Xuất thống kê và biểu đồ EDA cho DS1 & DS2 vào `reports/eda_ds1_ds2/`. |
| | `build_ds1_ds2.py` | Thực thi cắt sát hộp sọ + CLAHE trên toàn bộ DS1 & DS2, chia 5-Fold CV cho DS1 và Stratified 70/15/15 cho DS2, xuất `dataset/ds1_annotations.json` và `dataset/ds2_annotations.json`. |
| | `check_ds1_ds2.py` | Kiểm tra tự động kích thước ảnh – mask, tính nhị phân `{0, 255}`, đối chiếu `gt_box` và xuất `reports/build_ds1_ds2/check_samples.png`. |
| **Dataset 3 (`brisc2025`)** *(Nguyễn Thành Hải)* | `ds3_index.py` | Phân tích tên tệp chuẩn `brisc2025_<split>_<index>_<tumor>_<plane>_<seq>` và đối sánh 1-1 hai chiều giữa `images/` và `masks/`. |
| | `eda_ds3.py` | Kiểm định chất lượng 6.000 lát cắt T1 MRI và 4.793 mask, phát hiện 46 nhóm trùng lặp `SHA256`, xuất báo cáo và biểu đồ vào `reports/eda_ds3/`. |
| | `build_ds3.py` | Kiểm định 1-1, lọc CLAHE, nhị phân hóa mask `{0, 255}`, trích xuất `gt_box`, gom 3.646 cụm `patient_id` bằng Union-Find và chia tập Stratified 70/15/15 ra `dataset/ds3_annotations.json`. |
| | `check_ds3.py` | Xác nhận tự động 6.000 bản ghi: khớp kích thước 1-1, mask `{0, 255}`, `gt_box` chính xác, 0% rò rỉ `patient_id` & `SHA256` giữa `train/val/test`, xuất `reports/build_ds3/check_samples.png`. |
| **Dataset 4 (`brats2020`)** *(Lâm Gia Bảo)* | `eda_brats.py` | Khảo sát thống kê 369 ca 3D MRI BraTS 2020 (`.nii/.nii.gz` và `.h5`), xuất thống kê và biểu đồ vào `reports/eda_ds4/`. |
| | `build_dataset.py` | Chuẩn hóa Z-score trên vùng não ($\text{voxel} > 0$), cắt lát Axial 2D, lọc lát rỗng (`< 500 px`), giữ lát u $\ge 50\text{ px}$, ghép 3 kênh `[FLAIR, T1ce, T2]`, tạo mặt nạ Whole Tumor $\text{WT} = (\text{seg} > 0)$ kèm `gt_box` và xuất `dataset/brats2020_annotations.json`. |
| | `check_dataset.py` | Kiểm định tự động không rò rỉ `case_id` giữa `train/val/test`, mọi lát có u thỏa `tumor_area >= 50 px`, khớp `bbox` và Z-score. |

---

## 2. Lệnh thực thi nhanh

```bash
# 1. Dataset 1 (navoneel) & Dataset 2 (indk214)
python -m data.eda_ds1_ds2
python -m data.build_ds1_ds2
python -m data.check_ds1_ds2

# 2. Dataset 3 (BRISC 2025)
python -m data.eda_ds3
python -m data.build_ds3
python -m data.check_ds3

# 3. Dataset 4 (BraTS 2020)
python data/eda_brats.py --data_dir dataset/brats2020 --out_dir reports/eda_ds4
python data/build_dataset.py --data_dir dataset/brats2020 --out_dir dataset/brats2020_png --out_json dataset/brats2020_annotations.json --min_tumor_area 50 --neg_ratio 1.0
python data/check_dataset.py --out_dir dataset/brats2020_png --min_tumor_area 50

# 4. Kiểm thử UnifiedBrainMRIDataset & Albumentations
python -m unittest discover tests
```
