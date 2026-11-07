# Mô-đun `data/` — Tiền xử lý Dataset 1 (`navoneel`) & Dataset 2 (`indk214`)

Thư mục `data/` trên nhánh **`HNNQ_TVH_tuan1`** cung cấp bộ công cụ khảo sát thống kê (EDA), cắt sát hộp sọ tự động (**Extreme-Point Skull Cropping + CLAHE**), trích xuất tọa độ hộp chuẩn `gt_box` và nạp dữ liệu đồng nhất qua `UnifiedBrainMRIDataset`.

## Danh sách các tệp mã nguồn

| Tệp | Chức năng chính |
| :--- | :--- |
| `skull_crop.py` | Cài đặt `skull_crop_box`, `crop_with_mask`, `apply_clahe`, `binarize_mask(thresh=127)` và `mask_to_box` (`[x_min, y_min, x_max, y_max]`). |
| `ds12_index.py` | Liệt kê chuẩn hóa toàn bộ mẫu của Dataset 1 (`253` ảnh, loại bỏ thư mục trùng `brain_tumor_dataset/`) và Dataset 2 (`2.192` cặp Segmentation + `7.023` ảnh Classification). |
| `eda_ds1_ds2.py` | Xuất thống kê và biểu đồ EDA cho DS1 & DS2 vào `reports/eda_ds1_ds2/`. |
| `build_ds1_ds2.py` | Thực thi cắt sát hộp sọ + CLAHE trên toàn bộ DS1 & DS2, xuất ảnh đã cắt vào `dataset/ds1_cropped/`, `dataset/ds2_cropped/` và xuất `dataset/ds1_annotations.json`, `dataset/ds2_annotations.json`. |
| `check_ds1_ds2.py` | Kiểm tra tính nhất quán kích thước ảnh – mask, tính nhị phân `{0, 255}` của mask, đối chiếu lại `gt_box` từ mask đã cắt và lưu lưới ảnh minh họa `reports/build_ds1_ds2/check_samples.png`. |
| `transforms.py` | Cấu hình pipeline tăng cường dữ liệu Albumentations đồng bộ cho ảnh, mask và bounding box (`pascal_voc`). |
| `unified_dataset.py` | Lớp `UnifiedBrainMRIDataset(torch.utils.data.Dataset)` trả về đồng thời nhánh ViT (`224 × 224`) và nhánh MedSAM (`1024 × 1024`). |

## Lệnh thực thi nhanh

```bash
python -m data.eda_ds1_ds2
python -m data.build_ds1_ds2
python -m data.check_ds1_ds2
```


===


# Mô-đun `data/` — Kiểm định chất lượng & Tiền xử lý Dataset 3 (BRISC 2025)

Thư mục `data/` trên nhánh **`HNNQ_NTH_tuan1`** cung cấp bộ công cụ kiểm định chất lượng khớp nối 1-1 Ảnh – Mask, tăng cường tương phản cục bộ **CLAHE**, trích xuất tọa độ hộp giới hạn `gt_box = [x_min, y_min, x_max, y_max]` và chia tập **Stratified Train / Validation / Test (70% / 15% / 15%)** chống rò rỉ bệnh nhân cho **Dataset 3 – BRISC 2025**.

## Danh sách các tệp mã nguồn

| Tệp | Chức năng chính |
| :--- | :--- |
| `ds3_index.py` | Phân tích cú pháp tên tệp chuẩn `brisc2025_<split>_<index>_<tumor>_<plane>_<seq>`, kiểm tra đối sánh 1-1 hai chiều giữa `images/` và `masks/`. |
| `eda_ds3.py` | Kiểm định chất lượng toàn bộ 6.000 lát cắt T1 MRI và 4.793 mask, phát hiện 46 nhóm trùng lặp `SHA256`, xuất báo cáo và biểu đồ vào `reports/eda_ds3/`. |
| `build_ds3.py` | Kiểm định 1-1, lọc CLAHE, chuẩn hóa mask nhị phân `{0, 255}`, trích xuất `gt_box`, gom 3.646 cụm `patient_id` bằng đồ thị liên thông (Union-Find) và chia tập Stratified 70% / 15% / 15% ra `dataset/ds3_annotations.json`. |
| `check_ds3.py` | Xác nhận tự động toàn bộ 6.000 bản ghi: khớp kích thước 1-1, mask nhị phân `{0, 255}`, `gt_box` chính xác tuyệt đối, 0% rò rỉ `patient_id` & `SHA256` giữa `train/val/test`, xuất `reports/build_ds3/check_samples.png`. |
| `skull_crop.py` | Các hàm xử lý ảnh và mặt nạ dùng chung (`read_image`, `write_png`, `binarize_mask`, `mask_to_box`, `apply_clahe`, `skull_crop_box`). |
| `transforms.py` | Cấu hình pipeline tăng cường dữ liệu Albumentations đồng bộ cho ảnh, mask và bounding box (`pascal_voc`). |
| `unified_dataset.py` | Lớp `UnifiedBrainMRIDataset(torch.utils.data.Dataset)` trả về đồng thời nhánh ViT (`224 × 224`) và nhánh MedSAM (`1024 × 1024`). |

## Lệnh thực thi nhanh

```bash
python -m data.eda_ds3
python -m data.build_ds3
python -m data.check_ds3
```
