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
