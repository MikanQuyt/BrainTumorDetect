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
