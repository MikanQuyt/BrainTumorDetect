# BrainTumorDetect — Tuần 1: EDA, Cắt sát hộp sọ tự động (Skull Cropping + CLAHE) & Trích xuất Bounding Box cho Dataset 1 & 2

**Sinh viên thực hiện:** Thái Văn Huy · MSSV: 23130068 (`thai_van_huy_tuan1`)

Nhánh **`thai_van_huy_tuan1`** thực hiện trọn vẹn 2 nhiệm vụ Tuần 1:
1. **Tải và phân tích thống kê (EDA)** cho **Dataset 1 (`navoneel` – 253 ảnh)** và **Dataset 2 (`indk214` – Segmentation & Classification)**.
2. **Lập trình và chạy script Cắt sát hộp sọ tự động (Automated Extreme-Point Skull Cropping bằng OpenCV + CLAHE)** trên toàn bộ Dataset 1 & 2; tự động trích xuất tọa độ hộp chuẩn **`[x_min, y_min, x_max, y_max]`** bao quanh Ground-Truth Mask của Dataset 2 xuất ra tệp **`ds2_annotations.json`**.

> Xem chi tiết nội dung rà soát và chỉnh sửa tại: **[chinhsuaHuy.md](chinhsuaHuy.md)** và báo cáo thống kê chi tiết tại **[output_lan2/BAO_CAO_TUAN1.md](output_lan2/BAO_CAO_TUAN1.md)**.

---

## 1. Cấu trúc thư mục trên nhánh

```text
BrainTumorDetect/
├── data/
│   ├── skull_crop.py                # Thuật toán Extreme-Point Skull Cropping (OpenCV), CLAHE & mask_to_box
│   ├── ds12_index.py                # Bộ lập chỉ mục chuẩn hóa cho Dataset 1 (navoneel) và Dataset 2 (indk214)
│   ├── eda_ds1_ds2.py               # Phân tích thống kê (EDA) & xuất biểu đồ cho Dataset 1 & 2
│   ├── build_ds1_ds2.py             # Chạy Skull Cropping + CLAHE toàn bộ DS1 & DS2, xuất ds1/ds2_annotations.json
│   ├── check_ds1_ds2.py             # Kiểm chứng tự động ảnh/mask đã cắt, tính nhị phân {0, 255} và tọa độ gt_box
│   └── README.md                    # Hướng dẫn chi tiết mô-đun data/
├── preprocess.py                    # Script tích hợp thống kê, cắt hộp sọ cực trị + CLAHE & xuất output_lan2/ds2_annotations.json
├── output_lan2/
│   ├── BAO_CAO_TUAN1.md             # Báo cáo phân tích thống kê (EDA) và kết quả tiền xử lý Dataset 1 & 2
│   ├── dataset_statistics.json      # Toàn bộ số liệu thống kê kích thước, phân bố nhãn, ảnh trùng & diện tích u
│   ├── ds2_annotations.json         # Tọa độ hộp chuẩn [x_min, y_min, x_max, y_max] của 2.192 cặp ảnh–mask Dataset 2
│   └── ds_annotations.json          # Bản sao tương thích ngược
├── dataset/
│   ├── ds1_annotations.json         # 253 bản ghi chuẩn hóa của Dataset 1 (sau khi cắt hộp sọ + CLAHE)
│   └── ds2_annotations.json         # 9.215 bản ghi chuẩn hóa của Dataset 2 (2.192 mẫu có mask + gt_box, 7.023 mẫu phân loại)
├── reports/
│   ├── eda_ds1_ds2/                 # Báo cáo & biểu đồ EDA: eda_summary.txt, class_counts.csv, fig_*.png
│   └── build_ds1_ds2/               # Nhật ký cắt hộp sọ (crop_log.csv) và lưới ảnh kiểm chứng (check_samples.png)
├── chinhsuaHuy.md                   # Báo cáo kiểm tra và các hạng mục đã hoàn thiện trên nhánh thai_van_huy_tuan1
└── README.md
```

---

## 2. Tóm tắt kết quả EDA – Dataset 1 & Dataset 2

- **Dataset 1 (`navoneel`)**: **253 ảnh** (`yes` – có u: **155**, `no` – không u: **98**), đã loại bỏ 253 tệp bản sao trong `brain_tumor_dataset/`; gồm **184 kích thước ảnh khác nhau** (`150×168` đến `1920×1427`), **0 ảnh hỏng**.
- **Dataset 2 (`indk214` – Classification)**: **7.023 ảnh** (`Training`: **5.712**, `Testing`: **1.311**) thuộc 4 lớp (`no_tumor`: **2.000**, `pituitary`: **1.757**, `meningioma`: **1.645**, `glioma`: **1.621**).
- **Dataset 2 (`indk214` – Segmentation)**: **2.192 cặp ảnh – mask** (`pituitary`: **930**, `meningioma`: **708**, `glioma`: **554**), đồng nhất kích thước `512 × 512`, **0 ảnh thiếu mask**, **0 mask rỗng**. Nền mask gốc có mức xám `2–3` do nén ảnh nên được nhị phân hóa bằng ngưỡng `>= 128` (`> 127`).

---

## 3. Hướng dẫn chạy lệnh

```bash
# Chạy phân tích thống kê (EDA) và xuất biểu đồ vào reports/eda_ds1_ds2/
python -m data.eda_ds1_ds2

# Chạy Automated Extreme-Point Skull Cropping + CLAHE và xuất dataset/ds2_annotations.json
python -m data.build_ds1_ds2

# Kiểm chứng tự động toàn bộ ảnh/mask đã cắt và tọa độ gt_box [x_min, y_min, x_max, y_max]
python -m data.check_ds1_ds2

# Hoặc chạy script tích hợp preprocess.py để xuất ra output_lan2/ds2_annotations.json
python preprocess.py --navoneel dataset/navoneel --indk214 dataset/indk214 --output output_lan2
```
