# BrainTumorDetect — Tuần 1: Tiền xử lý & Cắt lát 2D Dataset 4 (BraTS 2020)

Nhánh **`Lam_Gia_Bao_tuan1`** thực hiện toàn bộ quy trình khảo sát thống kê (EDA), chuẩn hóa cường độ tín hiệu Z-score 3D, cắt lát ngang 2D (Axial Slicing), gộp mặt nạ Whole Tumor (WT), trích xuất tọa độ Bounding Box chuẩn và lấy mẫu cân bằng lát lành cho bộ dữ liệu **BraTS 2020 (369 ca bệnh 3D MRI)**.

---

## 1. Cấu trúc thư mục trên nhánh

```text
BrainTumorDetect/
├── src/
│   └── data/
│       ├── eda_brats.py         # Khảo sát thống kê 3D BraTS2020 (hỗ trợ cả .nii/.nii.gz và .h5)
│       ├── build_dataset.py     # Pipeline chuẩn hóa Z-score, cắt lát 2D, tạo WT mask, bbox & chia tập theo ca
│       ├── check_dataset.py     # Kiểm định tự động không rò rỉ ca bệnh, ngưỡng diện tích u >= 50px, bbox & Z-score
│       └── README.md            # Hướng dẫn nhanh mô-đun src/data/
├── reports/
│   ├── eda_overview.txt         # Tổng kết số liệu khảo sát 369 ca BraTS2020
│   ├── eda_notes.md             # Phân tích chi tiết và quyết định thiết kế pipeline
│   ├── eda_summary.csv          # Thống kê từng ca bệnh (shape, spacing, số lát não, số lát u, thể tích WT)
│   ├── fig_*.png                # Các biểu đồ phân bố thể tích u, tỷ lệ lát u, cường độ tín hiệu theo modality
│   └── processed/               # Kết quả xuất mẫu: case_summary.csv, metadata.csv, splits.csv, check_samples.png
├── .gitignore
└── README.md
```

---

## 2. Tóm tắt kết quả khảo sát (EDA) trên 369 ca BraTS 2020

- **Số lượng & Định dạng**: **369/369** ca bệnh đầy đủ 4 chuỗi xung (`FLAIR`, `T1`, `T1ce`, `T2`) và mặt nạ phân vùng `seg`, không có ca nào thiếu modality hoặc lệch kích thước.
- **Kích thước & Hệ trục**: Đồng nhất `240 × 240 × 155` voxel, khoảng cách điểm ảnh `1.0 × 1.0 × 1.0 mm`, hướng trục `LPS` (155 lát cắt ngang theo trục $z$ mỗi ca).
- **Phân bố lát cắt**:
  - Trung bình **137.9 lát có mô não / ca** (dao động `98 – 149` lát) và **17.1 lát rỗng / ca** ở đỉnh sọ và nền sọ.
  - Trung bình **66.2 lát có u / ca** (trung vị `68`, dao động `23 – 102` lát). Tổng toàn tập gồm **24.422 lát có u** và **26.476 lát mô não lành**.
- **Nhãn tổn thương**: Gồm các nhãn `{0, 1, 2, 4}` (nhãn `3` không tồn tại trong chuẩn BraTS 2020). Thể tích khối u toàn phần (Whole Tumor – WT) có trung vị **90.7 $\text{cm}^3$** (`7.3 – 361.8` $\text{cm}^3$).

---

## 3. Quy trình xử lý trong `src/data/build_dataset.py`

1. **Đọc đa chuỗi xung (`.nii/.nii.gz` hoặc `.h5`)**: Trích xuất 3 chuỗi xung giàu thông tin cấu trúc và ranh giới u nhất gồm **`[FLAIR, T1ce, T2]`** cùng nhãn phân vùng `seg`.
2. **Chuẩn hóa Z-score theo từng Volume & Modality**:
   - Chỉ tính trung bình $\mu$ và độ lệch chuẩn $\sigma$ trên vùng mô não ($\text{voxel} > 0$):
     $$z = \frac{x - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$
   - Các điểm ảnh nền ngoài hộp sọ được giữ cố định bằng `0`.
3. **Cắt lát 2D theo mặt phẳng ngang (Axial) & Lọc nhiễu**:
   - Loại bỏ các lát rỗng ở đỉnh sọ và nền sọ có diện tích mô não `< 500 pixels` (`--min_brain_area 500`).
   - Chỉ giữ lại các lát có u đạt diện tích tối thiểu **`>= 50 pixels`** (`--min_tumor_area 50`) nhằm loại bỏ nhiễu điểm ảnh rời rạc ở rìa khối u.
   - Lấy mẫu cân bằng một lượng lát cắt mô não lành (`has_tumor = 0`) theo tỷ lệ `--neg_ratio` trên từng ca bệnh.
4. **Gộp nhãn Whole Tumor (WT) & Trích xuất Bounding Box**:
   - Hợp nhất các vùng hoại tử (`NCR/NET = 1`), phù nề quanh u (`ED = 2`) và u bắt thuốc (`ET = 4`) thành mặt nạ nhị phân **Whole Tumor**: $\text{WT} = (\text{seg} > 0)$.
   - Tự động trích xuất tọa độ hộp giới hạn chuẩn `[x_min, y_min, x_max, y_max]` bao quanh vùng WT.
5. **Đóng gói ảnh 3 kênh & Chia tập theo ca bệnh (Patient-level Split)**:
   - Ghép 3 kênh `[FLAIR, T1ce, T2]` thành ảnh `.png` 3 kênh (`dataset/brats2020_png/images/`), mặt nạ nhị phân `.png` (`dataset/brats2020_png/masks/`), tệp `.npz` (`float16`) và xuất tệp chú thích chuẩn `dataset/brats2020_annotations.json` tương thích trực tiếp với `UnifiedBrainMRIDataset`.
   - Chia tập **Train / Val / Test (70% / 15% / 15%) theo từng ca bệnh (`case_id`)**, đảm bảo các lát cắt của cùng một bệnh nhân không bao giờ bị rò rỉ giữa tập huấn luyện và tập kiểm thử.

---

## 4. Hướng dẫn cài đặt và chạy lệnh

### Cài đặt thư viện
```bash
pip install nibabel h5py opencv-python numpy pandas matplotlib tqdm
```

### Bước 1: Chạy khảo sát thống kê (EDA)
```bash
python src/data/eda_brats.py --data_dir dataset/brats2020 --out_dir reports
```

### Bước 2: Chạy pipeline cắt lát 2D, tạo mặt nạ WT, Bounding Box & xuất JSON
```bash
python src/data/build_dataset.py \
    --data_dir dataset/brats2020 \
    --out_dir dataset/brats2020_png \
    --out_json dataset/brats2020_annotations.json \
    --min_tumor_area 50 \
    --neg_ratio 1.0
```

### Bước 3: Kiểm định chất lượng dữ liệu đầu ra
```bash
python src/data/check_dataset.py --out_dir dataset/brats2020_png --min_tumor_area 50
```
