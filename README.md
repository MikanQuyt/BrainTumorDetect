# BRISC 2025 - Tiền Xử Lý và Kiểm Định Dataset MRI U Não

Dự án thực hiện đầy đủ quy trình kiểm định chất lượng dữ liệu BRISC 2025, tiền xử lý tăng cường tương phản bằng CLAHE, trích xuất Bounding Box từ Segmentation Mask, và phân chia tập dữ liệu Train / Validation / Test theo tỷ lệ 70% / 15% / 15% đảm bảo **không rò rỉ dữ liệu theo bệnh nhân (Zero Patient Data Leakage)**.

---

## 1. Kết Quả Kiểm Định Chất Lượng Dataset (Data Quality Audit)

- **Đường dẫn dataset gốc:** `E:\archive\brisc2025`
- **Tính toàn vẹn Checksum SHA-256:**
  - `manifest.csv`: **MATCH (Hợp lệ 100%)**
  - `manifest.json`: **MATCH (Hợp lệ 100%)**
- **Kiểm tra file thiếu (Missing files):** **0** / 15,586 file.
- **Kiểm tra file lỗi/hỏng (Corrupted files):** **0** file lỗi đọc.
- **Kiểm tra nhãn không hợp lệ (Invalid labels):** **0** nhãn sai (100% thuộc `gl`, `me`, `pi`, `no` và `ax`, `co`, `sa`).
- **Tương thích kích thước ảnh MRI và Mask:**
  - Tổng số cặp Segmentation đã kiểm tra: **4,793 cặp**
  - Số cặp sai lệch kích thước: **0** (100% đều đạt chuẩn $512 \times 512$).
- **Đặc điểm Mask:**
  - Các giá trị pixel: `[0, 1, 2, ..., 10, 246, ..., 255]`.
  - Có hiện tượng làm mịn biên (antialiasing), đã áp dụng binarization với ngưỡng `mask > 127` để thu được mask nhị phân hoàn hảo.

---

## 2. Kỹ Thuật Tiền Xử Lý Áp Dụng

### A. Tăng Cường Tương Phản CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Tham số:** `clipLimit=2.0`, `tileGridSize=(8, 8)`
- **Ý nghĩa y khoa:** Giúp phân tách rõ ranh giới giữa mô u, chất trắng (White Matter), chất xám (Gray Matter) và dịch não tủy (CSF) trong ảnh chụp cộng hưởng từ T1 mà không làm khuếch đại nhiễu nền.

### B. Trích Xuất Bounding Box Từ Mask
- Phát hiện biên u bằng `cv2.findContours`.
- Trích xuất 2 định dạng:
  1. **Pixel Box (`xyxy`):** `[xmin, ymin, xmax, ymax]` dùng cho segmentation ROI, crop vùng u.
  2. **YOLO Normalized Box:** `[class_id, x_center, y_center, width, height]` chuẩn hóa $[0, 1]$ dùng cho các mô hình Object Detection (YOLOv8, YOLOv11, Faster R-CNN).

### C. Phân Chia Dữ Liệu Train / Val / Test (70% / 15% / 15%) Theo Bệnh Nhân
- **Gom cụm chuỗi lát cắt bệnh nhân:** 4,793 lát cắt u được gom thành **402 nhóm bệnh nhân** (`patient_id`) dựa trên chuỗi cắt giải phẫu liên tục của từng ca chụp.
- **Phân chia bằng Group Stratified Split:**
  - **Train:** **3,354 slices (~69.98%)** - 281 bệnh nhân
  - **Validation:** **707 slices (~14.75%)** - 60 bệnh nhân
  - **Test:** **732 slices (~15.27%)** - 61 bệnh nhân
- **Khẳng định chống rò rỉ dữ liệu (Zero Data Leakage):**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 3. Cấu Trúc Thư Mục

```
E:\archive\brisc2025_pipeline\
├── audit_dataset.py               # Script kiểm định chất lượng và checksum
├── preprocess_pipeline.py         # Pipeline CLAHE, bbox extraction & patient split
├── git_workflow.sh               # Script tự động hoá quy trình Git
├── README.md                      # Tài liệu báo cáo dự án
└── output\
    ├── audit_report.json          # Báo cáo JSON kiểm định chất lượng
    ├── brisc2025_processed_manifest.csv # Bảng tổng hợp toạ độ bbox và phân chia tập
    └── samples\                   # Ảnh đối chứng 4 khung: Original - CLAHE - Mask - BBox
```

---

## 4. Hướng Dẫn Git Workflow (Khi Được Thêm Vào Repo)

Git Bash đã được cài đặt sẵn trên máy tại `C:\Program Files\Git\git-bash.exe`.

Mở Git Bash và thực hiện tuần tự:

```bash
# 1. Clone repository về máy
git clone <URL_REPO_CUA_BAN>
cd <TEN_THU_MUC_REPO>

# 2. Chuyển vào nhánh dev
git checkout dev

# 3. Tạo và chuyển sang nhánh tính năng ten_minh_tuan1
git checkout -b ten_minh_tuan1

# 4. Copy mã nguồn từ E:\archive\brisc2025_pipeline vào repo
cp -r /e/archive/brisc2025_pipeline/* .

# 5. Lưu lại commit
git add .
git commit -m "feat: Add BRISC 2025 audit, CLAHE enhancement, bbox extraction, and patient split 70/15/15"

# 6. Quay trở lại nhánh dev và gộp nhánh
git checkout dev
git merge ten_minh_tuan1

# 7. Đẩy lên nhánh dev trên remote repo
git push origin dev
```
