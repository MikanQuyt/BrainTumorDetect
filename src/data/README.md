# Pipeline dữ liệu BraTS 2020 (Cắt lát 2D – `src/data/`)

Mô-đun `src/data/` chứa toàn bộ mã nguồn khảo sát 3D, tiền xử lý cắt lát 2D và kiểm định chất lượng cho **Dataset 4 – BraTS 2020** trên nhánh `Lam_Gia_Bao_tuan1`.

## 1. Yêu cầu môi trường
- Python 3.10+
- Cài đặt các gói phụ thuộc:
  ```bash
  pip install nibabel h5py opencv-python numpy pandas matplotlib tqdm
  ```

## 2. Danh sách các script trong `src/data/`

| Tệp mã nguồn | Chức năng chính |
| :--- | :--- |
| `eda_brats.py` | Khảo sát toàn bộ 369 ca BraTS 2020 (hỗ trợ cả `.nii/.nii.gz` và `.h5`), xuất thống kê ra `reports/eda_summary.csv`, `reports/eda_overview.txt` và các biểu đồ `reports/fig_*.png`. |
| `build_dataset.py` | Chuẩn hóa Z-score trên vùng não ($\text{voxel} > 0$), cắt lát Axial theo trục $z$, lọc lát rỗng (`< 500 px`), giữ lát u có diện tích `>= 50 px`, lấy mẫu cân bằng lát lành, tạo mặt nạ Whole Tumor $\text{WT} = (\text{seg} > 0)$, trích xuất Bounding Box `[x_min, y_min, x_max, y_max]`, lưu ảnh 3 kênh `[FLAIR, T1ce, T2]` (`.png` & `.npz`) và xuất `dataset/brats2020_annotations.json`. |
| `check_dataset.py` | Kiểm tra tự động: không rò rỉ `case_id` giữa `train/val/test`, mọi lát có u đều thỏa `tumor_area >= 50 px`, kiểm tra khớp kích thước ảnh – mask, tính nhị phân của mask, độ chính xác của `bbox`, thống kê Z-score và vẽ lưới minh họa `check_samples.png`. |

## 3. Cách chạy

1. **Khảo sát 3D (EDA)**:
   ```bash
   python src/data/eda_brats.py --data_dir dataset/brats2020 --out_dir reports
   ```
2. **Tạo bộ dữ liệu 2D (`.png` 3 kênh `[FLAIR, T1ce, T2]` + `.npz` + `dataset/brats2020_annotations.json`)**:
   ```bash
   python src/data/build_dataset.py \
       --data_dir dataset/brats2020 \
       --out_dir dataset/brats2020_png \
       --out_json dataset/brats2020_annotations.json \
       --min_tumor_area 50 \
       --neg_ratio 1.0
   ```
3. **Kiểm tra chất lượng đầu ra**:
   ```bash
   python src/data/check_dataset.py --out_dir dataset/brats2020_png --min_tumor_area 50
   ```

## 4. Tóm tắt luồng xử lý
Đọc `FLAIR`, `T1ce`, `T2` + `seg` (từ `.nii/.nii.gz` hoặc `.h5`) $\rightarrow$ chuẩn hóa Z-score theo từng volume và modality (trên voxel não $> 0$, nền $= 0$) $\rightarrow$ cắt lát ngang (Axial) theo trục $z$ $\rightarrow$ lọc lát rỗng ở đỉnh sọ và nền sọ $\rightarrow$ giữ lại các lát có diện tích u $\ge 50\text{ pixels}$ và lấy mẫu cân bằng lát mô não lành $\rightarrow$ tạo mặt nạ Whole Tumor $\text{WT} = (\text{seg} > 0)$ kèm Bounding Box chuẩn $\rightarrow$ ghép 3 kênh `[FLAIR, T1ce, T2]` lưu thành ảnh `.png` 3 kênh (và `.npz`) $\rightarrow$ chia `train/val/test` (70% / 15% / 15%) theo ca bệnh (`case_id`).
