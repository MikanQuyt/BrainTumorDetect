# Pipeline dữ liệu BraTS2020 (cắt lát 2D)

## Yêu cầu
Python 3.10+; `pip install nibabel h5py opencv-python numpy pandas matplotlib tqdm`

## Chạy
1. Khảo sát 3D (hỗ trợ cả `.nii/.nii.gz` và `.h5`):
   ```bash
   python src/data/eda_brats.py --data_dir dataset/brats2020 --out_dir reports
   ```
2. Tạo dataset 2D (`.png` 3 kênh `[FLAIR, T1ce, T2]` + `.npz` + `dataset/brats2020_annotations.json`):
   ```bash
   python src/data/build_dataset.py --data_dir dataset/brats2020 --out_dir dataset/brats2020_png --min_tumor_area 50 --neg_ratio 1.0
   ```
3. Kiểm tra chất lượng:
   ```bash
   python src/data/check_dataset.py --out_dir dataset/brats2020_png
   ```

## Các bước xử lý
Đọc FLAIR/T1ce/T2 + seg (từ `.nii/.nii.gz` hoặc `.h5`) $\rightarrow$ chuẩn hóa Z-score theo từng volume và modality (trên voxel não $> 0$, nền $= 0$) $\rightarrow$ cắt lát ngang (Axial) theo trục $z$ $\rightarrow$ lọc lát rỗng ở đỉnh sọ và nền sọ $\rightarrow$ giữ lại các lát có diện tích u $\ge 50\text{ pixels}$ và lấy mẫu cân bằng lát mô não lành $\rightarrow$ tạo mặt nạ Whole Tumor $\text{WT} = (\text{seg} > 0)$ kèm Bounding Box chuẩn $\rightarrow$ ghép 3 kênh `[FLAIR, T1ce, T2]` lưu thành ảnh `.png` 3 kênh (và `.npz`) $\rightarrow$ chia `train/val/test` theo ca bệnh.
