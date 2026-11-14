# Danh mục Ảnh Kết quả Thực nghiệm theo Tuần (`image/`)

Thư mục `image/` lưu trữ toàn bộ hình ảnh kết quả, sơ đồ quy trình và biểu đồ đánh giá của dự án, được phân chia trực quan theo từng tuần thực hiện:

```text
image/
├── tuan1/                      # Tuần 1: Tiền xử lý Hộp sọ, CLAHE, 3D-to-2D BraTS & Phân tích EDA 4 bộ dữ liệu
│   ├── fig1_multi_dataset_preprocessing_flowchart.png
│   ├── fig2_four_datasets_class_distribution.png
│   ├── fig3_resolution_and_split_comparison.png
│   ├── ds1_ds2_check_samples.png
│   ├── ds3_check_samples.png
│   ├── ds4_check_samples.png
│   ├── eda_ds1_ds2_class_counts.png
│   ├── eda_ds1_ds2_image_sizes.png
│   ├── eda_ds2_tumor_area.png
│   ├── eda_ds3_class_plane_counts.png
│   ├── eda_ds3_image_sizes.png
│   ├── eda_ds3_tumor_area.png
│   ├── eda_ds4_intensity_mean_per_modality.png
│   ├── eda_ds4_sample_case_full.png
│   ├── eda_ds4_tumor_slice_ratio.png
│   ├── eda_ds4_tumor_slices_per_case.png
│   ├── eda_ds4_wt_volume.png
│   └── samples_ds3/            # 9 lát cắt mẫu đối chiếu của BRISC 2025 (Glioma, Meningioma, Pituitary)
│
├── tuan2/                      # Tuần 2: ViT Baseline, Zero-shot SAM vs MedSAM & SAM Everything Mode
│   ├── vit_ds1_full_finetune_loss.png
│   ├── vit_ds2_learning_curves.png
│   ├── vit_ds3_training_curves_linear_probe.png
│   ├── vit_ds3_training_curves_full_finetune.png
│   ├── vit_ds3_confusion_matrix_linear_probe.png
│   ├── vit_ds3_confusion_matrix_full_finetune.png
│   ├── zeroshot_sam_vs_medsam_dice_boxplot.png
│   ├── zeroshot_sam_vs_medsam_delta_by_blur_tertile.png
│   ├── zeroshot_sam_vs_medsam_delta_vs_sharpness.png
│   ├── zeroshot_sam_vs_medsam_qualitative_ds2.png
│   ├── zeroshot_sam_vs_medsam_qualitative_ds4.png
│   ├── medsam_oracle_box_brisc2025_qualitative_grid.png
│   ├── sam_everything_summary_figure.png
│   └── sam_everything_overlays/ # 50 ảnh kiểm thử SAM Everything Mode (25 ảnh BRISC 2025 + 25 ảnh DS1 Navoneel)
│
└── tuan3/                      # Tuần 3: Kiến trúc ViT-LoRA & 2 Cơ chế Tự sinh Lời nhắc (Auto-Prompt Generator)
    ├── fig1_lora_params_and_training_curves.png
    ├── fig2_mechanism1_rollout_layercam_ds1.png
    └── fig3_mechanism2_vitdet_box_head_ds2_ds3_ds4.png
```
