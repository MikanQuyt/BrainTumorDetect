# Tuần 2: Zero-shot SAM và MedSAM với hộp chuẩn (Oracle GT Box) trên DS2 và DS4

Tác giả: Lâm Gia Bảo. Nhánh: `Lam_Gia_Bao_tuan2`.

## 1. Mục tiêu

1. Chạy zero-shot SAM gốc và zero-shot MedSAM với prompt chuẩn (hộp giới hạn lấy từ mask chuẩn) trên Dataset 2 (indk214) và Dataset 4 (BraTS2020).
2. Đánh giá định lượng MedSAM so với SAM gốc, đặc biệt trên các khối u có biên mờ.
3. Xác định mức trần điểm phân vùng (upper-bound Dice và IoU) khi dùng hộp chuẩn, làm mốc tham chiếu cho các tuần sau.

## 2. Thiết lập thí nghiệm

| Mục | Cấu hình |
|---|---|
| Mô hình | SAM ViT-B (`sam_vit_b_01ec64.pth`) và MedSAM ViT-B (`medsam_vit_b.pth`), không huấn luyện thêm |
| Prompt | Một hộp giới hạn lấy từ mask chuẩn (`gt_box`, biên loại trừ); `multimask_output=False` |
| SAM | `SamPredictor`, ảnh RGB uint8 |
| MedSAM | Quy trình suy luận chính thức: resize 1024×1024, chuẩn hóa min-max về [0,1], hộp quy đổi sang hệ 1024, ngưỡng sigmoid 0,5 |
| Đối chứng | **Box-fill**: tô kín cả hộp chuẩn làm mask dự đoán |
| DS4 (BraTS2020) | Tập `val`, chỉ các lát có u (`gt_box` khác rỗng): **3.313 lát, 55 ca**. Ảnh PNG 3 kênh `[FLAIR, T1ce, T2]` (Z-score, cắt ở ±3σ, quy về 8-bit) |
| DS2 (indk214) | Đánh giá trên cả **Tập Validation chuẩn hóa (329 ảnh, chia Stratified 70/15/15 từ 2.192 cặp ảnh–mask theo lớp u tại `ds2/ds2_split.csv`)** lẫn **Toàn bộ 2.192 cặp ảnh – mask phân vùng** (sau cắt sọ + CLAHE) |
| Chỉ số | Dice và IoU, tính ở độ phân giải của ảnh đã lưu |
| Thống kê | Chênh lệch ghép cặp (MedSAM − SAM) theo từng ảnh; khoảng tin cậy 95% bằng bootstrap 5.000 lần lấy mẫu lại theo ca; kiểm định Wilcoxon trên điểm trung bình theo ca |
| Độ mờ biên | Gradient Sobel trung bình trong dải hẹp (khoảng 4 px) quanh viền mask chuẩn, trên kênh đầu tiên (FLAIR với DS4), chuẩn hóa theo khoảng cường độ; **giá trị nhỏ = biên mờ**. Chia 3 nhóm phân vị (T1 = mờ nhất) |
| Phần cứng | GPU NVIDIA RTX 4060 (8 GB) |

Tập DS2 không có mã bệnh nhân nên mỗi ảnh được xem là một nhóm độc lập khi bootstrap (xem mục Hạn chế).

## 3. Kết quả

### 3.1. Dice và IoU với hộp chuẩn

**DS4, tập val (3.313 lát, 55 ca)**

| Phương pháp | Dice (TB lát) | Dice (TB theo ca) | IoU (TB lát) | IoU (TB theo ca) |
|---|---|---|---|---|
| SAM | 0,787 | 0,777 | 0,680 | 0,670 |
| MedSAM | 0,788 | 0,782 | 0,678 | 0,670 |
| Box-fill | 0,665 | 0,664 | 0,513 | 0,513 |

**DS2, tập Validation (329 ảnh, Stratified 15% — chi tiết tại `val_protocol/week2_summary.md`)**

| Phương pháp | Dice (TB) | Độ lệch chuẩn | IoU (TB) | Độ lệch chuẩn |
|---|---|---|---|---|
| SAM | 0,844 | 0,119 | 0,745 | 0,155 |
| MedSAM | **0,846** | **0,099** | **0,745** | **0,137** |
| Box-fill | 0,824 | 0,072 | 0,707 | 0,097 |

**DS2, toàn bộ 2.192 ảnh**

| Phương pháp | Dice (TB) | Độ lệch chuẩn | IoU (TB) | Độ lệch chuẩn |
|---|---|---|---|---|
| SAM | 0,848 | 0,115 | 0,751 | 0,150 |
| MedSAM | 0,845 | 0,099 | 0,743 | 0,137 |
| Box-fill | 0,824 | 0,073 | 0,707 | 0,098 |

### 3.2. So sánh ghép cặp MedSAM − SAM

| Dataset | Chỉ số | Δ trung bình | Khoảng tin cậy 95% | Tỷ lệ MedSAM thắng | Wilcoxon p (theo ca) |
|---|---|---|---|---|---|
| DS4 | Dice | +0,002 | −0,010 đến 0,014 | 47,7% | 0,669 |
| DS4 | IoU | −0,002 | −0,017 đến 0,012 | 47,7% | 0,558 |
| DS2 | Dice | −0,003 | −0,007 đến 0,002 | 46,8% | < 0,001 |
| DS2 | IoU | −0,007 | −0,013 đến −0,002 | 46,8% | < 0,001 |

### 3.3. Theo độ mờ của biên (Δ Dice = MedSAM − SAM)

| Dataset | Nhóm | n | Dice SAM | Dice MedSAM | Δ | Khoảng tin cậy 95% |
|---|---|---|---|---|---|---|
| DS4 | T1 mờ nhất | 1.104 | 0,739 | 0,757 | +0,018 | 0,001 đến 0,036 |
| DS4 | T2 trung bình | 1.104 | 0,798 | 0,811 | +0,013 | −0,005 đến 0,033 |
| DS4 | T3 nét nhất | 1.105 | 0,824 | 0,798 | −0,026 | −0,037 đến −0,015 |
| DS2 | T1 mờ nhất | 731 | 0,798 | 0,816 | +0,018 | 0,009 đến 0,027 |
| DS2 | T2 trung bình | 730 | 0,845 | 0,842 | −0,004 | −0,011 đến 0,004 |
| DS2 | T3 nét nhất | 731 | 0,901 | 0,878 | −0,023 | −0,030 đến −0,017 |

### 3.4. Theo kích thước u (Δ Dice = MedSAM − SAM)

| Dataset | Nhóm (diện tích mask, px) | n | Dice SAM | Dice MedSAM | Δ | Khoảng tin cậy 95% |
|---|---|---|---|---|---|---|
| DS4 | T1 nhỏ (50–782) | 1.104 | 0,686 | 0,659 | −0,027 | −0,044 đến −0,008 |
| DS4 | T2 vừa (782–2.017) | 1.104 | 0,810 | 0,823 | +0,013 | −0,007 đến 0,036 |
| DS4 | T3 lớn (2.018–5.330) | 1.105 | 0,864 | 0,883 | +0,019 | 0,006 đến 0,032 |
| DS2 | T1 nhỏ (277–2.069) | 731 | 0,856 | 0,823 | −0,033 | −0,041 đến −0,026 |
| DS2 | T2 vừa (2.077–4.863) | 730 | 0,854 | 0,856 | +0,002 | −0,006 đến 0,009 |
| DS2 | T3 lớn (4.863–25.461) | 731 | 0,833 | 0,856 | +0,023 | 0,015 đến 0,031 |

Biểu đồ minh họa (cùng thư mục): `fig_dice_boxplot.png`, `fig_delta_vs_sharpness.png`, `fig_delta_by_blur_tertile.png`, `fig_qualitative_ds4.png`, `fig_qualitative_ds2.png`. Các bảng đầy đủ nằm trong `summary_table.csv`, `paired_stats.csv`, `strata_blur.csv`, `strata_size.csv`.

## 4. Nhận xét

1. **Không có bằng chứng MedSAM vượt trội SAM ở mức toàn tập.** Ở DS4, chênh lệch Dice là +0,002 (khoảng tin cậy 95% đi qua 0, p = 0,669). Ở DS2, chênh lệch là −0,003 (khoảng tin cậy đi qua 0); phép kiểm định Wilcoxon cho p < 0,001 nhưng cỡ hiệu ứng rất nhỏ (d_z = −0,026 với Dice), nên sự khác biệt này có thể phát hiện được về mặt thống kê nhưng không đáng kể về mặt thực tế.
2. **Cùng một kiểu khác biệt xuất hiện ở cả hai dataset.** MedSAM nhỉnh hơn SAM ở nhóm biên mờ nhất (+0,018 ở cả DS4 và DS2) và ở u lớn (+0,019 ở DS4, +0,023 ở DS2), trong khi SAM nhỉnh hơn ở nhóm biên nét nhất (−0,026 và −0,023) và ở u nhỏ (−0,027 và −0,033). Lợi thế của MedSAM trên biên mờ như vậy chỉ ở mức khoảng 0,02 Dice và không đủ để thành lợi thế chung.
3. **Cả hai mô hình đều hơn đáng kể việc chỉ tô kín hộp chuẩn ở DS4** (Dice 0,78 so với 0,66), nhưng ở **DS2 mức tăng nhỏ** (0,85 so với 0,82), vì u ở DS2 thường chiếm phần lớn diện tích hộp; phần lớn mốc điểm ở DS2 đến từ chính thông tin của hộp chuẩn.
4. Ảnh định tính cho thấy cả hai mô hình đều bám sát vùng u; sai khác chủ yếu ở việc có bao gồm vùng phù nề quanh lõi u hay không và ở các u rất nhỏ.

## 5. Mốc upper-bound cho các tuần sau (hộp chuẩn, zero-shot)

| Dataset | Mô hình tốt nhất (theo Dice TB theo ca) | Dice | IoU | Box-fill (Dice) |
|---|---|---|---|---|
| DS4 (val, 55 ca) | MedSAM | 0,7822 | 0,6703 | 0,664 |
| DS2 (2.192 ảnh) | SAM | 0,8479 | 0,7508 | 0,824 |

Chênh lệch giữa hai mô hình rất nhỏ (xem mục 3.2), nên mốc nên được hiểu là khoảng 0,78 Dice ở DS4 và khoảng 0,85 Dice ở DS2. Chi tiết và quy trình đo nằm trong `oracle_upper_bound.json`. Các tuần sau, khi hộp lấy từ bộ phát hiện thay cho hộp chuẩn, khoảng cách tới mốc này đo được độ hao hụt do lỗi định vị hộp.

## 6. Hạn chế

- **DS2 không có tập Validation định nghĩa sẵn.** Kết quả DS2 tính trên toàn bộ 2.192 ảnh. Khi nhóm chốt cột `split` cho DS2, có thể lọc lại từ `reports/week2/ds2/oracle_results.csv` mà không cần chạy lại mô hình.
- **DS2 không có mã bệnh nhân**, mỗi ảnh được coi là một nhóm độc lập; nếu nhiều ảnh thuộc cùng một bệnh nhân, khoảng tin cậy và p-value của DS2 có thể quá lạc quan.
- Phân tầng theo độ mờ và theo kích thước là phân tích khám phá (6 so sánh, không hiệu chỉnh nhiều phép thử); hai yếu tố này có thể liên quan với nhau và chưa được tách riêng.
- "Độ mờ biên" là chỉ số do chúng tôi tự định nghĩa (gradient quanh viền trên kênh đầu tiên), chỉ là một cách đo gần đúng.
- Hộp chuẩn sát mask nên mốc upper-bound là lạc quan so với hộp lấy từ bộ phát hiện.
- Chỉ dùng ViT-B và một cách lấy hộp; ảnh DS4 là PNG 8-bit cắt ở ±3σ (mất thông tin ở giá trị cực trị). Pipeline nạp ảnh của nhóm (`unified_dataset.py`) chia 255 cho MedSAM, còn thí nghiệm này dùng min-max theo ảnh theo quy trình chính thức; với ảnh có nền đen hai cách gần như trùng nhau.
- Chưa có lời giải thích đã kiểm chứng cho việc MedSAM không nổi trội; giả thuyết (ảnh đầu vào là 3 chuỗi xung khác nhau thay vì ảnh xám nhân ba) cần được kiểm tra thêm, chưa phải kết luận.

## 7. Tái lập

```bash
# DS4 (val)
python eval/zeroshot_oracle_eval.py --ann dataset/brats2020_annotations.json --root . --split val --sam_ckpt <sam_vit_b_01ec64.pth> --medsam_ckpt <medsam_vit_b.pth> --tag ds4 --out_dir reports/week2/ds4
# DS2 (toàn bộ, ảnh tạo lại bằng: python -m data.build_ds1_ds2 --root <thư mục làm việc>)
python eval/zeroshot_oracle_eval.py --ann <thư mục làm việc>/dataset/ds2_annotations.json --root <thư mục làm việc> --split "" --sam_ckpt <...> --medsam_ckpt <...> --tag ds2 --out_dir reports/week2/ds2
# Phân tích
python eval/analyze_week2.py --csv ds4=reports/week2/ds4/oracle_results.csv ds2=reports/week2/ds2/oracle_results.csv --out_dir reports/week2
```

Phiên bản thư viện và checksum checkpoint được ghi trong `reports/week2/ds4/run_info.json` và `reports/week2/ds2/run_info.json`.
