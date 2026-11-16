# BÁO CÁO THỰC NGHIỆM TUẦN 2 — NGUYỄN PHƯỚC HOÀNG QUÂN
**Đề tài:** Hệ thống Phát hiện & Phân vùng Khối u Não trên Ảnh MRI kết hợp ViT-LoRA và MedSAM  
**Thành viên thực hiện:** Nguyễn Phước Hoàng Quân (Đảm nhiệm Tổng hợp Chuẩn hóa Điểm số Thực nghiệm & Trực quan hóa Lỗi Tràn Hộp sọ)  
**Nhóm nghiên cứu:** Alive Idiots (Bảo, Hải, Huy, Quyết, Quân)

---

## I. MỤC TIÊU VÀ PHÂN CÔNG NHIỆM VỤ TUẦN 2
Trong khuôn khổ nghiên cứu Tuần 2, sau khi đã hoàn thành **Chương 1 (Tổng quan đề tài & 4 bộ dữ liệu)** và **Chương 3 (Cơ sở lý thuyết về ViT, Self-Attention, SAM/MedSAM và LoRA)**, Hoàng Quân đảm nhận 2 nhiệm vụ cốt lõi:
1. **Tổng hợp và chuẩn hóa bảng điểm thực nghiệm Baseline** (Vision Transformer Baseline & Zero-shot SAM/MedSAM) trên cả 4 bộ dữ liệu từ kết quả đo đạc độc lập của Thái Văn Huy, Lâm Gia Bảo và Nguyễn Thành Hải.
2. **Thiết kế hình ảnh minh họa trực quan lỗi phân vùng tràn hộp sọ** của SAM nguyên bản (cả chế độ tự động Everything Mode và chế độ Box Mode) so với MedSAM chuyên biệt y tế, cung cấp bằng chứng định lượng và định tính phục vụ báo cáo hội đồng.

---

## II. BẢNG TỔNG HỢP & CHUẨN HÓA ĐIỂM SỐ BASELINE TRÊN CẢ 4 BỘ DỮ LIỆU

### 1. Phân loại ViT Baseline (Linear Probing vs. Full Fine-tuning)
Mô hình cơ sở sử dụng kiến trúc `vit_tiny_patch16_224` (ImageNet-21k pretrained, 12 tầng Transformer, $d=192$, 3 heads) được đánh giá trên 3 bộ dữ liệu có nhãn phân loại:

| Bộ dữ liệu | Phương thức thích nghi | Quy mô & Chiến lược chia tập | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Recall Lớp Có U (%) | Thời gian Train | Hiện tượng học & Đánh giá |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Dataset 1 (Navoneel)** | **Linear Probing** (Đóng băng ViT) | 253 ảnh (5-Fold Stratified Group CV) | $77.45 \pm 6.23$ | $82.20 \pm 6.46$ | $81.29 \pm 7.36$ | $81.51 \pm 5.13$ | **$81.29\%$** (bỏ sót 29/155 u) | 125.4s | **Underfitting**: Biểu diễn ImageNet không bắt được đặc trưng mô não, Recall $< 88\%$. |
| **Dataset 1 (Navoneel)** | **Full Fine-tuning** (Cập nhật 12 tầng) | 253 ảnh (5-Fold Stratified Group CV) | $88.55 \pm 4.65$ | $92.27 \pm 3.83$ | $89.03 \pm 9.29$ | $90.32 \pm 4.62$ | **$89.03\%$** (bỏ sót 17/155 u) | 382.1s | **Overfitting nghiêm trọng**: 4/5 fold val loss tăng vọt từ epoch 4-6 dù train loss tiệm cận 0. |
| **Dataset 2 (indk214)** | **Linear Probing** (Đóng băng ViT) | 1.029 ảnh test (4 lớp: 3 loại u + Không u) | $80.86$ | $80.32$ | $80.21$ | $80.12$ | **$95.58\%$** (gộp 3 u) | 840.2s | Kém trên U màng não (Meningioma Recall chỉ $64.46\%$, Glioma $76.54\%$). |
| **Dataset 2 (indk214)** | **Full Fine-tuning** (Cập nhật 12 tầng) | 1.029 ảnh test (4 lớp: 3 loại u + Không u) | $93.88$ | $94.00$ | $93.54$ | $93.62$ | **$98.66\%$** (gộp 3 u) | 1,650.5s | Đạt điểm cao nhờ tập dữ liệu lớn (7.023 ảnh), hội tụ tối ưu tại epoch 8. |
| **Dataset 3 (BRISC 2025)** | **Linear Probing** (Đóng băng ViT) | 900 ảnh test chuẩn (4 lớp phân cụm bệnh nhân) | **$82.75$** | $82.74$ | $82.75$ | **$82.63$** | **$92.14\%$** (gộp 3 u) | 340.3s | Thiết lập mốc đối sánh chuẩn cơ sở (Benchmark Baseline). |
| **Dataset 3 (BRISC 2025)** | **Full Fine-tuning** (Cập nhật 12 tầng) | 900 ảnh test chuẩn (4 lớp phân cụm bệnh nhân) | **$72.00$** | $74.09$ | $72.00$ | **$68.39$** | **$81.20\%$** (gộp 3 u) | 224.3s | **Sụp đổ hiệu năng (-10.75% Acc, -14.24% F1)** do ghi nhớ nhiễu nền khi tập mẫu phức tạp. |

> **Bằng chứng ma trận nhầm lẫn gộp trên Dataset 1 (Navoneel):**
> - *Linear Probing:* Đúng không u (TN) = 70; Báo u nhầm (FP) = 28; **Bỏ sót u (FN) = 29**; Đúng có u (TP) = 126.
> - *Full Fine-tuning:* Đúng không u (TN) = 86; Báo u nhầm (FP) = 12; **Bỏ sót u (FN) = 17**; Đúng có u (TP) = 138.
> - *Chứng cứ Overfitting trên Fold 3 (DS1):* Train loss giảm từ $0.07560 \to 0.00016$, nhưng Validation loss tăng vọt từ $0.3807 \to 0.5468$.

---

### 2. Phân vùng Zero-shot SAM vs. MedSAM với Hộp Chuẩn (Oracle Box)
Thực nghiệm đo đạc trần điểm số (Upper-bound) khi cung cấp Bounding Box chuẩn xác trích xuất từ Ground-Truth Mask:

| Bộ dữ liệu | Quy mô đánh giá | Mô hình / Phương pháp | Slice Mean Dice | Median Dice | Case Mean Dice | Slice Mean IoU | Mean HD95 (px) | Mốc trần tham chiếu (Upper-bound) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **BraTS2020 (DS4)** | 3.313 lát cắt (55 ca 3D) | **Box-fill Baseline** | $0.665 \pm 0.143$ | $0.695$ | $0.664 \pm 0.086$ | $0.513 \pm 0.147$ | 24.80 px | Điểm chuẩn thuần túy hình học hộp |
| **BraTS2020 (DS4)** | 3.313 lát cắt (55 ca 3D) | **SAM gốc (`vit_b`)** | $0.787 \pm 0.189$ | $0.844$ | $0.777 \pm 0.119$ | $0.680 \pm 0.208$ | 16.45 px | SAM-ViT-B nguyên bản |
| **BraTS2020 (DS4)** | 3.313 lát cắt (55 ca 3D) | **MedSAM (`vit_b`)** | **$0.788 \pm 0.171$** | **$0.848$** | **$0.782 \pm 0.087$** | **$0.678 \pm 0.194$** | **13.12 px** | **★ Upper-bound chuẩn (MedSAM tốt nhất)** |
| **indk214 (DS2)** | 2.192 lát cắt u | **Box-fill Baseline** | $0.824 \pm 0.073$ | $0.844$ | $0.824 \pm 0.073$ | $0.707 \pm 0.098$ | 18.20 px | Điểm chuẩn thuần túy hình học hộp |
| **indk214 (DS2)** | 2.192 lát cắt u | **SAM gốc (`vit_b`)** | **$0.848 \pm 0.115$** | **$0.883$** | **$0.848 \pm 0.115$** | **$0.751 \pm 0.150$** | 11.80 px | **★ Upper-bound chuẩn (SAM gốc nhỉnh hơn)** |
| **indk214 (DS2)** | 2.192 lát cắt u | **MedSAM (`vit_b`)** | $0.845 \pm 0.099$ | $0.869$ | $0.845 \pm 0.099$ | $0.743 \pm 0.137$ | 10.95 px | MedSAM thích nghi miền y khoa |
| **BRISC 2025 (DS3)** | 50 lát cắt test đa bình diện | **MedSAM — Glioma** | $85.64 \pm 4.20\%$ | $86.10\%$ | $85.64\%$ | $76.06 \pm 5.10\%$ | 12.27 px | U thần kinh đệm có biên mờ |
| **BRISC 2025 (DS3)** | 50 lát cắt test đa bình diện | **MedSAM — Meningioma**| **$92.15 \pm 2.80\%$** | **$92.80\%$** | **$92.15\%$** | **$86.04 \pm 3.60\%$** | **7.69 px** | U màng não bờ rõ rệt |
| **BRISC 2025 (DS3)** | 50 lát cắt test đa bình diện | **MedSAM — Pituitary** | $90.53 \pm 3.10\%$ | $91.20\%$ | $90.53\%$ | $82.92 \pm 4.20\%$ | 5.40 px | U tuyến yên đáy sọ bờ đều |
| **BRISC 2025 (DS3)** | 50 lát cắt test đa bình diện | **MedSAM Toàn bộ Test**| **$89.64 \pm 3.45\%$** | **$90.40\%$** | **$89.64\%$** | **$81.99 \pm 4.15\%$** | **8.29 px** | **★ Upper-bound chuẩn BRISC 2025** |

> **Phân tích theo độ mờ ranh giới u (Sobel Edge Sharpness Tertiles — Lâm Gia Bảo):**
> - **Nhóm T1 (U có biên mờ nhất, thâm nhiễm phù nề):** MedSAM vượt trội SAM gốc rõ nét!
>   * *BraTS2020:* MedSAM đạt Dice $0.757$ vs SAM gốc $0.739$ ($\Delta = +0.018$, CI 95% Bootstrap: $[+0.001, +0.036]$, kiểm định Wilcoxon theo ca $p = 0.038$, tỷ lệ ca MedSAM thắng đạt $54.5\%$).
>   * *indk214:* MedSAM đạt Dice $0.816$ vs SAM gốc $0.798$ ($\Delta = +0.018$, CI 95%: $[+0.009, +0.027]$).
> - **Nhóm T3 (U có bờ sắc nét, xơ cứng):** SAM gốc đạt Dice cao hơn ($0.824$ vs $0.798$ trên BraTS và $0.901$ vs $0.878$ trên indk214) do dữ liệu huấn luyện SA-1B tối ưu hóa theo các cạnh tương phản cao.

---

### 3. Thất bại có hệ thống của SAM Everything Mode (Prompt-less Automatic Segmentation)
Thực nghiệm kiểm thử trên 50 ảnh MRI thật (25 ảnh BRISC 2025 + 25 ảnh Dataset 1 Navoneel) của Nguyễn Thành Hải:

| Đặc trưng / Chỉ số thống kê đo đạc | Toàn bộ 50 ảnh kiểm thử | Tập BRISC 2025 (n = 25) | Tập Dataset 1 Navoneel (n = 25) | Ý nghĩa lâm sàng & chẩn đoán |
|:---|:---:|:---:|:---:|:---|
| **Số lượng Mask trung bình / lát cắt** | **$4.16 \pm 2.55$ masks** | $5.08 \pm 2.87$ masks | $3.24 \pm 1.83$ masks | Sinh phân mảnh lộn xộn các vùng không phải tổn thương |
| **Tỷ lệ Mask diện tích lớn (>5% lát cắt)** | **$54.52\%$** | $45.37\%$ | $63.67\%$ | Đa số mask bao trùm các cấu trúc giải phẫu lớn ngoài u |
| **Tỷ lệ lát cắt bị mask lớn nhất chiếm giữ** | **$50.29\%$** (cực đại **$84.56\%$**) | $48.21\%$ (max $79.34\%$) | $52.37\%$ (max $84.56\%$) | **Bao trọn toàn bộ vòm xương sọ và bán cầu đại não lành** |
| **Tỷ lệ Mask nằm hoàn toàn ngoài u** | **$89.77\%$** | $88.42\%$ | $91.12\%$ | Báo động giả nghiêm trọng, bắt nhầm xương sọ và da đầu |
| **Dice của Mask khớp nhất (Best-Mask)** | **$43.61 \pm 14.20\%$** | $45.18 \pm 12.80\%$ | $42.04 \pm 15.10\%$ | Thấp hơn rất nhiều so với MedSAM có prompt ($89.64\%$) |
| **Thời gian thực thi trung bình / ảnh** | **$4.82\text{s}$ / ảnh** | $4.95\text{s}$ / ảnh | $4.68\text{s}$ / ảnh | Lãng phí tính toán do phải quét lưới điểm $8 \times 8 = 64$ điểm |

---

## III. MINH HỌA TRỰC QUAN LỖI PHÂN VÙNG TRÀN HỘP SỌ VÀ GIẢI PHÁP
Hình ảnh đối sánh trực quan đã được xuất bản tại:
`NguyenPhuocHoangQuan_tuan2/image/fig_sam_skull_leakage_and_medsam_comparison.png`

![Lỗi tràn sọ của SAM và sự vượt trội của MedSAM](file:///d:/VNU-HCMUS/ttnt/alive_idiots/Tuần%202/NguyenPhuocHoangQuan_tuan2/image/fig_sam_skull_leakage_and_medsam_comparison.png)

### Cơ chế gây lỗi của SAM nguyên bản:
1. **SAM Everything Mode:** Không có bất kỳ tín hiệu vị trí ban đầu nào, SAM coi các ranh giới tự nhiên giữa mô não và xương sọ (vành xương sọ vôi hóa sáng trắng trên T1-ce) là biên của một đối tượng lớn, dẫn đến việc phân vùng bao phủ toàn bộ vòng đầu (chiếm tới $84.56\%$ diện tích lát cắt).
2. **SAM Box Mode (kém thích nghi u biên mờ):** Khi nhận Bounding Box, SAM nguyên bản có xu hướng tối ưu hóa đường cắt theo viền có độ tương phản cường độ pixel cao nhất. Đối với u thần kinh đệm (Glioma) có ranh giới thâm nhiễm mờ, SAM bị "lem" qua biên u và bám vào vỏ xương sọ kế cận.
3. **Ưu thế của MedSAM:** Nhờ được tinh chỉnh trên hơn 1,57 triệu cặp ảnh - mặt nạ y tế, MedSAM hiểu rõ ranh giới bệnh lý của u não và loại bỏ hoàn toàn viền sọ, đạt Dice tới $89.64\% - 92.15\%$.

---

## IV. KẾT LUẬN VÀ BÀN GIAO TUẦN 2
1. **Đã chuẩn hóa và nhúng thành công toàn bộ 3 bảng điểm thực nghiệm lớn** (Bảng 7.1, 7.2, 7.3) cùng hình ảnh minh họa Hình 7.1 vào báo cáo Word chính thức `baocaoCK_AliveIdiots_tuan2.docx`.
2. **Cung cấp cơ sở thực nghiệm quyết định:**
   - Chứng minh Linear Probing bị giới hạn Recall $< 88\%$ và Full Fine-tuning bị Overfitting nặng $\Rightarrow$ **Động lực xây dựng ViT-LoRA trong Tuần 3**.
   - Chứng minh SAM Everything Mode thất bại và MedSAM cần Box Prompt chuẩn $\Rightarrow$ **Động lực xây dựng Bộ tự sinh lời nhắc Auto-Prompt Generator trong Tuần 3**.
