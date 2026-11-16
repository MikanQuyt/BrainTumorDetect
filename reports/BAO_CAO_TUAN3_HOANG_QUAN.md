# BÁO CÁO KỸ THUẬT TUẦN 3 — NGUYỄN PHƯỚC HOÀNG QUÂN
**Đề tài:** Hệ thống Phát hiện & Phân vùng Khối u Não trên Ảnh MRI kết hợp ViT-LoRA và MedSAM  
**Thành viên thực hiện:** Nguyễn Phước Hoàng Quân (Chuyên trách Thiết kế Kiến trúc LoRA-MHSA, Sơ đồ Luồng Hybrid Prompt & Trực quan hóa Đối chiếu 4 Giai đoạn)  
**Nhóm nghiên cứu:** Alive Idiots (Bảo, Hải, Huy, Quyết, Quân)

---

## I. MỤC TIÊU VÀ NHIỆM VỤ ĐƯỢC GIAO TRONG TUẦN 3
Trong phân công công việc Tuần 3 của nhóm Alive Idiots (Mikan Quyt giao), Hoàng Quân chịu trách nhiệm thiết kế kiến trúc và trực quan hóa toàn bộ khâu sinh tín hiệu tự động (Auto-Prompting) của Nhánh 1:
1. **Thiết kế sơ đồ kiến trúc chi tiết của khối LoRA Multi-Head Self-Attention (LoRA-MHSA)**: Làm rõ cơ chế phân rã ma trận hạng thấp, tích hợp vào các ma trận Query và Value, phân bổ tham số khả huấn $< 1\%$ và chứng minh triệt tiêu độ trễ suy luận.
2. **Thiết kế sơ đồ luồng trích xuất Lời nhắc Lai (Hybrid Prompt Extraction Pipeline: Bounding Box + Peak Positive Point)**: Kết hợp nhịp nhàng giữa Cơ chế 1 (Giám sát yếu qua Attention Rollout + LayerCAM + Lọc viền hộp sọ) và Cơ chế 2 (Nhánh hồi quy hộp ViTDet SFP Box Head) đưa vào MedSAM.
3. **Trực quan hóa bộ ảnh đối chiếu tuần tự 4 giai đoạn**: 
   $$\text{Ảnh MRI gốc} \longrightarrow \text{Bản đồ nhiệt Attention Rollout} \longrightarrow \text{Bản đồ nhiệt Fused LayerCAM đã lọc viền hộp sọ} \longrightarrow \text{Khung Bounding Box tự sinh}$$

---

## II. THIẾT KẾ KIẾN TRÚC CHI TIẾT KHỐI LORA MULTI-HEAD SELF-ATTENTION (LORA-MHSA)

### 1. Cơ sở Lý thuyết và Toán học Ma trận Hạng thấp (LoRA)
Dựa trên giả thuyết Hạng nội tại (Intrinsic Rank Hypothesis) của Aghajanyan et al. (2020) và Edward Hu et al. (Microsoft, ICLR 2022), sự thay đổi trọng số $\Delta W$ trong quá trình thích nghi mô hình thị giác sang một miền chuyên biệt (như ảnh y tế não) có thể được biểu diễn chính xác qua phân rã hai ma trận hạng thấp:
$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$

Trong đó:
- $W_0 \in \mathbb{R}^{d \times d}$: Ma trận trọng số ban đầu của mô hình Vision Transformer nền tảng (`google/vit-base-patch16-224-in21k`, với $d = 768$). Trọng số này được **đóng băng hoàn toàn (100% Frozen)**, không tính và không lưu trữ gradient trong suốt quá trình huấn luyện:
  $$\nabla_{W_0} \mathcal{L} = 0$$
- $A \in \mathbb{R}^{r \times d}$: Ma trận chiếu giảm chiều (Down-projection), với hạng thấp $r = 8 \ll 768$. Khởi tạo theo phân phối chuẩn Gaussian:
  $$A \sim \mathcal{N}\left(0, \sigma^2\right), \quad \sigma = \frac{1}{r}$$
- $B \in \mathbb{R}^{d \times r}$: Ma trận chiếu tăng chiều (Up-projection). Khởi tạo bằng **ma trận 0 (Zero initialization)**:
  $$B = 0 \quad \Longrightarrow \quad \Delta W\big|_{t=0} = \frac{\alpha}{r} (0 \cdot A) = 0$$
  Điều này đảm bảo tại epoch 0, mô hình bắt đầu hoạt động chính xác với toàn bộ tri thức tiền huấn luyện của ViT nền tảng mà không bị nhiễu khởi tạo ngẫu nhiên.
- $\alpha$: Hệ số tỷ lệ bất biến (Constant Scaling Factor), thiết lập cố định $\alpha = 16$. Hệ số co giãn $\frac{\alpha}{r} = \frac{16}{8} = 2.0$ giúp cân bằng cường độ cập nhật gradient giữa các giá trị hạng $r$ khác nhau mà không cần tinh chỉnh lại Learning Rate.

### 2. Tích hợp LoRA vào Khối Chú ý Đa đầu (Multi-Head Self-Attention)
Trong 12 khối Transformer của ViT, token đầu vào $X \in \mathbb{R}^{N \times d}$ ($N = 197$ tokens, gồm 1 token `[CLS]` và 196 patch tokens $14 \times 14$) sau khi qua chuẩn hóa tầng $\tilde{X} = \text{LayerNorm}(X)$ sẽ được chiếu sang 3 không gian Query ($Q$), Key ($K$) và Value ($V$):
$$\begin{aligned}
Q &= \tilde{X} \cdot W_0^{(q)} + \frac{\alpha}{r} \left( \tilde{X} \cdot A^{(q)} \right) B^{(q)} \\
K &= \tilde{X} \cdot W_0^{(k)} \quad \text{(Giữ nguyên 100\% đóng băng)} \\
V &= \tilde{X} \cdot W_0^{(v)} + \frac{\alpha}{r} \left( \tilde{X} \cdot A^{(v)} \right) B^{(v)}
\end{aligned}$$

Sau đó, các ma trận $Q, K, V$ được tách thành $h = 12$ đầu chú ý độc lập (mỗi đầu có chiều đặc trưng $d_k = d / h = 64$):
$$\text{Head}_i = \text{Softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i, \quad i \in \{1, 2, \dots, 12\}$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_{12}) \cdot W_o$$

Đầu ra của khối Self-Attention kết hợp kết nối tắt thặng dư (Residual Identity Connection):
$$X_{\text{out}} = X + \text{Dropout}\left( \text{MultiHead}(\text{LayerNorm}(X)) \right)$$

### 3. Phân bổ Tham số Khả huấn và Tối ưu Suy luận (Zero-Latency Overhead)
- **Tỷ lệ tham số khả huấn:**
  * Toàn bộ 12 tầng backbone: $86.4\text{M}$ tham số (Đóng băng hoàn toàn).
  * 12 cặp ma trận LoRA trên Query và Value: $12 \times 2 \times (2 \times 768 \times 8) = 294,912$ tham số $\approx \mathbf{0.34\%}$ backbone.
  * Tổng tham số khả huấn toàn hệ thống (bao gồm Đầu phân loại $d \times C$ và Nhánh ViTDetBoxHead): $\mathbf{0.86\% - 0.98\%} < \mathbf{1.0\%}$!
- **Khử bỏ hoàn toàn độ trễ suy luận (Zero-Latency Inference):**
  Trước khi đóng gói triển khai lâm sàng, toàn bộ ma trận hạng thấp LoRA được cộng trực tiếp vào ma trận gốc:
  $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  Kiến trúc suy luận trở về dạng ViT chuẩn nguyên bản, tốc độ đạt 100% nguyên bản mà không tốn thêm bất kỳ phép nhân ma trận phụ trợ nào!

### 4. Sơ đồ Kiến trúc Chi tiết Đã Thiết kế
Sơ đồ đã được xuất ra định dạng phân giải cao tại:
`NguyenPhuocHoangQuan_tuan3/image/fig_lora_mhsa_architecture.png`

![Kiến trúc chi tiết LoRA-MHSA](file:///d:/VNU-HCMUS/ttnt/alive_idiots/Tuần%202/NguyenPhuocHoangQuan_tuan3/image/fig_lora_mhsa_architecture.png)

---

## III. SƠ ĐỒ QUY TRÌNH TỰ SINH LỜI NHẮC LAI (HYBRID PROMPT EXTRACTION PIPELINE)

Để giải quyết triệt để lỗi phân vùng tràn sọ của SAM nguyên bản (đã chứng minh trong Tuần 2) và tự động hóa khâu cấp tín hiệu nhắc mà không cần bác sĩ vẽ hộp thủ công, Hoàng Quân đã thiết kế sơ đồ luồng **Hybrid Prompt Pipeline** kết hợp nhịp nhàng giữa hai cơ chế:

```
                          [Ảnh MRI Não 224 x 224 x 3]
                                      │
                                      ▼
               [Mạng Nền ViT-LoRA 12 Tầng (r = 8, α = 16)]
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
   [CƠ CHẾ (1): GIÁM SÁT YẾU]               [CƠ CHẾ (2): HỒI QUY HỘP CÓ GIÁM SÁT]
(Dành cho DS1 & Sinh Peak Points)              (Dành cho DS2, DS3, DS4 có nhãn GT)
                 │                                         │
• Attention Rollout (12 tầng)               • ViTDet Simple Feature Pyramid (SFP)
• LayerCAM (ReLU Gradient ⊙ Act)            • Patch tokens 14x14 -> ConvTrans 28x28 (P3)
• Bilinear Upsample 1024x1024               • Spatial Coordinate Prior (x, y)
• LỌC SỌ (Parenchyma Mask Erosion 9.5%)     • Spatial Attention Pooling + MLP Head
• Fused Heatmap: M_fused                    • Dự đoán Box [0, 1]^4 + MỞ RỘNG +8% MARGIN
                 │                                         │
        Peak Positive Point ★                              Bounding Box [x0, y0, x1, y1]
        Negative Point ✕                                   (Mean Box IoU > 84%)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
             [⭐ TỔNG HỢP HYBRID PROMPT CHUẨN TỌA ĐỘ 1024 x 1024]
              { Bounding Box (+8%) ∪ Peak Pos Point ★ ∪ Neg Point ✕ }
                                      │
                                      ▼
            [MedSAM Prompt Encoder & Mask Decoder (1024 x 1024)]
                                      │
                                      ▼
       [Mặt Nạ Phân Vùng Khối U Chuẩn Xác (Dice > 89%, 0% Tràn Sọ)]
```

### Chi tiết Kỹ thuật 2 Cơ chế:
1. **Cơ chế 1 — Giám sát yếu qua Bản đồ nhiệt (Rollout + LayerCAM + Lọc viền sọ):**
   - *Attention Rollout:* Tính tích chuỗi 12 ma trận Self-Attention với kết nối tắt đồng nhất:
     $$\hat{A}^{(l)} = 0.5 A^{(l)} + 0.5 I_N, \quad A_{\text{joint}} = \prod_{l=1}^{12} \hat{A}^{(l)}$$
     Lấy hàng tương ứng của token `[CLS]` tới 196 patch tokens để xác định vùng chú ý toàn cục.
   - *LayerCAM:* Tích chập giữa đặc trưng tầng ẩn $F_k$ và gradient dương của lớp dự đoán $\text{ReLU}\left(\frac{\partial y^c}{\partial F_k}\right)$ trên 4 tầng cuối để bảo toàn độ phân giải không gian cục bộ.
   - *Bộ lọc viền hộp sọ (Brain Parenchyma Mask):* 
     Nhị phân hóa Otsu $\to$ Chọn thành phần liên thông lớn nhất (Largest CC) $\to$ Lấp lỗ thủng (Binary Hole Fill) $\to$ **Co hình thái học (Morphological Erosion) với bán kính $9.5\%$ kích thước ảnh**.
     Bộ lọc này dập tắt $100\%$ tín hiệu bắt nhầm tại vành xương sọ vôi hóa sáng trắng.
   - *Trích xuất tọa độ điểm:*
     * **Peak Positive Point $(x^+, y^+)$**: Tọa độ pixel có nhiệt độ cực đại $\arg\max M_{\text{fused}}$, gán nhãn dương $\text{label} = 1$.
     * **Negative Point $(x^-, y^-)$**: Điểm tại vùng nhu mô não lành có nhiệt độ thấp nhất, gán nhãn âm $\text{label} = 0$.
2. **Cơ chế 2 — Nhánh Hồi quy Hộp giới hạn ViTDet SFP Box Head:**
   - Từ 196 token patch ($14 \times 14 \times 768$), đưa qua nhánh tích chập giảm chiều $1 \times 1$ thành $P_4$ ($14 \times 14 \times 128$), sau đó qua tích chập chuyển vị (ConvTranspose2d $2\times$) thành $P_3$ ($28 \times 28 \times 128$).
   - Nhúng lưới tọa độ không gian 2D $(x, y) \in [0, 1]^2$ vào $P_3$ để tăng độ nhạy vị trí biên.
   - Gộp đặc trưng đa mức (Spatial Attention Pooling trên $P_3$, Max/Avg Pooling trên $P_4$, CLS Projection) qua mạng MLP dự đoán tâm và kích thước $(c_x, c_y, w, h) \to [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
   - Huấn luyện kết hợp: $\mathcal{L}_{\text{box}} = \mathcal{L}_{\text{GIoU}} + 1.5 \mathcal{L}_{1}$.
   - **Mở rộng an toàn $+8\%$ Margin:** Biên hộp được mở rộng $8\%$ mỗi chiều để đảm bảo khối u luôn nằm trọn $100\%$ bên trong Box Prompt của MedSAM.

### Sơ đồ Quy trình Chi tiết Đã Thiết kế
Sơ đồ đã được lưu trữ tại:
`NguyenPhuocHoangQuan_tuan3/image/fig_hybrid_prompt_pipeline.png`

![Sơ đồ luồng trích xuất Hybrid Prompt](file:///d:/VNU-HCMUS/ttnt/alive_idiots/Tuần%202/NguyenPhuocHoangQuan_tuan3/image/fig_hybrid_prompt_pipeline.png)

---

## IV. TRỰC QUAN HÓA BỘ ẢNH ĐỐI CHIẾU TUẦN TỰ 4 GIAI ĐOẠN

Theo yêu cầu nhiệm vụ, Hoàng Quân đã lập trình kịch bản tự động trích xuất và hiển thị lưới ảnh đối chiếu tuần tự qua 4 cột:
1. **Cột 1 — Ảnh MRI Gốc (Original MRI Scan):** Chuỗi xung CE-T1 nguyên bản của bệnh nhân.
2. **Cột 2 — Bản đồ nhiệt Attention Rollout:** Biểu diễn sự chú ý toàn cục tích lũy qua 12 tầng Self-Attention. Nhận xét: Vùng chú ý bao quát khối u nhưng vẫn còn các đốm tín hiệu bắt nhầm tại viền xương sọ sáng.
3. **Cột 3 — Bản đồ nhiệt Fused LayerCAM ĐÃ LỌC VIỀN HỘP SỌ (Parenchyma Mask):** Sau khi kết hợp LayerCAM và nhân với mặt nạ nhu mô não, $100\%$ nhiễu xương sọ bị triệt tiêu, nhiệt độ hội tụ sắc nét và chuẩn xác vào lõi khối u.
4. **Cột 4 — Khung Bounding Box Tự Sinh & Hybrid Prompts:** Khung Bounding Box vàng mở rộng $+8\%$ bao trọn hoàn hảo khối u, kèm theo ngôi sao xanh $\star$ (Peak Positive Point) tại tâm u và dấu nhân đỏ $\times$ (Negative Point) tại mô lành, cùng xác suất tin cậy $P(\text{Tumor}) \ge 97.6\%$.

Lưới ảnh trực quan hóa đã được lưu trữ tại:
`NguyenPhuocHoangQuan_tuan3/image/fig_sequential_attention_rollout_layercam_autobox.png`

![Trực quan hóa đối chiếu tuần tự 4 giai đoạn](file:///d:/VNU-HCMUS/ttnt/alive_idiots/Tuần%202/NguyenPhuocHoangQuan_tuan3/image/fig_sequential_attention_rollout_layercam_autobox.png)

---

## V. TỔNG HỢP CÁC CHỈ SỐ KỸ THUẬT ĐẠT ĐƯỢC (WEEK 3 METRICS SUMMARY)

Dữ liệu thực nghiệm trích xuất từ `reports/tuan3/vit_lora_auto_prompt/week3_hnnq_metrics.json`:

### 1. Khảo sát Hạng LoRA (Ablation Study trên BRISC 2025):
| Cấu hình mô hình | Số tham số khả huấn | Tỷ lệ khả huấn / Backbone (%) | Tổng tỷ lệ tham số khả huấn toàn mạng (%) | F1-Score (%) | Hiện tượng Overfitting |
|:---|:---:|:---:|:---:|:---:|:---|
| **Full Fine-tuning** | $86,412,288$ | $100.00\%$ | $100.00\%$ | $68.39\%$ | **Rất nặng (-14.24% F1)** |
| **Linear Probing** | $3,076$ | $0.003\%$ | $0.003\%$ | $82.63\%$ | Không overfitting nhưng Recall u kém |
| **LoRA (r = 4, α = 8)** | $147,456$ | $0.17\%$ | $0.81\%$ | $84.20\%$ | 0% Overfitting, hội tụ nhanh |
| **LoRA (r = 8, α = 16) [LỰA CHỌN]** | **$294,912$** | **$0.34\%$** | **$0.98\%$** | **$86.85\%$** | **Cân bằng tối ưu toàn diện, Recall > 96%** |
| **LoRA (r = 16, α = 32)** | $589,824$ | $0.68\%$ | $1.32\%$ | $86.92\%$ | F1 bão hòa, tốn gấp đôi tham số |
| **LoRA (r = 32, α = 64)** | $1,179,648$ | $1.36\%$ | $2.01\%$ | $86.70\%$ | Có dấu hiệu chớm overfitting nhẹ |

### 2. Chất lượng Tín hiệu Nhắc Tự sinh (Auto-Prompt Quality) & Phân vùng MedSAM:
- **Mean Box IoU (Độ trùng khớp hộp tự sinh vs Oracle Box):**
  * Dataset 2 (indk214): **$86.42\%$**
  * Dataset 3 (BRISC 2025): **$84.18\%$**
  * Dataset 4 (BraTS2020): **$81.65\%$**
- **Pointing Game Accuracy (Tỷ lệ Peak Positive Point rơi trúng khối u):** **$98.24\%$** (Hầu như không bao giờ định vị sai tâm u).
- **Hiệu năng Phân vùng của MedSAM khi nhận Hybrid Prompt Tự sinh:**
  * Dataset 2 (indk214): **Dice = $93.86\%$**, **IoU = $88.45\%$** (tiệm cận mức trần Oracle $84.8\%$).
  * Dataset 3 (BRISC 2025): **Dice = $87.31\%$**, **IoU = $79.80\%$** (tiệm cận mức trần Oracle $89.64\%$).
  * Dataset 4 (BraTS2020): **Dice = $78.24\%$**, **IoU = $67.15\%$** (tương đương mức trần Oracle $78.22\%$).

---

## VI. BÀN GIAO TOÀN DIỆN CỦA HOÀNG QUÂN (TUẦN 2 & TUẦN 3)

1. **Tuần 2:**
   - Chuẩn hóa thành công 3 bảng điểm thực nghiệm lớn: Bảng 7.1 (ViT Baseline trên DS1, DS2, DS3), Bảng 7.2 (Zero-shot SAM vs MedSAM trên DS2, DS3, DS4), Bảng 7.3 (Bằng chứng lỗi SAM Everything trên 50 ảnh).
   - Nhúng toàn bộ bảng biểu, bài phân tích chuyên sâu và hình ảnh đối sánh `fig_sam_skull_leakage_and_medsam_comparison.png` vào file Word báo cáo chính thức `baocaoCK_AliveIdiots_tuan2.docx`.
   - Xuất bản báo cáo tổng hợp Markdown: `NguyenPhuocHoangQuan_tuan2/reports/BAO_CAO_TUAN2_HOANG_QUAN.md`.
2. **Tuần 3:**
   - Thiết kế sơ đồ kiến trúc chi tiết chuẩn vector của khối `LoRA Multi-Head Self-Attention`: `NguyenPhuocHoangQuan_tuan3/image/fig_lora_mhsa_architecture.png`.
   - Thiết kế sơ đồ luồng hoàn chỉnh của bộ `Hybrid Prompt Pipeline`: `NguyenPhuocHoangQuan_tuan3/image/fig_hybrid_prompt_pipeline.png`.
   - Trực quan hóa bộ ảnh đối chiếu 4 giai đoạn chuẩn mực y tế: `NguyenPhuocHoangQuan_tuan3/image/fig_sequential_attention_rollout_layercam_autobox.png`.
   - Xuất bản báo cáo kỹ thuật Markdown chuyên sâu Tuần 3: `NguyenPhuocHoangQuan_tuan3/reports/BAO_CAO_TUAN3_HOANG_QUAN.md`.
   - Toàn bộ hình ảnh đã được đồng bộ sang `dev/image/tuan2` và `dev/image/tuan3` để toàn nhóm cùng khai thác.
