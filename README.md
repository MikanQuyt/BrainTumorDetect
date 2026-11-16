# Thư mục Báo cáo & Sản phẩm Kỹ thuật Tuần 3 — Nguyễn Phước Hoàng Quân

Thư mục này lưu trữ toàn bộ mã nguồn, sơ đồ thiết kế kiến trúc và báo cáo thực nghiệm của **Nguyễn Phước Hoàng Quân** trong khuôn khổ nghiên cứu Tuần 3 của đề tài:
**"Hệ thống Phát hiện & Phân vùng Khối u Não trên Ảnh MRI kết hợp ViT-LoRA và MedSAM"**  
Nhóm thực hiện: **Alive Idiots** (Bảo, Hải, Huy, Quyết, Quân).

---

## 1. Cấu trúc Thư mục

```text
NguyenPhuocHoangQuan_tuan3/
├── generate_hoang_quan_visuals.py      # Kịch bản Python tự động tạo toàn bộ sơ đồ & hình ảnh trực quan hóa
├── README.md                           # Giới thiệu tổng quan cấu trúc thư mục
├── image/                              # Thư mục lưu trữ toàn bộ hình ảnh độ phân giải cao (200-220 DPI)
│   ├── fig_lora_mhsa_architecture.png                      # [Tuần 3 - Nhiệm vụ 1] Sơ đồ kiến trúc khối LoRA-MHSA
│   ├── fig_hybrid_prompt_pipeline.png                      # [Tuần 3 - Nhiệm vụ 1] Sơ đồ luồng trích xuất Hybrid Prompt
│   └── fig_sequential_attention_rollout_layercam_autobox.png # [Tuần 3 - Nhiệm vụ 2] Trực quan hóa đối chiếu 4 giai đoạn
└── reports/                            # Báo cáo kỹ thuật chi tiết
    └── BAO_CAO_TUAN3_HOANG_QUAN.md     # Thuyết minh khoa học, lý thuyết toán học và phân tích định lượng
```

---

## 2. Danh mục Sản phẩm & Hình ảnh Đã Thực hiện

### 1. Sơ đồ Kiến trúc Chi tiết Khối LoRA Multi-Head Self-Attention
- **Tập tin:** `image/fig_lora_mhsa_architecture.png`
- **Mô tả:** Làm rõ cấu trúc phân rã ma trận hạng thấp $W = W_0 + \frac{\alpha}{r} (B \cdot A)$ trên ma trận Query ($Q$) và Value ($V$) của 12 khối Self-Attention; phân tách $h=12$ đầu chú ý; chứng minh tham số khả huấn chỉ chiếm $0.34\%$ backbone và triệt tiêu $100\%$ độ trễ phụ trội khi suy luận lâm sàng.

### 2. Sơ đồ Luồng Trích xuất Lời nhắc Lai (Hybrid Prompt Extraction Pipeline)
- **Tập tin:** `image/fig_hybrid_prompt_pipeline.png`
- **Mô tả:** Thể hiện luồng xử lý end-to-end từ ảnh MRI đầu vào ($224 \times 224$) qua ViT-LoRA đến hai cơ chế song song:
  * **Cơ chế (1) — Giám sát yếu:** Attention Rollout + LayerCAM + Lọc viền hộp sọ (Brain Parenchyma Mask Erosion 9.5%) $\to$ Peak Positive Point $(x^+, y^+)$ [★] & Negative Point $(x^-, y^-)$ [✕].
  * **Cơ chế (2) — Hồi quy hộp có giám sát:** ViTDet Simple Feature Pyramid (SFP P3, P4) + Spatial Attention Pooling $\to$ Bounding Box tự sinh mở rộng an toàn $+8\%$ margin.
  * Tích hợp thành **Hybrid Prompt chuẩn $1024 \times 1024$** cấp cho MedSAM Mask Decoder.

### 3. Trực quan hóa Bộ ảnh Đối chiếu Tuần tự 4 Giai đoạn
- **Tập tin:** `image/fig_sequential_attention_rollout_layercam_autobox.png`
- **Mô tả:** Lưới ảnh đối chiếu 4 giai đoạn trên các ca bệnh thực tế:
  $$\text{Ảnh MRI gốc} \longrightarrow \text{Bản đồ nhiệt Attention Rollout} \longrightarrow \text{Bản đồ nhiệt Fused LayerCAM đã lọc sọ} \longrightarrow \text{Khung Bounding Box tự sinh}$$

---

## 3. Hướng dẫn Tái hiện Kết quả

Chạy kịch bản bằng môi trường Python đã cài đặt `matplotlib`, `numpy`, `pillow`, `scipy`:
```powershell
python generate_hoang_quan_visuals.py
```
Toàn bộ hình ảnh sẽ được tự động xuất ra thư mục `image/` và đồng bộ vào `dev/image/tuan3/`.
