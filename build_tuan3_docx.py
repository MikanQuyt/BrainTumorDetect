"""
Kịch bản tạo baocaoCK_AliveIdiots_tuan3.docx cho NguyenPhuocHoangQuan_tuan3:
Kế thừa từ baocaoCK_AliveIdiots_tuan2.docx và bổ sung toàn bộ kết quả Tuần 3:
  1. Hình ảnh và thuyết minh chi tiết khối LoRA Multi-Head Self-Attention (LoRA-MHSA)
  2. Hình ảnh và thuyết minh quy trình tự sinh tín hiệu nhắc lai (Hybrid Prompt Pipeline)
  3. Lưới ảnh đối chiếu 4 giai đoạn: Ảnh gốc -> Attention Rollout -> Fused LayerCAM lọc sọ -> Khung Bounding Box tự sinh
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC_DOC = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan2\baocaoCK_AliveIdiots_tuan2.docx")
DST_DOC = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan3\baocaoCK_AliveIdiots_tuan3.docx")

IMG_LORA = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan3\image\fig_lora_mhsa_architecture.png")
IMG_PROMPT = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan3\image\fig_hybrid_prompt_pipeline.png")
IMG_SEQ = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan3\image\fig_sequential_attention_rollout_layercam_autobox.png")

doc = docx.Document(SRC_DOC)

# Tìm vị trí mục "Cơ chế tích hợp LoRA vào khối Attention của EfficientViT" hoặc "Cơ chế tự động cấp tín hiệu nhắc"
idx_lora = None
idx_prompt = None
idx_exp_lora = None

for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if "Cơ chế tích hợp LoRA vào khối Attention" in t:
        idx_lora = i
    elif "Cơ chế tự động cấp tín hiệu nhắc" in t:
        idx_prompt = i
    elif "Kết quả huấn luyện và đánh giá hệ thống đề xuất EfficientViT-SAM + LoRA" in t:
        idx_exp_lora = i

print(f"Indices: lora={idx_lora}, prompt={idx_prompt}, exp_lora={idx_exp_lora}")

# 1. Bổ sung ảnh LoRA-MHSA vào phần Lý thuyết LoRA
if idx_lora is not None and IMG_LORA.exists():
    target_p = doc.paragraphs[idx_lora + 1]
    p_img = target_p.insert_paragraph_before()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(IMG_LORA), width=Inches(6.4))

    p_cap = target_p.insert_paragraph_before("Hình 2.8: Sơ đồ kiến trúc chi tiết của khối LoRA Multi-Head Self-Attention (LoRA-MHSA) với ma trận phân rã hạng thấp r = 8, α = 16 trên các tầng Query và Value.")
    p_cap.style = 'Hinh' if 'Hinh' in [s.name for s in doc.styles] else 'Normal'
    p_cap.paragraph_format.space_after = Pt(8)
    for r in p_cap.runs:
        r.font.bold = True
        r.font.italic = True
        r.font.size = Pt(8.5)
        r.font.name = "Times New Roman"

# 2. Bổ sung ảnh Hybrid Prompt Pipeline vào phần Auto-Prompting
if idx_prompt is not None and IMG_PROMPT.exists():
    target_p = doc.paragraphs[idx_prompt + 1]
    p_img = target_p.insert_paragraph_before()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(IMG_PROMPT), width=Inches(6.4))

    p_cap = target_p.insert_paragraph_before("Hình 6.4: Sơ đồ luồng trích xuất Lời nhắc Lai (Hybrid Prompt Extraction Pipeline) kết hợp nhịp nhàng giữa Bản đồ nhiệt Giám sát yếu (Attention Rollout + LayerCAM + Lọc viền sọ) và Nhánh Hồi quy Hộp ViTDet SFP Box Head.")
    p_cap.style = 'Hinh' if 'Hinh' in [s.name for s in doc.styles] else 'Normal'
    p_cap.paragraph_format.space_after = Pt(8)
    for r in p_cap.runs:
        r.font.bold = True
        r.font.italic = True
        r.font.size = Pt(8.5)
        r.font.name = "Times New Roman"

# 3. Bổ sung Lưới ảnh đối chiếu 4 giai đoạn vào phần Thực nghiệm Tuần 3
if idx_exp_lora is not None and IMG_SEQ.exists():
    target_p = doc.paragraphs[idx_exp_lora + 1]
    p_img = target_p.insert_paragraph_before()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(IMG_SEQ), width=Inches(6.4))

    p_cap = target_p.insert_paragraph_before("Hình 7.2: Trực quan hóa bộ ảnh đối chiếu tuần tự 4 giai đoạn trên các ca bệnh thực tế: Ảnh MRI gốc → Bản đồ nhiệt Attention Rollout → Bản đồ nhiệt Fused LayerCAM đã lọc viền hộp sọ (Parenchyma Mask) → Khung Bounding Box tự sinh.")
    p_cap.style = 'Hinh' if 'Hinh' in [s.name for s in doc.styles] else 'Normal'
    p_cap.paragraph_format.space_after = Pt(8)
    for r in p_cap.runs:
        r.font.bold = True
        r.font.italic = True
        r.font.size = Pt(8.5)
        r.font.name = "Times New Roman"

doc.save(DST_DOC)
print(f"Tạo thành công {DST_DOC} với kích thước {DST_DOC.stat().st_size} bytes!")
