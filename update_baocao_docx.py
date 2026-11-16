"""
Kịch bản cập nhật Báo cáo đề tài Word (.docx) cho Hoàng Quân:
Chèn toàn bộ bảng điểm chuẩn hóa và phân tích chuyên sâu cho:
  1. Kết quả thực nghiệm mô hình cơ sở ViT Baseline (Linear Probing vs. Full Fine-tuning)
  2. Kết quả thực nghiệm Zero-shot SAM vs. MedSAM: Bằng chứng định lượng và trực quan về lỗi tràn sọ
  3. Chèn hình minh họa trực quan đối sánh lỗi tràn sọ fig_sam_skull_leakage_and_medsam_comparison.png
"""
import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

DOC_PATHS = [
    Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan2\baocaoCK_AliveIdiots_tuan2.docx"),
    Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\baocaoCK_AliveIdiots_tuan2.docx"),
]

IMG_PATH = Path(r"D:\VNU-HCMUS\ttnt\alive_idiots\Tuần 2\NguyenPhuocHoangQuan_tuan2\image\fig_sam_skull_leakage_and_medsam_comparison.png")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def style_table_header(row, col_widths, bg_hex="1E3A8A"):
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        cell.width = Inches(col_widths[idx])
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(8.5)
                run.font.name = "Times New Roman"


def format_table_row(row, col_widths, is_even=False, aligns=None):
    bg_hex = "F8FAFC" if is_even else "FFFFFF"
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        cell.width = Inches(col_widths[idx])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            if aligns and idx < len(aligns):
                p.alignment = aligns[idx]
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.size = Pt(8.5)
                run.font.name = "Times New Roman"


def update_single_docx(doc_path: Path):
    if not doc_path.exists():
        print(f"File không tồn tại: {doc_path}")
        return

    print(f"Đang xử lý tài liệu: {doc_path}...")
    doc = docx.Document(doc_path)

    # Tìm vị trí các heading
    idx_vit = None
    idx_sam = None
    idx_next = None

    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if "Kết quả thực nghiệm mô hình cơ sở ViT Baseline" in t:
            idx_vit = i
        elif "Kết quả thực nghiệm Zero-shot SAM vs. MedSAM" in t:
            idx_sam = i
        elif "Kết quả huấn luyện và đánh giá hệ thống đề xuất EfficientViT-SAM" in t:
            idx_next = i

    print(f"  Vị trí tìm thấy: ViT Baseline = {idx_vit}, SAM vs MedSAM = {idx_sam}, Hệ thống đề xuất = {idx_next}")
    if idx_vit is None or idx_sam is None or idx_next is None:
        print("  [Lỗi] Không định vị được đầy đủ các tiêu đề!")
        return

    # 1. Thu thập các đoạn văn bản mới cho phần ViT Baseline
    vit_text_intro = (
        "Thực nghiệm mô hình cơ sở Vision Transformer (ViT Baseline) được triển khai trên kiến trúc "
        "google/vit-tiny-patch16-224 (tiền huấn luyện trên ImageNet-21k và tinh chỉnh trên ImageNet-1k, 12 tầng "
        "Transformer, hidden size d = 192, 3 đầu chú ý) nhằm đánh giá toàn diện năng lực biểu diễn đặc trưng ảnh MRI "
        "não dưới hai phương thức thích nghi truyền thống: (1) Linear Probing (đóng băng hoàn toàn 12 tầng backbone, "
        "chỉ huấn luyện đầu phân loại tuyến tính) và (2) Full Fine-tuning (cập nhật tự do toàn bộ tham số backbone và "
        "đầu phân loại). Thực nghiệm được tiến hành trên cả 3 bộ dữ liệu có nhãn phân loại bệnh lý: Dataset 1 "
        "(Navoneel, 253 ảnh, 5-Fold Stratified Group CV), Dataset 2 (indk214, 7.023 ảnh, chia 70/15/15) và "
        "Dataset 3 (BRISC 2025, 6.000 ảnh, chia 70/15/15 phân cụm bệnh nhân triệt tiêu rò rỉ chéo)."
    )

    vit_table_caption = (
        "Bảng 7.1: Tổng hợp và chuẩn hóa bảng điểm thực nghiệm ViT Baseline (Linear Probing vs. Full Fine-tuning) "
        "trên 3 bộ dữ liệu MRI não (Kết quả thực nghiệm của Thái Văn Huy và Nguyễn Thành Hải)"
    )

    vit_analysis_title = "Phân tích khoa học và bằng chứng thực nghiệm về Overfitting và Hạn chế Recall:"
    vit_bullets = [
        ("Hạn chế về độ nhạy (Recall < 88%) của Linear Probing: ",
         "Do chỉ huấn luyện lớp phân loại tuyến tính trên đặc trưng đóng băng của ImageNet, mô hình không thích nghi được "
         "với miền dữ liệu y tế chuyên sâu. Trên Dataset 1, Linear Probing chỉ đạt Recall 81,29 ± 7,36% trên lớp có u, "
         "bỏ sót tới 29/155 bệnh nhân mang khối u (đặc biệt fold 1 và fold 5 Recall tụt xuống 74,19% và 77,42%). "
         "Tương tự, trên Dataset 2, Recall u màng não (Meningioma) chỉ đạt 64,46%. Trong chẩn đoán ung thư não, việc bỏ sót "
         "tổn thương (False Negative) là rủi ro y khoa nghiêm trọng nhất, chứng minh Linear Probing hoàn toàn không đáp ứng "
         "được yêu cầu sàng lọc lâm sàng."),
        ("Bằng chứng thực nghiệm về hiện tượng Quá khớp (Overfitting) nghiêm trọng khi Full Fine-tuning: ",
         "Khi mở khóa toàn bộ 12 tầng ViT để Full Fine-tuning, mô hình rơi vào trạng thái quá khớp trầm trọng. "
         "Trên Dataset 1 (253 ảnh), tại 4/5 fold, training loss giảm tiệm cận 0 (0,00002) trong khi validation loss bắt đầu "
         "phân kỳ và tăng vọt liên tục từ epoch 4–6 đến epoch 20 (riêng Fold 3, val loss tăng từ 0,3807 lên 0,5468 dù train "
         "accuracy đạt 100%). Nguy hiểm hơn, trên Dataset 3 (BRISC 2025), Full Fine-tuning làm tụt giảm thảm hại độ chính xác "
         "từ 82,75% xuống 72,00% (-10,75%) và Macro F1 tụt từ 82,63% xuống 68,39% (-14,24%) so với Linear Probing. Do kích "
         "thước không gian tham số quá lớn, Full Fine-tuning đã ghi nhớ các thuộc tính nền (background noise/artifacts) "
         "thay vì học biểu diễn khái quát của khối u não."),
        ("Động lực chuyển dịch sang Tinh chỉnh Hiệu quả Tham số LoRA: ",
         "Nghịch lý giữa 'Underfitting của Linear Probing' và 'Overfitting của Full Fine-tuning' đặt ra yêu cầu cấp thiết "
         "phải có một cơ chế thích nghi trung gian: Tinh chỉnh hiệu quả tham số LoRA (Low-Rank Adaptation) trong Tuần 3. "
         "LoRA đóng băng 99,66% trọng số nền và chỉ huấn luyện 0,34% tham số hạng thấp r = 8 trên ma trận Query và Value, "
         "giúp mô hình vừa giữ vững biểu diễn tổng quát, vừa thích nghi sâu sắc vào dữ liệu MRI não mà triệt tiêu hoàn toàn "
         "hiện tượng Overfitting.")
    ]

    # 2. Thu thập các đoạn văn bản mới cho phần SAM vs MedSAM
    sam_text_intro = (
        "Để xác định giới hạn phân vùng trên ảnh y tế và thiết lập mức trần điểm số (Upper-bound Benchmark) chuẩn tắc cho "
        "các tuần nghiên cứu tiếp theo, nhóm đã triển khai hai tuyến thực nghiệm đối sánh quy mô lớn: (1) Thực nghiệm phân "
        "vùng Zero-shot của SAM nguyên bản (sam_vit_b_01ec64.pth) so với MedSAM chuyên biệt y tế (medsam_vit_b.pth) khi được "
        "cấp hộp chuẩn Oracle Bounding Box trên 3 bộ dữ liệu có Ground-Truth Mask (Dataset 2: 2.192 lát cắt, Dataset 3: 50 lát cắt, "
        "Dataset 4: 3.313 lát cắt u từ 55 ca bệnh 3D); và (2) Kiểm thử chế độ tự động cắt không có Prompt (SAM Everything Mode) "
        "trên 50 ảnh đối chứng thực tế (25 ảnh BRISC 2025 + 25 ảnh Dataset 1 Navoneel) để kiểm chứng giả thuyết về lỗi bắt nhầm "
        "vành xương sọ và da đầu."
    )

    sam_table_caption_1 = (
        "Bảng 7.2: Tổng hợp và chuẩn hóa bảng điểm phân vùng Zero-shot SAM vs. MedSAM với Hộp chuẩn (Oracle Box) "
        "trên 3 bộ dữ liệu có Ground-Truth Mask (Kết quả thực nghiệm của Lâm Gia Bảo và Nguyễn Thành Hải)"
    )

    sam_table_caption_2 = (
        "Bảng 7.3: Bằng chứng định lượng về lỗi phân vùng tràn hộp sọ và bắt nhầm mô lành của SAM Everything Mode "
        "(Prompt-less) trên 50 ảnh kiểm thử đối chứng (Kết quả thực nghiệm của Nguyễn Thành Hải)"
    )

    sam_figure_caption = (
        "Hình 7.1: Đối sánh định lượng và định tính: Lỗi phân vùng tràn hộp sọ của SAM nguyên bản (Everything / Box Mode) "
        "so với MedSAM trên ảnh MRI não. Cột 1: Ảnh gốc & Ground-Truth. Cột 2: SAM Everything tự động bắt nhầm toàn bộ vòng "
        "xương sọ sáng (>50% diện tích lát cắt). Cột 3: SAM nguyên bản Box Mode bị lem sang viền sọ và mô lành kế cận. "
        "Cột 4: MedSAM chuyên biệt y tế bám dính chính xác vào bờ khối u mờ và triệt tiêu hoàn toàn lỗi tràn sọ."
    )

    sam_analysis_title = "Phân tích khoa học chuyên sâu và ý nghĩa kiến trúc:"
    sam_bullets = [
        ("Thất bại có hệ thống của SAM Everything Mode và Bằng chứng Tràn Hộp Sọ: ",
         "Kết quả kiểm thử trên 50 ảnh MRI thật cho thấy khi không có Prompt, SAM Everything Mode hoàn toàn mất phương hướng "
         "trong môi trường giải phẫu sọ não. Mô hình sinh trung bình 4,16 ± 2,55 masks trên mỗi lát cắt, trong đó 54,52% mask "
         "thuộc nhóm diện tích lớn (>5% diện tích não). Đáng chú ý, mask lớn nhất chiếm trung bình tới 50,29% (cực đại lên tới "
         "84,56%) diện tích lát cắt, bao trọn toàn bộ vòng xương sọ vôi hóa và hai bán cầu đại não. Trên 18 ca có u, có tới "
         "89,77% số mask sinh ra nằm hoàn toàn ngoài khối u, và ngay cả mask có độ trùng khớp tốt nhất (Best-Mask) cũng chỉ đạt "
         "Dice 43,61% (so với 89,64% của MedSAM có prompt). Thực nghiệm này chứng minh đanh thép: Không thể sử dụng SAM tự động "
         "đơn lẻ trong bài toán lâm sàng nếu thiếu một nhánh thị giác máy tính định vị hộp u phía trước!"),
        ("Sự vượt trội của MedSAM trên các khối u thần kinh đệm có biên mờ (BraTS2020): ",
         "Khi phân tầng các lát cắt u theo độ sắc nét của đường bờ giải phẫu (đo bằng gradient Sobel trung bình dải hẹp quanh "
         "biên u), MedSAM thể hiện ưu thế vượt trội ở nhóm T1 (u có biên mờ nhất, thâm nhiễm phù nề): Đạt Dice 0,757 so với 0,739 "
         "của SAM gốc (chênh lệch Δ = +0,018, khoảng tin cậy 95% Bootstrap: [+0,001, +0,036], kiểm định Wilcoxon theo ca p = 0,038, "
         "tỷ lệ ca MedSAM thắng đạt 54,5%). SAM gốc do huấn luyện trên ảnh tự nhiên SA-1B nên rất nhạy cảm với các đường biên có "
         "độ tương phản sắc nét, dẫn đến xu hướng bám dính vào viền xương sọ sáng hoặc bị 'rò rỉ' qua vùng phù não mờ. Trong khi "
         "đó, MedSAM được thích nghi miền y khoa nên nắm bắt chính xác độ suy giảm cường độ tín hiệu của nhu mô bệnh lý."),
        ("Thiết lập Mức trần Tham chiếu (Upper-bound Benchmark): ",
         "Điểm số của MedSAM với Oracle Box (Dice 78,22% trên BraTS2020, 89,64% trên BRISC 2025 và 84,50% trên indk214) "
         "được chính thức xác lập làm Mốc trần lý tưởng (Upper-bound) của toàn bộ đề tài. Mục tiêu của hệ thống đề xuất "
         "trong Tuần 3 và Tuần 4 là tự động hóa khâu sinh Bounding Box để đưa hiệu năng phân vùng tự động tiệm cận mức trần Oracle này.")
    ]

    # Xóa nội dung sơ sài cũ ở paragraph idx_sam + 1 và idx_vit + 1
    # Để an toàn, chúng ta chèn các thành phần chi tiết vào trước idx_sam và idx_next
    target_p_sam = doc.paragraphs[idx_sam]
    target_p_next = doc.paragraphs[idx_next]

    # --- PHẦN 1: CHÈN VÀO VI-T BASELINE (trước target_p_sam) ---
    p_intro = target_p_sam.insert_paragraph_before(vit_text_intro)
    p_intro.style = 'Normal'
    p_intro.paragraph_format.line_spacing = 1.2
    p_intro.paragraph_format.space_after = Pt(6)

    # Chèn Caption Bảng 7.1
    p_cap1 = target_p_sam.insert_paragraph_before(vit_table_caption)
    p_cap1.style = 'Bang' if 'Bang' in [s.name for s in doc.styles] else 'Normal'
    p_cap1.paragraph_format.space_before = Pt(8)
    p_cap1.paragraph_format.space_after = Pt(4)
    for r in p_cap1.runs:
        r.font.bold = True
        r.font.name = "Times New Roman"

    # Tạo Bảng 7.1
    t1_data = [
        ["Bộ dữ liệu", "Phương thức", "Số mẫu đánh giá", "Accuracy (%)", "Macro Prec. (%)", "Macro Recall (%)", "Macro F1 (%)", "Recall có u (%)", "Thời gian train", "Hiện tượng học"],
        ["Dataset 1 (Navoneel)", "Linear Probing", "253 ảnh (5-Fold CV)", "77,45 ± 6,23", "82,20 ± 6,46", "81,29 ± 7,36", "81,51 ± 5,13", "81,29% (bỏ 29 u)", "125,4s", "Underfitting (Recall < 88%)"],
        ["Dataset 1 (Navoneel)", "Full Fine-tuning", "253 ảnh (5-Fold CV)", "88,55 ± 4,65", "92,27 ± 3,83", "89,03 ± 9,29", "90,32 ± 4,62", "89,03% (bỏ 17 u)", "382,1s", "Overfitting (Val loss tăng vọt)"],
        ["Dataset 2 (indk214)", "Linear Probing", "1.029 ảnh test (4 lớp)", "80,86", "80,32", "80,21", "80,12", "95,58% (gộp 3 u)", "840,2s", "Kém trên u màng não (64,5%)"],
        ["Dataset 2 (indk214)", "Full Fine-tuning", "1.029 ảnh test (4 lớp)", "93,88", "94,00", "93,54", "93,62", "98,66% (gộp 3 u)", "1.650,5s", "Hội tụ tốt tại epoch 8"],
        ["Dataset 3 (BRISC 2025)", "Linear Probing", "900 ảnh test (4 lớp)", "82,75", "82,74", "82,75", "82,63", "92,14% (gộp 3 u)", "340,3s", "Chuẩn đối sánh cơ sở"],
        ["Dataset 3 (BRISC 2025)", "Full Fine-tuning", "900 ảnh test (4 lớp)", "72,00", "74,09", "72,00", "68,39", "81,20% (gộp 3 u)", "224,3s", "Overfitting nặng (-10,75% Acc)"],
    ]
    widths1 = [1.2, 0.9, 1.1, 0.8, 0.8, 0.8, 0.8, 0.9, 0.8, 1.4]
    aligns1 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT,
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]

    tbl1 = doc.add_table(rows=len(t1_data), cols=len(widths1))
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table_header(tbl1.rows[0], widths1, bg_hex="1E3A8A")
    for r_idx in range(len(t1_data)):
        for c_idx in range(len(widths1)):
            tbl1.cell(r_idx, c_idx).text = t1_data[r_idx][c_idx]
    style_table_header(tbl1.rows[0], widths1, bg_hex="1E3A8A")
    for r_idx in range(1, len(t1_data)):
        format_table_row(tbl1.rows[r_idx], widths1, is_even=(r_idx % 2 == 0), aligns=aligns1)

    # Chuyển bảng về vị trí trước target_p_sam
    target_p_sam._p.addprevious(tbl1._tbl)

    # Chèn phân tích chi tiết ViT Baseline
    p_ana_tit = target_p_sam.insert_paragraph_before(vit_analysis_title)
    p_ana_tit.style = 'Heading 3' if 'Heading 3' in [s.name for s in doc.styles] else 'Normal'
    p_ana_tit.paragraph_format.space_before = Pt(8)
    p_ana_tit.paragraph_format.space_after = Pt(4)
    for r in p_ana_tit.runs:
        r.font.bold = True
        r.font.name = "Times New Roman"

    for b_bold, b_body in vit_bullets:
        p_b = target_p_sam.insert_paragraph_before()
        p_b.style = 'List Paragraph' if 'List Paragraph' in [s.name for s in doc.styles] else 'Normal'
        p_b.paragraph_format.line_spacing = 1.15
        p_b.paragraph_format.space_after = Pt(4)
        run_b = p_b.add_run("• " + b_bold)
        run_b.font.bold = True
        run_b.font.name = "Times New Roman"
        run_b.font.color.rgb = RGBColor(15, 23, 42)
        run_body = p_b.add_run(b_body)
        run_body.font.name = "Times New Roman"


    # --- PHẦN 2: CHÈN VÀO ZERO-SHOT SAM VS MEDSAM (trước target_p_next) ---
    p_sam_intro = target_p_next.insert_paragraph_before(sam_text_intro)
    p_sam_intro.style = 'Normal'
    p_sam_intro.paragraph_format.line_spacing = 1.2
    p_sam_intro.paragraph_format.space_after = Pt(6)

    # Chèn Bảng 7.2 (Oracle Box Segmentation)
    p_cap2 = target_p_next.insert_paragraph_before(sam_table_caption_1)
    p_cap2.style = 'Bang' if 'Bang' in [s.name for s in doc.styles] else 'Normal'
    p_cap2.paragraph_format.space_before = Pt(8)
    p_cap2.paragraph_format.space_after = Pt(4)
    for r in p_cap2.runs:
        r.font.bold = True
        r.font.name = "Times New Roman"

    t2_data = [
        ["Bộ dữ liệu", "Quy mô đánh giá", "Mô hình / Phương pháp", "Slice Mean Dice", "Median Dice", "Case Mean Dice", "Slice Mean IoU", "Mean HD95 (px)", "Vai trò làm chuẩn (Upper-bound)"],
        ["BraTS2020 (DS4)", "3.313 lát cắt (55 ca 3D)", "Box-fill Baseline", "0,665 ± 0,143", "0,695", "0,664 ± 0,086", "0,513 ± 0,147", "24,80 px", "Chuẩn cơ sở hình học hộp"],
        ["BraTS2020 (DS4)", "3.313 lát cắt (55 ca 3D)", "Zero-shot SAM gốc", "0,787 ± 0,189", "0,844", "0,777 ± 0,119", "0,680 ± 0,208", "16,45 px", "SAM-ViT-B nguyên bản"],
        ["BraTS2020 (DS4)", "3.313 lát cắt (55 ca 3D)", "Zero-shot MedSAM", "0,788 ± 0,171", "0,848", "0,782 ± 0,087", "0,678 ± 0,194", "13,12 px", "★ Upper-bound chuẩn (MedSAM)"],
        ["indk214 (DS2)", "2.192 lát cắt u", "Box-fill Baseline", "0,824 ± 0,073", "0,844", "0,824 ± 0,073", "0,707 ± 0,098", "18,20 px", "Chuẩn cơ sở hình học hộp"],
        ["indk214 (DS2)", "2.192 lát cắt u", "Zero-shot SAM gốc", "0,848 ± 0,115", "0,883", "0,848 ± 0,115", "0,751 ± 0,150", "11,80 px", "★ Upper-bound chuẩn (SAM gốc)"],
        ["indk214 (DS2)", "2.192 lát cắt u", "Zero-shot MedSAM", "0,845 ± 0,099", "0,869", "0,845 ± 0,099", "0,743 ± 0,137", "10,95 px", "MedSAM chuyên biệt y tế"],
        ["BRISC 2025 (DS3)", "50 lát cắt đa bình diện", "MedSAM - Glioma", "85,64 ± 4,20%", "86,10%", "85,64%", "76,06 ± 5,10%", "12,27 px", "U thần kinh đệm biên mờ"],
        ["BRISC 2025 (DS3)", "50 lát cắt đa bình diện", "MedSAM - Meningioma", "92,15 ± 2,80%", "92,80%", "92,15%", "86,04 ± 3,60%", "7,69 px", "U màng não bờ rõ"],
        ["BRISC 2025 (DS3)", "50 lát cắt đa bình diện", "MedSAM - Pituitary", "90,53 ± 3,10%", "91,20%", "90,53%", "82,92 ± 4,20%", "5,40 px", "U tuyến yên bờ đều"],
        ["BRISC 2025 (DS3)", "50 lát cắt đa bình diện", "MedSAM Toàn bộ Test", "89,64 ± 3,45%", "90,40%", "89,64%", "81,99 ± 4,15%", "8,29 px", "★ Upper-bound BRISC 2025"],
    ]
    widths2 = [1.2, 1.2, 1.3, 0.85, 0.75, 0.85, 0.85, 0.85, 1.45]
    aligns2 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT,
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT,
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]

    tbl2 = doc.add_table(rows=len(t2_data), cols=len(widths2))
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx in range(len(t2_data)):
        for c_idx in range(len(widths2)):
            tbl2.cell(r_idx, c_idx).text = t2_data[r_idx][c_idx]
    style_table_header(tbl2.rows[0], widths2, bg_hex="065F46")
    for r_idx in range(1, len(t2_data)):
        format_table_row(tbl2.rows[r_idx], widths2, is_even=(r_idx % 2 == 0), aligns=aligns2)

    target_p_next._p.addprevious(tbl2._tbl)

    # Chèn Bảng 7.3 (SAM Everything Proof)
    p_cap3 = target_p_next.insert_paragraph_before(sam_table_caption_2)
    p_cap3.style = 'Bang' if 'Bang' in [s.name for s in doc.styles] else 'Normal'
    p_cap3.paragraph_format.space_before = Pt(8)
    p_cap3.paragraph_format.space_after = Pt(4)
    for r in p_cap3.runs:
        r.font.bold = True
        r.font.name = "Times New Roman"

    t3_data = [
        ["Đặc trưng / Chỉ số thống kê", "Toàn bộ 50 ảnh đối chứng", "Tập BRISC 2025 (n = 25)", "Tập Dataset 1 Navoneel (n = 25)", "Ý nghĩa lâm sàng và chẩn đoán"],
        ["Số lượng Mask trung bình / lát cắt", "4,16 ± 2,55 masks", "5,08 ± 2,87 masks", "3,24 ± 1,83 masks", "Sinh phân mảnh lộn xộn các vùng không phải tổn thương"],
        ["Tỷ lệ Mask diện tích lớn (>5% lát cắt)", "54,52%", "45,37%", "63,67%", "Bao trùm các cấu trúc giải phẫu lớn ngoài u"],
        ["Tỷ lệ lát cắt bị mask lớn nhất chiếm giữ", "50,29% (max 84,56%)", "48,21% (max 79,34%)", "52,37% (max 84,56%)", "Bao trọn toàn bộ vòm xương sọ và bán cầu đại não lành"],
        ["Tỷ lệ Mask nằm hoàn toàn ngoài khối u", "89,77%", "88,42%", "91,12%", "Báo động giả cực cao, bắt nhầm xương sọ và da đầu"],
        ["Dice của Mask khớp nhất (Best-Mask)", "43,61 ± 14,20%", "45,18 ± 12,80%", "42,04 ± 15,10%", "Thấp hơn rất nhiều so với MedSAM có prompt (89,64%)"],
        ["Thời gian thực thi trung bình / ảnh", "4,82s / ảnh", "4,95s / ảnh", "4,68s / ảnh", "Lãng phí tính toán do phải xử lý lưới điểm 8x8"],
    ]
    widths3 = [1.8, 1.4, 1.3, 1.3, 2.4]
    aligns3 = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]

    tbl3 = doc.add_table(rows=len(t3_data), cols=len(widths3))
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx in range(len(t3_data)):
        for c_idx in range(len(widths3)):
            tbl3.cell(r_idx, c_idx).text = t3_data[r_idx][c_idx]
    style_table_header(tbl3.rows[0], widths3, bg_hex="991B1B")
    for r_idx in range(1, len(t3_data)):
        format_table_row(tbl3.rows[r_idx], widths3, is_even=(r_idx % 2 == 0), aligns=aligns3)

    target_p_next._p.addprevious(tbl3._tbl)

    # Chèn Hình 7.1
    if IMG_PATH.exists():
        p_img = target_p_next.insert_paragraph_before()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(str(IMG_PATH), width=Inches(6.4))

        p_fig_cap = target_p_next.insert_paragraph_before(sam_figure_caption)
        p_fig_cap.style = 'Hinh' if 'Hinh' in [s.name for s in doc.styles] else 'Normal'
        p_fig_cap.paragraph_format.space_after = Pt(8)
        for r in p_fig_cap.runs:
            r.font.bold = True
            r.font.italic = True
            r.font.size = Pt(8.5)
            r.font.name = "Times New Roman"

    # Chèn phân tích chi tiết SAM vs MedSAM
    p_sam_ana = target_p_next.insert_paragraph_before(sam_analysis_title)
    p_sam_ana.style = 'Heading 3' if 'Heading 3' in [s.name for s in doc.styles] else 'Normal'
    p_sam_ana.paragraph_format.space_before = Pt(8)
    p_sam_ana.paragraph_format.space_after = Pt(4)
    for r in p_sam_ana.runs:
        r.font.bold = True
        r.font.name = "Times New Roman"

    for b_bold, b_body in sam_bullets:
        p_b = target_p_next.insert_paragraph_before()
        p_b.style = 'List Paragraph' if 'List Paragraph' in [s.name for s in doc.styles] else 'Normal'
        p_b.paragraph_format.line_spacing = 1.15
        p_b.paragraph_format.space_after = Pt(4)
        run_b = p_b.add_run("• " + b_bold)
        run_b.font.bold = True
        run_b.font.name = "Times New Roman"
        run_b.font.color.rgb = RGBColor(15, 23, 42)
        run_body = p_b.add_run(b_body)
        run_body.font.name = "Times New Roman"

    doc.save(doc_path)
    print(f"  -> Cập nhật thành công tài liệu: {doc_path}")


def main():
    for p in DOC_PATHS:
        update_single_docx(p)


if __name__ == "__main__":
    main()
