import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

CLASS_MAPPING = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'no_tumor': 3
}

def extract_bounding_boxes(mask, class_id=0):
    """
    Trích xuất danh sách Bounding Boxes từ Mask nhị phân.
    Trả về cả toạ độ pixel [xmin, ymin, xmax, ymax] và định dạng YOLO normalized [x_c, y_c, w, h].
    """
    H, W = mask.shape[:2]
    # Ngưỡng nhị phân loại bỏ viền mờ do nội suy hoặc khử răng cưa
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    yolo_boxes = []
    total_tumor_pixels = int(np.count_nonzero(bin_mask))
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:  # Bỏ qua các đốm nhiễu siêu nhỏ
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])
            
            # YOLO format normalized [0, 1]
            x_center = (x + w / 2.0) / W
            y_center = (y + h / 2.0) / H
            norm_w = w / float(W)
            norm_h = h / float(H)
            yolo_boxes.append([class_id, round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)])
            
    return bboxes, yolo_boxes, total_tumor_pixels

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng Contrast Limited Adaptive Histogram Equalization (CLAHE) tăng cường tương phản.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def group_patients_and_split(df_masks, chunk_size=12, random_state=42):
    """
    Gom cụm các lát cắt liên tục thành từng chuỗi bệnh nhân (patient_id)
    và chia Train / Validation / Test theo tỷ lệ 70% / 15% / 15% không rò rỉ dữ liệu.
    """
    seg = df_masks.copy()
    
    # Gom cụm theo chuỗi lát cắt của từng phân loại và mặt cắt phẳng
    for (tumor, plane), group in seg.groupby(['tumor_code', 'plane_code']):
        group = group.sort_values('index')
        for i, idx in enumerate(group.index):
            pat_num = i // chunk_size
            seg.loc[idx, 'patient_id'] = f"{tumor}_{plane}_pat_{pat_num:03d}"
            
    patients_df = seg[['patient_id', 'tumor_label']].drop_duplicates()
    
    # Phân tầng 70% Train, 30% Temp (15% Val + 15% Test)
    train_pats, temp_pats = train_test_split(
        patients_df['patient_id'],
        test_size=0.30,
        stratify=patients_df['tumor_label'],
        random_state=random_state
    )
    
    temp_df = patients_df[patients_df['patient_id'].isin(temp_pats)]
    val_pats, test_pats = train_test_split(
        temp_df['patient_id'],
        test_size=0.50,
        stratify=temp_df['tumor_label'],
        random_state=random_state
    )
    
    train_set = set(train_pats)
    val_set = set(val_pats)
    test_set = set(test_pats)
    
    # Kiểm tra rò rỉ dữ liệu (Zero Data Leakage Assertion)
    assert len(train_set & val_set) == 0, "LỖI: Rò rỉ bệnh nhân giữa Train và Val!"
    assert len(train_set & test_set) == 0, "LỖI: Rò rỉ bệnh nhân giữa Train và Test!"
    assert len(val_set & test_set) == 0, "LỖI: Rò rỉ bệnh nhân giữa Val và Test!"
    
    seg.loc[seg['patient_id'].isin(train_set), 'split_70_15_15'] = 'train'
    seg.loc[seg['patient_id'].isin(val_set), 'split_70_15_15'] = 'val'
    seg.loc[seg['patient_id'].isin(test_set), 'split_70_15_15'] = 'test'
    
    return seg

def run_preprocessing_pipeline(dataset_dir, output_dir, generate_samples=True):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(dataset_dir, "manifest.csv")
    df = pd.read_csv(manifest_csv)
    
    # Lấy 4,793 cặp mask và linked_image từ manifest
    seg_masks = df[df['is_mask']].copy()
    
    print("\n--- BƯỚC 1: GOM CỤM BỆNH NHÂN VÀ CHIA TỶ LỆ 70% / 15% / 15% ---")
    seg_processed = group_patients_and_split(seg_masks, chunk_size=12, random_state=42)
    
    counts = seg_processed['split_70_15_15'].value_counts()
    percs = seg_processed['split_70_15_15'].value_counts(normalize=True) * 100
    print(f"Tổng số bệnh nhân: {seg_processed['patient_id'].nunique()}")
    print("Phân phối lát cắt:")
    for split_name in ['train', 'val', 'test']:
        print(f" - {split_name.upper():<5}: {counts[split_name]:>5} slices ({percs[split_name]:.2f}%)")
        
    print("\n--- BƯỚC 2: TRÍCH XUẤT BOUNDING BOX VÀ TÍNH TOÁN CLAHE ---")
    results = []
    
    total = len(seg_processed)
    for idx, (row_idx, row) in enumerate(seg_processed.iterrows()):
        mask_rel = row['relative_path']
        img_rel = row['linked_image']
        
        img_full = os.path.join(dataset_dir, img_rel)
        mask_full = os.path.join(dataset_dir, mask_rel)
        
        mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        class_id = CLASS_MAPPING.get(row['tumor_label'], 0)
        bboxes, yolo_boxes, tumor_pixels = extract_bounding_boxes(mask, class_id=class_id)
        
        results.append({
            'filename': row['filename'],
            'image_path': img_rel,
            'mask_path': mask_rel,
            'tumor_label': row['tumor_label'],
            'plane_label': row['plane_label'],
            'patient_id': row['patient_id'],
            'split_70_15_15': row['split_70_15_15'],
            'num_bboxes': len(bboxes),
            'bboxes_pixel_xyxy': json.dumps(bboxes),
            'bboxes_yolo': json.dumps(yolo_boxes),
            'tumor_pixels': tumor_pixels,
            'tumor_area_ratio': round(tumor_pixels / (512 * 512), 6)
        })
        
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f" -> Đã trích xuất Bounding Box: {idx + 1}/{total} ảnh...")

    res_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "brisc2025_processed_manifest.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Đã lưu manifest sau xử lý tại: {output_csv}")

    # Generate sample images if requested
    if generate_samples:
        print("\n--- BƯỚC 3: TẠO ẢNH MẪU MINH HOẠ (GỐC vs CLAHE vs MASK vs BBOX) ---")
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        sample_rows = res_df.groupby('tumor_label').head(3)
        for _, s_row in sample_rows.iterrows():
            img_path = os.path.join(dataset_dir, s_row['image_path'])
            mask_path = os.path.join(dataset_dir, s_row['mask_path'])
            
            orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # CLAHE enhancement
            enhanced = apply_clahe(orig)
            
            # Bounding box overlay on CLAHE image
            vis_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            bboxes = json.loads(s_row['bboxes_pixel_xyxy'])
            for box in bboxes:
                cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(vis_img, s_row['tumor_label'], (box[0], max(box[1]-6, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            # Mask color overlay (Red)
            mask_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            mask_rgb[mask > 127] = [0, 0, 255]
            
            # Headers for each panel
            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            cv2.putText(orig_bgr, "Original MRI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
            clahe_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.putText(clahe_bgr, "CLAHE Enhanced", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            
            cv2.putText(mask_rgb, "Mask Overlay", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(vis_img, "Tumor BBox", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # Stack horizontally: 512 x (512 * 4) = 512 x 2048
            panel = np.hstack([orig_bgr, clahe_bgr, mask_rgb, vis_img])
            
            base_name = os.path.splitext(s_row['filename'])[0]
            out_sample_file = os.path.join(samples_dir, f"sample_{s_row['tumor_label']}_{base_name}.jpg")
            cv2.imwrite(out_sample_file, panel)
            print(f" -> Đã lưu ảnh so sánh: {out_sample_file}")

    print("\n" + "=" * 60)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH TIỀN XỬ LÝ VÀ PHÂN CHIA DỮ LIỆU!")
    print("=" * 60)

if __name__ == "__main__":
    data_dir = r"E:\archive\brisc2025"
    out_dir = r"E:\archive\brisc2025_pipeline\output"
    run_preprocessing_pipeline(data_dir, out_dir, generate_samples=True)
