import os
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    str(REPO_ROOT / "dataset" / "brisc2025" / "brisc2025")
    if (REPO_ROOT / "dataset" / "brisc2025" / "brisc2025").exists()
    else r"E:\archive\brisc2025"
)
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "output")

# Nhãn chuẩn của toàn dự án: 0 No Tumor, 1 Glioma, 2 Meningioma, 3 Pituitary
CLASS_MAPPING = {
    'no_tumor': 0,
    'glioma': 1,
    'meningioma': 2,
    'pituitary': 3,
}


def extract_bounding_boxes(mask, class_id=1):
    """Trích xuất hộp chuẩn toàn cục gt_box = [x_min, y_min, x_max, y_max] bao quanh toàn bộ Ground-Truth Mask,
    cùng danh sách các hộp thành phần và định dạng YOLO normalized [class_id, x_c, y_c, w, h].
    """
    H, W = mask.shape[:2]
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return None, [], [], 0

    # Hộp chuẩn bao quanh toàn bộ vùng u trên mask (x_max, y_max là biên loại trừ)
    gt_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    total_tumor_pixels = int(xs.size)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    yolo_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])
            x_center = (x + w / 2.0) / float(W)
            y_center = (y + h / 2.0) / float(H)
            norm_w = w / float(W)
            norm_h = h / float(H)
            yolo_boxes.append([class_id, round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)])

    if not bboxes:
        x0, y0, x1, y1 = gt_box
        w, h = x1 - x0, y1 - y0
        bboxes = [gt_box]
        yolo_boxes = [[
            class_id,
            round((x0 + w / 2.0) / float(W), 6),
            round((y0 + h / 2.0) / float(H), 6),
            round(w / float(W), 6),
            round(h / float(H), 6),
        ]]

    return gt_box, bboxes, yolo_boxes, total_tumor_pixels


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Áp dụng Contrast Limited Adaptive Histogram Equalization (CLAHE) tăng cường tương phản cục bộ."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    return np.stack([clahe.apply(image[..., ch]) for ch in range(image.shape[2])], axis=-1)


def group_patients_and_split(df_samples, random_state=42):
    """Gom cụm bệnh nhân bằng đồ thị liên thông (Union-Find) kết hợp:
      1. Trùng khớp mã băm nội dung ảnh (sha256) — chặn đứng rò rỉ 46 cụm ảnh trùng giữa train và test gốc.
      2. Chuỗi lát cắt liên tiếp trong cùng tập gốc (split), loại u (tumor_code), mặt phẳng cắt (plane_code) và kích thước (width, height).
    Sau đó chia Stratified Group 70% Train / 15% Validation / 15% Test đảm bảo 0% rò rỉ bệnh nhân.
    """
    seg = df_samples.copy().reset_index(drop=True)
    n = len(seg)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # 1. Liên kết tuyệt đối theo sha256 của ảnh MRI
    by_sha = defaultdict(list)
    for i, sha in enumerate(seg['img_sha256']):
        by_sha[sha].append(i)
    for idxs in by_sha.values():
        for k in idxs[1:]:
            union(idxs[0], k)

    # 2. Liên kết các lát cắt liên tiếp trong cùng (split, tumor_code, plane_code, width, height)
    for (_, tumor, plane, w, h), grp in seg.groupby(['split', 'tumor_code', 'plane_code', 'width', 'height']):
        grp_sorted = grp.sort_values('index')
        idxs = grp_sorted.index.tolist()
        indices = grp_sorted['index'].tolist()
        run_len = 1
        for k in range(len(idxs) - 1):
            if indices[k + 1] - indices[k] == 1 and run_len < 12:
                union(idxs[k], idxs[k + 1])
                run_len += 1
            else:
                run_len = 1

    comp_map = {}
    patient_ids = []
    for i in range(n):
        r = find(i)
        if r not in comp_map:
            tumor = seg.loc[i, 'tumor_code']
            plane = seg.loc[i, 'plane_code']
            comp_map[r] = f"{tumor}_{plane}_pat_{len(comp_map) + 1:04d}"
        patient_ids.append(comp_map[r])
    seg['patient_id'] = patient_ids

    # Chia tập Stratified Group 70% Train / 15% Val / 15% Test theo từng lớp u
    rng = np.random.default_rng(random_state)
    pat_groups = defaultdict(list)
    for i, pid in enumerate(seg['patient_id']):
        pat_groups[pid].append(i)

    by_class = defaultdict(list)
    for pid, idxs in pat_groups.items():
        cls_lbl = seg.loc[idxs[0], 'tumor_label']
        plane_lbl = seg.loc[idxs[0], 'plane_code']
        by_class[cls_lbl].append((pid, plane_lbl, idxs))

    split_col = [None] * n
    for cls_lbl, groups in sorted(by_class.items()):
        n_cls = sum(len(idxs) for _, _, idxs in groups)
        t_test = int(round(0.15 * n_cls))
        t_val = int(round(0.15 * n_cls))
        t_train = n_cls - t_test - t_val
        targets = {'train': t_train, 'val': t_val, 'test': t_test}
        counts = {'train': 0, 'val': 0, 'test': 0}
        plane_cnt = {sp: Counter() for sp in ('train', 'val', 'test')}

        perm = rng.permutation(len(groups))
        shuffled = [groups[i] for i in perm]
        shuffled.sort(key=lambda g: len(g[2]), reverse=True)

        for pid, plane_lbl, idxs in shuffled:
            k = len(idxs)
            cands = [sp for sp in ('test', 'val', 'train') if counts[sp] + k <= targets[sp]]
            if not cands:
                cands = ['train', 'val', 'test']
            best_sp = min(
                cands,
                key=lambda sp: (
                    (counts[sp] + k - targets[sp]) / max(1, targets[sp]),
                    plane_cnt[sp][plane_lbl] / max(1, targets[sp]),
                ),
            )
            counts[best_sp] += k
            plane_cnt[best_sp][plane_lbl] += k
            for idx in idxs:
                split_col[idx] = best_sp

    seg['split_70_15_15'] = split_col

    # Kiểm tra chống rò rỉ dữ liệu (Zero Patient & SHA-256 Data Leakage Assertion)
    for col in ('patient_id', 'img_sha256'):
        s_train = set(seg.loc[seg['split_70_15_15'] == 'train', col])
        s_val = set(seg.loc[seg['split_70_15_15'] == 'val', col])
        s_test = set(seg.loc[seg['split_70_15_15'] == 'test', col])
        assert len(s_train & s_val) == 0, f"LỖI: Rò rỉ {col} giữa Train và Val!"
        assert len(s_train & s_test) == 0, f"LỖI: Rò rỉ {col} giữa Train và Test!"
        assert len(s_val & s_test) == 0, f"LỖI: Rò rỉ {col} giữa Val và Test!"

    return seg


def run_preprocessing_pipeline(dataset_dir, output_dir, generate_samples=True, save_clahe_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(dataset_dir, "manifest.csv")
    df = pd.read_csv(manifest_csv)

    # Map sha256 của từng ảnh từ manifest
    img_rows = df[~df['is_mask']].set_index('relative_path')
    seg_masks = df[df['is_mask']].copy()
    seg_masks['img_sha256'] = seg_masks['linked_image'].map(img_rows['sha256'])

    print("\n--- BƯỚC 1: GOM CỤM BỆNH NHÂN (CHỐNG RÒ RỈ SHA-256 & CHUỖI QUÉT) VÀ CHIA TỶ LỆ 70% / 15% / 15% ---")
    seg_processed = group_patients_and_split(seg_masks, random_state=42)

    counts = seg_processed['split_70_15_15'].value_counts()
    percs = seg_processed['split_70_15_15'].value_counts(normalize=True) * 100
    print(f"Tổng số bệnh nhân (Segmentation): {seg_processed['patient_id'].nunique()}")
    print("Phân phối lát cắt:")
    for split_name in ['train', 'val', 'test']:
        print(f" - {split_name.upper():<5}: {counts[split_name]:>5} slices ({percs[split_name]:.2f}%)")

    print("\n--- BƯỚC 2: KIỂM ĐỊNH 1-1, LỌC CLAHE VÀ TRÍCH XUẤT BOUNDING BOX CHUẨN ---")
    results = []
    dropped = []
    total = len(seg_processed)

    for idx, (_, row) in enumerate(seg_processed.iterrows()):
        mask_rel = row['relative_path']
        img_rel = row['linked_image']

        img_full = os.path.join(dataset_dir, img_rel)
        mask_full = os.path.join(dataset_dir, mask_rel)

        img = cv2.imread(img_full, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_full, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            dropped.append((img_rel, "corrupted_image_or_mask"))
            continue

        ih, iw = img.shape[:2]
        mh, mw = mask.shape[:2]
        if (ih, iw) != (mh, mw):
            dropped.append((img_rel, f"dimension_mismatch_{iw}x{ih}_vs_{mw}x{mh}"))
            continue

        class_id = CLASS_MAPPING.get(row['tumor_label'], 1)
        gt_box, bboxes, yolo_boxes, tumor_pixels = extract_bounding_boxes(mask, class_id=class_id)
        if gt_box is None or tumor_pixels == 0:
            dropped.append((img_rel, "empty_mask"))
            continue

        # Áp dụng bộ lọc CLAHE lên ảnh MRI
        enhanced_img = apply_clahe(img)
        if save_clahe_dir:
            out_img_p = Path(save_clahe_dir) / "images" / f"{Path(row['filename']).stem}.png"
            out_msk_p = Path(save_clahe_dir) / "masks" / f"{Path(row['filename']).stem}.png"
            out_img_p.parent.mkdir(parents=True, exist_ok=True)
            out_msk_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_p), enhanced_img)
            cv2.imwrite(str(out_msk_p), ((mask > 127).astype(np.uint8) * 255))

        results.append({
            'filename': row['filename'],
            'image_path': img_rel,
            'mask_path': mask_rel,
            'width': iw,
            'height': ih,
            'tumor_label': row['tumor_label'],
            'plane_label': row['plane_label'],
            'patient_id': row['patient_id'],
            'split_70_15_15': row['split_70_15_15'],
            'gt_box_xyxy': json.dumps(gt_box),
            'num_bboxes': len(bboxes),
            'bboxes_pixel_xyxy': json.dumps(bboxes),
            'bboxes_yolo': json.dumps(yolo_boxes),
            'tumor_pixels': tumor_pixels,
            'tumor_area_ratio': round(tumor_pixels / float(iw * ih), 6)
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f" -> Đã kiểm định 1-1, lọc CLAHE & trích xuất Bounding Box: {idx + 1}/{total} ảnh...")

    print(f"Số mẫu lỗi/rỗng bị loại bỏ: {len(dropped)}")
    res_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "brisc2025_processed_manifest.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Đã lưu manifest sau xử lý tại: {output_csv}")

    if generate_samples and not res_df.empty:
        print("\n--- BƯỚC 3: TẠO ẢNH MẪU MINH HOẠ (GỐC vs CLAHE vs MASK vs BBOX) ---")
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        sample_rows = res_df.groupby('tumor_label').head(3)
        for _, s_row in sample_rows.iterrows():
            img_path = os.path.join(dataset_dir, s_row['image_path'])
            mask_path = os.path.join(dataset_dir, s_row['mask_path'])

            orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if orig is None or mask is None:
                continue

            enhanced = apply_clahe(orig)
            vis_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            gt_box = json.loads(s_row['gt_box_xyxy'])
            cv2.rectangle(vis_img, (gt_box[0], gt_box[1]), (gt_box[2] - 1, gt_box[3] - 1), (0, 255, 0), 2)
            cv2.putText(vis_img, s_row['tumor_label'], (gt_box[0], max(gt_box[1] - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            mask_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            mask_rgb[mask > 127] = [0, 0, 255]

            orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            cv2.putText(orig_bgr, "Original MRI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            clahe_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.putText(clahe_bgr, "CLAHE Enhanced", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            cv2.putText(mask_rgb, "Mask Overlay", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(vis_img, "Tumor BBox", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            panel = np.hstack([orig_bgr, clahe_bgr, mask_rgb, vis_img])
            base_name = os.path.splitext(s_row['filename'])[0]
            out_sample_file = os.path.join(samples_dir, f"sample_{s_row['tumor_label']}_{base_name}.jpg")
            cv2.imwrite(out_sample_file, panel)
            print(f" -> Đã lưu ảnh so sánh: {out_sample_file}")

    print("\n" + "=" * 60)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH TIỀN XỬ LÝ VÀ PHÂN CHIA DỮ LIỆU!")
    print("=" * 60)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    run_preprocessing_pipeline(data_dir, out_dir, generate_samples=True)
