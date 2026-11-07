import os
import sys
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def verify_checksum(file_path):
    sha_file = file_path + '.sha256'
    if not os.path.exists(sha_file):
        return None
    with open(sha_file, 'r', encoding='utf-8') as f:
        expected = f.read().split()[0].strip()
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual = h.hexdigest()
    return {
        'expected': expected,
        'actual': actual,
        'match': actual == expected
    }

def audit_brisc_dataset(dataset_dir):
    print("=" * 60)
    print("BẮT ĐẦU KIỂM ĐỊNH CHẤT LƯỢNG DATASET BRISC 2025")
    print(f"Đường dẫn dataset: {dataset_dir}")
    print("=" * 60)

    report = {
        "dataset_dir": dataset_dir,
        "manifest_checksums": {},
        "missing_files": [],
        "corrupted_files": [],
        "dimension_mismatches": [],
        "invalid_labels": [],
        "segmentation_pairs_verified": 0,
        "classification_images_verified": 0,
        "mask_pixel_values": [],
        "status": "PASS"
    }

    manifest_csv = os.path.join(dataset_dir, "manifest.csv")
    manifest_json = os.path.join(dataset_dir, "manifest.json")

    # 1. Checksum Verification
    print("\n1. Kiểm tra tính toàn vẹn Checksum SHA-256 của Manifest...")
    csv_chk = verify_checksum(manifest_csv)
    json_chk = verify_checksum(manifest_json)
    report["manifest_checksums"]["manifest.csv"] = csv_chk
    report["manifest_checksums"]["manifest.json"] = json_chk
    print(f" - manifest.csv SHA-256 match: {csv_chk['match'] if csv_chk else 'N/A'}")
    print(f" - manifest.json SHA-256 match: {json_chk['match'] if json_chk else 'N/A'}")

    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Không tìm thấy file manifest tại {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    print(f"Tổng số bản ghi trong manifest: {len(df)}")

    # 2. Check label validity
    print("\n2. Kiểm tra tính hợp lệ của nhãn (Labels)...")
    valid_tumors = {'gl', 'me', 'pi', 'no'}
    valid_planes = {'ax', 'co', 'sa'}
    
    invalid_rows = df[~df['tumor_code'].isin(valid_tumors) | ~df['plane_code'].isin(valid_planes)]
    if len(invalid_rows) > 0:
        print(f" CẢNH BÁO: Phát hiện {len(invalid_rows)} bản ghi có nhãn không hợp lệ!")
        report["invalid_labels"] = invalid_rows['relative_path'].tolist()
        report["status"] = "WARNING_INVALID_LABELS"
    else:
        print(" [OK] Tất cả nhãn tumor_code và plane_code đều hợp lệ 100%.")

    # 3. Verify Segmentation Pairs (Compatibility MRI vs Mask)
    print("\n3. Kiểm tra tính tương thích kích thước giữa ảnh MRI và Mask...")
    seg_masks = df[df['is_mask']]
    mask_pixels_set = set()

    for idx, row in seg_masks.iterrows():
        mask_rel = row['relative_path']
        img_rel = row['linked_image']
        
        mask_path = os.path.join(dataset_dir, mask_rel)
        img_path = os.path.join(dataset_dir, img_rel)

        # File existence
        if not os.path.exists(mask_path):
            report["missing_files"].append(mask_rel)
            continue
        if not os.path.exists(img_path):
            report["missing_files"].append(img_rel)
            continue

        # Open and check readability & dimensions
        try:
            with Image.open(mask_path) as m_im:
                m_size = m_im.size # (width, height)
                m_arr = np.array(m_im)
                mask_pixels_set.update(np.unique(m_arr).tolist())
        except Exception as e:
            report["corrupted_files"].append({"file": mask_rel, "error": str(e)})
            continue

        try:
            with Image.open(img_path) as i_im:
                i_size = i_im.size
        except Exception as e:
            report["corrupted_files"].append({"file": img_rel, "error": str(e)})
            continue

        if m_size != i_size:
            report["dimension_mismatches"].append({
                "image": img_rel,
                "image_size": i_size,
                "mask": mask_rel,
                "mask_size": m_size
            })

        report["segmentation_pairs_verified"] += 1
        if report["segmentation_pairs_verified"] % 1000 == 0:
            print(f" -> Đã kiểm tra {report['segmentation_pairs_verified']}/{len(seg_masks)} cặp ảnh-mask...")

    report["mask_pixel_values"] = sorted(list(mask_pixels_set))

    # 4. Verify Classification Images
    print("\n4. Kiểm tra toàn bộ ảnh trong Classification Task...")
    clf_imgs = df[df['task'] == 'classification']
    for idx, row in clf_imgs.iterrows():
        img_rel = row['relative_path']
        img_path = os.path.join(dataset_dir, img_rel)
        if not os.path.exists(img_path):
            report["missing_files"].append(img_rel)
            continue
        try:
            with Image.open(img_path) as im:
                _ = im.size
            report["classification_images_verified"] += 1
        except Exception as e:
            report["corrupted_files"].append({"file": img_rel, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM ĐỊNH DATASET:")
    print(f" - Tổng số file bị thiếu (Missing files): {len(report['missing_files'])}")
    print(f" - Tổng số file bị lỗi/hỏng (Corrupted files): {len(report['corrupted_files'])}")
    print(f" - Số cặp sai lệch kích thước (Dimension mismatches): {len(report['dimension_mismatches'])}")
    print(f" - Số nhãn không hợp lệ (Invalid labels): {len(report['invalid_labels'])}")
    print(f" - Tổng số cặp Segmentation đã xác thực: {report['segmentation_pairs_verified']}")
    print(f" - Tổng số ảnh Classification đã xác thực: {report['classification_images_verified']}")
    print(f" - Giá trị pixel xuất hiện trong Mask: {report['mask_pixel_values']}")
    print(f" -> Trạng thái kiểm định: {'ĐẠT CHUẨN (PASS)' if len(report['corrupted_files']) == 0 and len(report['dimension_mismatches']) == 0 else 'CÓ LỖI'}")
    print("=" * 60)

    output_path = os.path.join(os.path.dirname(__file__), "audit_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Báo cáo chi tiết đã được lưu tại: {output_path}")

    return report

if __name__ == "__main__":
    import sys
    data_dir = r"E:\archive\brisc2025"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    audit_brisc_dataset(data_dir)
