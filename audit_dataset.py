import os
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    str(REPO_ROOT / "dataset" / "brisc2025" / "brisc2025")
    if (REPO_ROOT / "dataset" / "brisc2025" / "brisc2025").exists()
    else r"E:\archive\brisc2025"
)


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
        "empty_masks": [],
        "invalid_labels": [],
        "segmentation_pairs_verified": 0,
        "classification_images_verified": 0,
        "distinct_resolutions_count": 0,
        "top_resolutions_wh": [],
        "duplicate_sha256_groups": 0,
        "duplicate_sha256_extra_files": 0,
        "cross_split_sha256_leak_groups": 0,
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

    # 3. Verify Segmentation Pairs (1-1 Compatibility MRI vs Mask & Empty Mask Check)
    print("\n3. Kiểm tra khớp nối kích thước 1-1 giữa ảnh MRI và Mask & kiểm tra nhãn rỗng...")
    seg_masks = df[df['is_mask']]
    mask_pixels_set = set()
    resolutions = {}

    for idx, row in seg_masks.iterrows():
        mask_rel = row['relative_path']
        img_rel = row['linked_image']

        mask_path = os.path.join(dataset_dir, mask_rel)
        img_path = os.path.join(dataset_dir, img_rel)

        if not os.path.exists(mask_path):
            report["missing_files"].append(mask_rel)
            continue
        if not os.path.exists(img_path):
            report["missing_files"].append(img_rel)
            continue

        try:
            with Image.open(mask_path) as m_im:
                m_size = m_im.size  # (width, height)
                m_arr = np.array(m_im.convert("L"))
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
            continue

        # Kiểm tra nhãn rỗng sai quy cách sau nhị phân hóa (ngưỡng > 127)
        if int(np.count_nonzero(m_arr > 127)) == 0:
            report["empty_masks"].append(mask_rel)
            continue

        resolutions[i_size] = resolutions.get(i_size, 0) + 1
        report["segmentation_pairs_verified"] += 1
        if report["segmentation_pairs_verified"] % 1000 == 0:
            print(f" -> Đã kiểm tra {report['segmentation_pairs_verified']}/{len(seg_masks)} cặp ảnh-mask...")

    report["mask_pixel_values"] = sorted(list(mask_pixels_set))

    # 4. Verify Classification Images & Duplicate SHA-256 Analysis
    print("\n4. Kiểm tra toàn bộ ảnh trong Classification Task & trùng lặp SHA-256...")
    clf_imgs = df[df['task'] == 'classification']
    for idx, row in clf_imgs.iterrows():
        img_rel = row['relative_path']
        img_path = os.path.join(dataset_dir, img_rel)
        if not os.path.exists(img_path):
            report["missing_files"].append(img_rel)
            continue
        try:
            with Image.open(img_path) as im:
                if row['tumor_code'] == 'no':
                    resolutions[im.size] = resolutions.get(im.size, 0) + 1
            report["classification_images_verified"] += 1
        except Exception as e:
            report["corrupted_files"].append({"file": img_rel, "error": str(e)})

    report["distinct_resolutions_count"] = len(resolutions)
    report["top_resolutions_wh"] = [
        {"size_wh": list(sz), "count": cnt}
        for sz, cnt in sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    if 'sha256' in clf_imgs.columns:
        sha_groups = clf_imgs.groupby('sha256')
        dup_groups = [g for _, g in sha_groups if len(g) > 1]
        report["duplicate_sha256_groups"] = len(dup_groups)
        report["duplicate_sha256_extra_files"] = int(sum(len(g) - 1 for g in dup_groups))
        report["cross_split_sha256_leak_groups"] = int(sum(g['split'].nunique() > 1 for g in dup_groups))

    # Summary
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM ĐỊNH DATASET:")
    print(f" - Tổng số file bị thiếu (Missing files): {len(report['missing_files'])}")
    print(f" - Tổng số file bị lỗi/hỏng (Corrupted files): {len(report['corrupted_files'])}")
    print(f" - Số cặp sai lệch kích thước 1-1 (Dimension mismatches): {len(report['dimension_mismatches'])}")
    print(f" - Số mask rỗng sai quy cách (Empty masks): {len(report['empty_masks'])}")
    print(f" - Số nhãn không hợp lệ (Invalid labels): {len(report['invalid_labels'])}")
    print(f" - Tổng số cặp Segmentation hợp lệ đã xác thực: {report['segmentation_pairs_verified']}")
    print(f" - Tổng số ảnh Classification hợp lệ đã xác thực: {report['classification_images_verified']}")
    print(f" - Số độ phân giải (W, H) khác nhau: {report['distinct_resolutions_count']}")
    print(f" - Phát hiện trùng lặp SHA-256: {report['duplicate_sha256_groups']} nhóm "
          f"({report['duplicate_sha256_extra_files']} ảnh dư, {report['cross_split_sha256_leak_groups']} nhóm rò rỉ train/test gốc)")
    passed = (
        len(report['corrupted_files']) == 0
        and len(report['dimension_mismatches']) == 0
        and len(report['empty_masks']) == 0
    )
    print(f" -> Trạng thái kiểm định: {'ĐẠT CHUẨN (PASS)' if passed else 'CÓ LỖI'}")
    print("=" * 60)

    output_path = os.path.join(os.path.dirname(__file__), "audit_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Báo cáo chi tiết đã được lưu tại: {output_path}")

    return report


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    audit_brisc_dataset(data_dir)
