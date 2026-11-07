"""Cắt sát hộp sọ + CLAHE trên toàn bộ DS1 (navoneel) và DS2 (indk214); xuất ds2_annotations.json.

Chạy từ thư mục gốc dự án:
    python -m data.build_ds1_ds2                 # chạy toàn bộ
    python -m data.build_ds1_ds2 --limit 20      # chạy thử 20 ảnh mỗi nhóm

Đầu ra:
    dataset/ds1_cropped/{no,yes}/*.png            ảnh đã cắt hộp sọ + CLAHE
    dataset/ds2_cropped/images|masks/*.png        ảnh (cắt + CLAHE) và mask (cắt cùng khung)
    dataset/ds1_annotations.json
    dataset/ds2_annotations.json                  có gt_box = [x_min, y_min, x_max, y_max] bao mask
    reports/build_ds1_ds2/crop_log.csv            nhật ký từng ảnh (khung cắt, cờ fallback, lý do loại)

gt_box nằm trong hệ tọa độ của ẢNH ĐÃ CẮT (khớp với file ảnh được lưu); gt_box_orig là tọa độ trong ảnh gốc.
x_max/y_max là biên loại trừ. Z-score chưa áp dụng ở bước này: ảnh lưu là uint8 sau CLAHE, chuẩn hóa
cường độ để dành cho bước nạp dữ liệu.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from data.ds12_index import (CLASS_NAME, ds1_items, ds2_classification_items,
                             ds2_segmentation_items)
from data.skull_crop import (apply_clahe, binarize_mask, crop_with_mask, mask_to_box,
                             read_image, write_png)


def process_item(item, root, out_img_dir, out_mask_dir):
    """Trả về (record | None, log_row). record là None nếu mẫu bị loại; lý do nằm trong log_row['status']."""
    log = dict(dataset=item["dataset"], source=item["source"], img=str(item["img"].relative_to(root)),
               status="ok", crop_x0="", crop_y0="", crop_x1="", crop_y1="", fallback_full="",
               expanded_for_mask="", crop_area_ratio="")
    image = read_image(item["img"])
    if image is None:
        log["status"] = "drop: ảnh hỏng"
        return None, log

    mask = None
    if item["mask"] is not None:
        raw = read_image(item["mask"], 0)
        if raw is None:
            log["status"] = "drop: mask hỏng"
            return None, log
        if raw.shape != image.shape[:2]:
            log["status"] = f"drop: lệch kích thước ảnh {image.shape[:2]} / mask {raw.shape}"
            return None, log
        mask = binarize_mask(raw)
        if item["has_tumor"] and not mask.any():
            log["status"] = "drop: mask rỗng nhưng nhãn có u"
            return None, log

    img_c, mask_c, (x0, y0, x1, y1), info = crop_with_mask(image, mask)
    h, w = image.shape[:2]
    log.update(crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1, fallback_full=info["fallback_full"],
               expanded_for_mask=info["expanded_for_mask"],
               crop_area_ratio=round((x1 - x0) * (y1 - y0) / (w * h), 4))

    out_path = out_img_dir / item["out_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(out_path, apply_clahe(img_c))

    mask_rel, box_crop, box_orig = None, None, None
    if mask_c is not None:
        out_mask = out_mask_dir / item["out_name"]
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        write_png(out_mask, mask_c * 255)
        mask_rel = out_mask.relative_to(root).as_posix()
        box_crop = mask_to_box(mask_c)
        box_orig = mask_to_box(mask)

    rec = dict(dataset=item["dataset"], source=item["source"],
               img_path=out_path.relative_to(root).as_posix(), mask_path=mask_rel,
               has_mask=mask_rel is not None, has_tumor=int(item["has_tumor"]),
               sub_class_label=int(item["label"]), gt_box=box_crop, gt_box_orig=box_orig,
               crop_box=[x0, y0, x1, y1], orig_split=item["orig_split"],
               split=None, fold=None)
    return rec, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N ảnh đầu của mỗi nhóm (chạy thử)")
    args = ap.parse_args()
    root = Path(args.root)

    ds2_seg, missing_mask = ds2_segmentation_items(root)
    groups = [
        ("DS1 navoneel", ds1_items(root), root / "dataset/ds1_cropped", None),
        ("DS2 segmentation", ds2_seg, root / "dataset/ds2_cropped/images", root / "dataset/ds2_cropped/masks"),
        ("DS2 classification", ds2_classification_items(root), root / "dataset/ds2_cropped/images",
         root / "dataset/ds2_cropped/masks"),
    ]

    ds1_records, ds2_records, logs = [], [], []
    for item in missing_mask:
        logs.append(dict(dataset="indk214", source="segmentation", img=str(item.relative_to(root)),
                         status="drop: không có mask tương ứng"))

    for name, items, img_dir, mask_dir in groups:
        items = items[: args.limit] if args.limit else items
        print(f"{name}: {len(items)} ảnh")
        for item in tqdm(items, desc=name):
            rec, log = process_item(item, root, img_dir, mask_dir or img_dir)
            logs.append(log)
            if rec is None:
                continue
            (ds1_records if item["dataset"] == "navoneel" else ds2_records).append(rec)

    (root / "dataset").mkdir(exist_ok=True)
    (root / "dataset/ds1_annotations.json").write_text(
        json.dumps(ds1_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "dataset/ds2_annotations.json").write_text(
        json.dumps(ds2_records, ensure_ascii=False, indent=2), encoding="utf-8")

    log_dir = root / "reports/build_ds1_ds2"
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in logs for k in row))
    with open(log_dir / "crop_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(logs)

    print(f"\nDS1: {len(ds1_records)} record -> dataset/ds1_annotations.json")
    print(f"DS2: {len(ds2_records)} record -> dataset/ds2_annotations.json")
    print("  theo nguồn/lớp:", dict(Counter((r['source'], CLASS_NAME[r['sub_class_label']]) for r in ds2_records)))
    drops = Counter(l["status"] for l in logs if l["status"] != "ok")
    print("Mẫu bị loại:", dict(drops) if drops else "không có")
    ok = [l for l in logs if l["status"] == "ok"]
    if ok:
        ratios = np.array([l["crop_area_ratio"] for l in ok], dtype=float)
        print(f"Khung cắt/ảnh gốc: trung bình {ratios.mean():.3f}, nhỏ nhất {ratios.min():.3f} | "
              f"fallback toàn ảnh: {sum(bool(l['fallback_full']) for l in ok)} | "
              f"mở rộng để bao mask: {sum(bool(l['expanded_for_mask']) for l in ok)}")


if __name__ == "__main__":
    main()
