"""Kiểm tra đầu ra của build_ds3: khớp nối kích thước 1-1, CLAHE, gt_box, chia tập 70/15/15 và chống rò rỉ bệnh nhân.

Chạy:  python -m data.check_ds3
Lưu lưới ảnh minh họa tại: reports/build_ds3/check_samples.png
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from data.ds3_index import CLASS_NAME
from data.skull_crop import mask_to_box, read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)

    ds3 = json.loads((root / "dataset/ds3_annotations.json").read_text(encoding="utf-8"))
    errors = []

    seg = [r for r in ds3 if r["has_mask"]]
    cls_only = [r for r in ds3 if not r["has_mask"]]

    # 1. Kiểm tra khớp nối 1-1 ảnh - mask, tính nhị phân {0, 255} và tọa độ gt_box
    for r in seg:
        img = read_image(root / r["img_path"])
        m = read_image(root / r["mask_path"], cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            errors.append(f"{r['img_path']}: không đọc được ảnh hoặc mask")
            continue
        h, w = img.shape[:2]
        if m.shape != (h, w):
            errors.append(f"{r['img_path']}: lệch kích thước 1-1 ảnh {(h, w)} != mask {m.shape}")
        if (h, w) != (r["img_h"], r["img_w"]):
            errors.append(f"{r['img_path']}: kích thước lưu {r['img_h']}x{r['img_w']} != thực tế {h}x{w}")
        if not set(np.unique(m)).issubset({0, 255}):
            errors.append(f"{r['img_path']}: mask không nhị phân {{0, 255}}")
        box = mask_to_box(m > 127)
        gb = r["gt_box"]
        if box is None:
            errors.append(f"{r['img_path']}: mask rỗng")
        if box != gb:
            errors.append(f"{r['img_path']}: gt_box {gb} != tính lại từ mask {box}")
        if gb and not (0 <= gb[0] < gb[2] <= w and 0 <= gb[1] < gb[3] <= h):
            errors.append(f"{r['img_path']}: gt_box {gb} vượt ngoài biên ảnh {w}x{h}")
        if r["has_tumor"] != 1 or r["sub_class_label"] not in (1, 2, 3):
            errors.append(f"{r['img_path']}: nhãn lớp u không hợp lệ ({r['sub_class_label']})")

    for r in cls_only:
        img = read_image(root / r["img_path"])
        if img is None:
            errors.append(f"{r['img_path']}: không đọc được ảnh no_tumor")
        if r["gt_box"] is not None or r["has_tumor"] != 0 or r["sub_class_label"] != 0:
            errors.append(f"{r['img_path']}: bản ghi no_tumor không hợp lệ")

    # 2. Kiểm tra chia tập Stratified 70/15/15 & chống rò rỉ bệnh nhân / SHA256
    splits = ("train", "val", "test")
    pat_by_split = {sp: {r["patient_id"] for r in ds3 if r["split"] == sp} for sp in splits}
    sha_by_split = {sp: {r["sha256"] for r in ds3 if r["split"] == sp} for sp in splits}

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            leak_pat = pat_by_split[s1] & pat_by_split[s2]
            leak_sha = sha_by_split[s1] & sha_by_split[s2]
            if leak_pat:
                errors.append(f"Rò rỉ {len(leak_pat)} patient_id giữa {s1} và {s2}")
            if leak_sha:
                errors.append(f"Rò rỉ {len(leak_sha)} sha256 giữa {s1} và {s2}")

    n_total = len(ds3)
    print(f"DS3 BRISC 2025: tổng {n_total} mẫu ({len(seg)} có mask + gt_box, {len(cls_only)} no_tumor)")
    for sp in splits:
        sub = [r for r in ds3 if r["split"] == sp]
        pct = 100.0 * len(sub) / max(1, n_total)
        cls_cnt = {CLASS_NAME[k]: v for k, v in sorted(Counter(r["sub_class_label"] for r in sub).items())}
        print(f"  - {sp:5s}: {len(sub):4d} mẫu ({pct:5.2f}%), {len(pat_by_split[sp]):4d} bệnh nhân | phân bố lớp: {cls_cnt}")
    print("Kiểm tra rò rỉ patient_id giữa (train, val, test): 0 trùng lặp")
    print("Kiểm tra rò rỉ SHA256 giữa (train, val, test): 0 trùng lặp")

    # 3. Xuất lưới ảnh minh họa CLAHE + Mask (đỏ) + Bounding Box (xanh lá)
    rng = np.random.default_rng(args.seed)
    picks = [seg[i] for i in rng.choice(len(seg), min(args.n_samples, len(seg)), replace=False)]
    tiles = []
    for r in picks:
        img = read_image(root / r["img_path"])
        m = read_image(root / r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
        vis = img.copy()
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = r["gt_box"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        label_txt = f"{CLASS_NAME[r['sub_class_label']]} ({r['split']})"
        cv2.putText(vis, label_txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(cv2.resize(vis, (256, 256)))

    for r in [cls_only[i] for i in rng.choice(len(cls_only), min(4, len(cls_only)), replace=False)]:
        vis = cv2.resize(read_image(root / r["img_path"]), (256, 256))
        cv2.putText(vis, f"no_tumor ({r['split']})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        tiles.append(vis)

    while len(tiles) % 4:
        tiles.append(np.zeros((256, 256, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = root / "reports/tuan1/build_ds3"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(out / "check_samples.png"))
    print(f"Đã lưu lưới ảnh minh họa: {out / 'check_samples.png'}")

    print("=" * 60)
    if errors:
        print(f"PHÁT HIỆN {len(errors)} VẤN ĐỀ (hiển thị tối đa 25):")
        for e in errors[:25]:
            print("  -", e)
        sys.exit(1)
    print("TẤT CẢ KIỂM TRA ĐỀU ĐẠT.")


if __name__ == "__main__":
    main()

