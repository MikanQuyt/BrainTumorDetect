"""Kiểm tra đầu ra của build_ds1_ds2: tính nhất quán hộp/mask và lưới ảnh minh họa.

Chạy:  python -m data.check_ds1_ds2
Kiểm tra: ảnh/mask tồn tại và cùng kích thước; gt_box tính lại từ mask đã cắt trùng khớp; gt_box nằm trong ảnh;
gt_box_orig - góc khung cắt = gt_box; mask nhị phân {0, 255}. Lưu reports/build_ds1_ds2/check_samples.png.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from data.skull_crop import mask_to_box, read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--n_samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)

    ds2 = json.loads((root / "dataset/ds2_annotations.json").read_text(encoding="utf-8"))
    ds1 = json.loads((root / "dataset/ds1_annotations.json").read_text(encoding="utf-8"))
    errors = []

    seg = [r for r in ds2 if r["has_mask"]]
    for r in seg:
        img = read_image(root / r["img_path"])
        m = read_image(root / r["mask_path"], 0)
        if img is None or m is None:
            errors.append(f"{r['img_path']}: không đọc được ảnh/mask")
            continue
        h, w = img.shape[:2]
        if m.shape != (h, w):
            errors.append(f"{r['img_path']}: ảnh {img.shape[:2]} != mask {m.shape}")
        if not set(np.unique(m)).issubset({0, 255}):
            errors.append(f"{r['img_path']}: mask không nhị phân {{0,255}}")
        box = mask_to_box(m > 127)
        gb = r["gt_box"]
        if box != gb:
            errors.append(f"{r['img_path']}: gt_box {gb} != tính lại từ mask {box}")
        if gb and not (0 <= gb[0] < gb[2] <= w and 0 <= gb[1] < gb[3] <= h):
            errors.append(f"{r['img_path']}: gt_box {gb} ngoài ảnh {w}x{h}")
        cx0, cy0 = r["crop_box"][:2]
        go = r["gt_box_orig"]
        if go and [go[0] - cx0, go[1] - cy0, go[2] - cx0, go[3] - cy0] != gb:
            errors.append(f"{r['img_path']}: gt_box_orig không khớp gt_box sau khi trừ góc khung cắt")
        if r["has_tumor"] != 1 or not r["sub_class_label"] in (1, 2, 3):
            errors.append(f"{r['img_path']}: nhãn không hợp lệ")

    for r in ds1 + [r for r in ds2 if not r["has_mask"]]:
        if read_image(root / r["img_path"]) is None:
            errors.append(f"{r['img_path']}: không đọc được ảnh")
        if r["gt_box"] is not None:
            errors.append(f"{r['img_path']}: ảnh không mask nhưng có gt_box")

    print(f"DS1: {len(ds1)} ảnh | DS2: {len(ds2)} ảnh ({len(seg)} có mask/hộp, {len(ds2) - len(seg)} chỉ phân loại)")

    # Lưới minh họa: ảnh đã cắt + mask (đỏ) + gt_box (xanh lá); thêm vài ảnh DS1
    rng = np.random.default_rng(args.seed)
    picks = [seg[i] for i in rng.choice(len(seg), min(args.n_samples, len(seg)), replace=False)]
    tiles = []
    for r in picks:
        img = read_image(root / r["img_path"])
        m = read_image(root / r["mask_path"], 0) > 127
        vis = img.copy()
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = r["gt_box"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        tiles.append(cv2.resize(vis, (256, 256)))
    for r in rng.choice(ds1, min(4, len(ds1)), replace=False):
        tiles.append(cv2.resize(read_image(root / r["img_path"]), (256, 256)))
    while len(tiles) % 4:
        tiles.append(np.zeros((256, 256, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = root / "reports/build_ds1_ds2"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", grid)[1].tofile(str(out / "check_samples.png"))
    print(f"Đã lưu lưới ảnh minh họa: {out / 'check_samples.png'}")

    print("=" * 60)
    if errors:
        print(f"PHÁT HIỆN {len(errors)} VẤN ĐỀ (hiện tối đa 25):")
        for e in errors[:25]:
            print("  -", e)
        sys.exit(1)
    print("TẤT CẢ KIỂM TRA ĐỀU ĐẠT.")


if __name__ == "__main__":
    main()

