"""EDA cho Dataset 1 (navoneel) và Dataset 2 (indk214: Segmentation & Classification).

Chạy từ thư mục gốc dự án:  python -m data.eda_ds1_ds2
Đầu ra trong reports/eda_ds1_ds2/:
    eda_summary.txt, class_counts.csv, image_stats.csv, ds2_mask_stats.csv,
    fig_class_counts.png, fig_image_sizes.png, fig_ds2_tumor_area.png
Số liệu đọc trực tiếp từ ảnh/mask gốc (trước khi cắt hộp sọ).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.ds12_index import (CLASS_NAME, ds1_items, ds2_classification_items,
                             ds2_segmentation_items)
from data.skull_crop import binarize_mask, mask_to_box, read_image


def image_row(item):
    img = read_image(item["img"], cv2.IMREAD_UNCHANGED)
    if img is None:
        return dict(ok=False)
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    gray = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return dict(ok=True, h=h, w=w, channels=channels, mean=float(gray.mean()), std=float(gray.std()),
                dark_frac=float((gray < 45).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="reports/eda_ds1_ds2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.root) / args.out
    out.mkdir(parents=True, exist_ok=True)

    seg_items, missing_mask = ds2_segmentation_items(root)
    groups = {"DS1 navoneel": ds1_items(root), "DS2 segmentation": seg_items,
              "DS2 classification": ds2_classification_items(root)}

    rows = []
    for gname, items in groups.items():
        for it in tqdm(items[: args.limit] if args.limit else items, desc=gname):
            r = image_row(it)
            r.update(group=gname, class_name=CLASS_NAME[it["label"]], has_tumor=it["has_tumor"],
                     split=it["orig_split"] or "-", name=it["img"].name)
            rows.append(r)
    df = pd.DataFrame(rows)
    corrupted = df[~df["ok"]]
    df = df[df["ok"]].copy()

    # ---- số lượng & phân bố lớp
    counts = df.groupby(["group", "split", "class_name"]).size().rename("n").reset_index()
    counts.to_csv(out / "class_counts.csv", index=False)
    df.drop(columns=["ok"]).to_csv(out / "image_stats.csv", index=False)

    # ---- thống kê mask DS2 segmentation (diện tích u, hộp, tính sạch của mask)
    mask_rows = []
    for it in tqdm(seg_items[: args.limit] if args.limit else seg_items, desc="DS2 masks"):
        raw = read_image(it["mask"], 0)
        img = read_image(it["img"])
        if raw is None or img is None:
            continue
        m = binarize_mask(raw)
        box = mask_to_box(m)
        mid = float(((raw > 8) & (raw < 247)).mean())
        mask_rows.append(dict(class_name=CLASS_NAME[it["label"]], name=it["img"].name,
                              size_match=raw.shape == img.shape[:2], empty=box is None,
                              area_px=int(m.sum()), area_frac=float(m.mean()),
                              box_w=(box[2] - box[0]) if box else 0, box_h=(box[3] - box[1]) if box else 0,
                              n_components=int(cv2.connectedComponents(m)[0] - 1),
                              uncertain_pixel_frac=mid, bg_level=int(raw.min())))
    mdf = pd.DataFrame(mask_rows)
    mdf.to_csv(out / "ds2_mask_stats.csv", index=False)

    # ---- hình
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (gname, g) in zip(axes, df.groupby("group")):
        vc = g["class_name"].value_counts()
        ax.bar(vc.index, vc.values, color="steelblue")
        ax.set_title(f"{gname} (n={len(g)})")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out / "fig_class_counts.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6, 5))
    for gname, g in df.groupby("group"):
        plt.scatter(g["w"], g["h"], s=6, alpha=0.4, label=gname)
    plt.xlabel("Chiều rộng (px)")
    plt.ylabel("Chiều cao (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_image_sizes.png", dpi=130)
    plt.close()

    if not mdf.empty:
        plt.figure(figsize=(7, 4))
        for cname, g in mdf.groupby("class_name"):
            plt.hist(g["area_frac"] * 100, bins=40, alpha=0.5, label=cname)
        plt.xlabel("Diện tích u (% ảnh)")
        plt.ylabel("Số ảnh")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "fig_ds2_tumor_area.png", dpi=130)
        plt.close()

    # ---- văn bản tổng kết
    L = ["== TỔNG QUAN =="]
    for gname, g in df.groupby("group"):
        L.append(f"{gname}: {len(g)} ảnh | kích thước: "
                 f"{dict(Counter(zip(g['w'], g['h'])).most_common(3))} | kênh: {dict(Counter(g['channels']))}")
        L.append(f"   phân bố lớp: {g['class_name'].value_counts().to_dict()}")
        L.append(f"   độ sáng TB {g['mean'].mean():.1f} | tỷ lệ pixel tối (<45) TB {g['dark_frac'].mean():.2f}")
    L.append(f"Ảnh hỏng/không đọc được: {len(corrupted)}")
    ds1 = df[df["group"] == "DS1 navoneel"]
    L.append(f"DS1 tổng {len(ds1)} ảnh (kỳ vọng 253): có u {int(ds1['has_tumor'].sum())}, "
             f"không u {int((1 - ds1['has_tumor']).sum())}")
    L.append(f"DS1 có {ds1[['w', 'h']].drop_duplicates().shape[0]} kích thước ảnh khác nhau (cần resize đồng nhất)")
    cls = df[df["group"] == "DS2 classification"]
    L.append(f"DS2 classification theo tập gốc: {cls['split'].value_counts().to_dict()}")
    L.append(f"DS2 segmentation: ảnh thiếu mask {len(missing_mask)}")
    if not mdf.empty:
        L.append(f"DS2 mask: lệch kích thước {int((~mdf['size_match']).sum())}, rỗng {int(mdf['empty'].sum())}, "
                 f"nền mask (giá trị nhỏ nhất) {dict(Counter(mdf['bg_level']).most_common(4))}")
        L.append(f"   mask có pixel trung gian (8<v<247): {mdf['uncertain_pixel_frac'].gt(0).mean():.1%} | "
                 f"nền mask khác 0: {mdf['bg_level'].gt(0).mean():.1%} (thường = 3) "
                 f"=> nhị phân hóa bằng ngưỡng 127, KHÔNG dùng '> 0' (sẽ coi cả ảnh là u)")
        for cname, g in mdf.groupby("class_name"):
            L.append(f"   {cname}: n={len(g)}, diện tích u TB {g['area_frac'].mean() * 100:.2f}% "
                     f"(min {g['area_frac'].min() * 100:.2f}%, max {g['area_frac'].max() * 100:.2f}%), "
                     f"số thành phần liên thông TB {g['n_components'].mean():.2f}")
    text = "\n".join(L)
    (out / "eda_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nĐã lưu vào: {out}")


if __name__ == "__main__":
    main()

