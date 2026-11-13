"""Kiểm định chất lượng & phân tích thống kê (EDA) cho Dataset 3 (BRISC 2025).

Chạy từ thư mục gốc dự án:  python -m data.eda_ds3
Đầu ra trong reports/eda_ds3/:
    eda_summary.txt, class_plane_counts.csv, image_stats.csv, ds3_mask_stats.csv,
    duplicate_sha256.csv, fig_class_plane_counts.png, fig_image_sizes.png, fig_ds3_tumor_area.png
"""
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.ds3_index import (CLASS_NAME, PLANE_NAME, ds3_classification_items,
                            ds3_segmentation_items)
from data.skull_crop import binarize_mask, mask_to_box, read_image


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_image(path):
    img = read_image(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return dict(ok=False)
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    gray = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    return dict(
        ok=True,
        h=h,
        w=w,
        channels=channels,
        mean=float(gray.mean()),
        std=float(gray.std()),
        dark_frac=float((gray < 45).mean()),
        sha256=file_sha256(path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default="reports/eda_ds3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    seg_items, missing_mask, orphan_mask, seg_bad_name = ds3_segmentation_items(root)
    cls_all, cls_bad_name = ds3_classification_items(root, only_no_tumor=False)
    cls_no_tumor = [it for it in cls_all if it["label"] == 0]

    if args.limit:
        seg_items = seg_items[: args.limit]
        cls_no_tumor = cls_no_tumor[: args.limit]

    # Kiểm tra trùng khớp 1-1 giữa ảnh có u ở classification_task và segmentation_task
    cls_tumor_stems = {it["stem"] for it in cls_all if it["label"] > 0}
    seg_stems = {it["stem"] for it in seg_items}
    stem_diff_cls_seg = len(cls_tumor_stems ^ seg_stems) if not args.limit else 0

    rows, mask_rows = [], []
    size_mismatch, empty_masks, corrupted_masks = [], [], []

    for it in tqdm(seg_items, desc="DS3 segmentation (img + mask)"):
        r = inspect_image(it["img"])
        r.update(
            source="segmentation",
            orig_split=it["orig_split"],
            class_name=CLASS_NAME[it["label"]],
            plane=PLANE_NAME[it["plane_code"]],
            has_tumor=it["has_tumor"],
            stem=it["stem"],
            name=it["img"].name,
        )
        rows.append(r)
        if not r["ok"]:
            continue

        raw = read_image(it["mask"], cv2.IMREAD_GRAYSCALE)
        if raw is None:
            corrupted_masks.append(it["mask"].name)
            continue
        mh, mw = raw.shape[:2]
        match_1to1 = (mh, mw) == (r["h"], r["w"])
        if not match_1to1:
            size_mismatch.append((it["stem"], (r["h"], r["w"]), (mh, mw)))

        m = binarize_mask(raw, thresh=127)
        box = mask_to_box(m)
        is_empty = box is None
        if is_empty:
            empty_masks.append(it["stem"])

        mid = float(((raw > 8) & (raw < 247)).mean())
        mask_rows.append(dict(
            stem=it["stem"],
            orig_split=it["orig_split"],
            class_name=CLASS_NAME[it["label"]],
            plane=PLANE_NAME[it["plane_code"]],
            img_h=r["h"],
            img_w=r["w"],
            mask_h=mh,
            mask_w=mw,
            size_match=match_1to1,
            empty=is_empty,
            area_px=int(m.sum()),
            area_frac=float(m.mean()),
            box_w=(box[2] - box[0]) if box else 0,
            box_h=(box[3] - box[1]) if box else 0,
            n_components=int(cv2.connectedComponents(m)[0] - 1) if not is_empty else 0,
            uncertain_pixel_frac=mid,
            bg_level=int(raw.min()),
            fg_level=int(raw.max()),
        ))

    for it in tqdm(cls_no_tumor, desc="DS3 classification (no_tumor)"):
        r = inspect_image(it["img"])
        r.update(
            source="classification_no_tumor",
            orig_split=it["orig_split"],
            class_name=CLASS_NAME[it["label"]],
            plane=PLANE_NAME[it["plane_code"]],
            has_tumor=it["has_tumor"],
            stem=it["stem"],
            name=it["img"].name,
        )
        rows.append(r)

    df = pd.DataFrame(rows)
    corrupted_imgs = df[~df["ok"]]
    df = df[df["ok"]].copy()
    mdf = pd.DataFrame(mask_rows)

    # Phân tích trùng lặp SHA256 (rò rỉ dữ liệu tiềm ẩn giữa train/test gốc)
    by_sha = defaultdict(list)
    for _, row in df.iterrows():
        by_sha[row["sha256"]].append(row)
    dup_rows = []
    cross_split_groups = 0
    for sha, grp in by_sha.items():
        if len(grp) > 1:
            splits = sorted({g["orig_split"] for g in grp})
            if len(splits) > 1:
                cross_split_groups += 1
            dup_rows.append(dict(
                sha256=sha,
                count=len(grp),
                cross_orig_split=len(splits) > 1,
                classes=",".join(sorted({g["class_name"] for g in grp})),
                planes=",".join(sorted({g["plane"] for g in grp})),
                stems=";".join(g["stem"] for g in grp),
            ))
    dup_df = pd.DataFrame(dup_rows)

    # Lưu CSV
    counts = (
        df.groupby(["source", "orig_split", "class_name", "plane"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(out / "class_plane_counts.csv", index=False)
    df.drop(columns=["ok"]).to_csv(out / "image_stats.csv", index=False)
    mdf.to_csv(out / "ds3_mask_stats.csv", index=False)
    dup_df.to_csv(out / "duplicate_sha256.csv", index=False)

    # Biểu đồ 1: Phân bố lớp & mặt cắt giải phẫu
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    vc_cls = df["class_name"].value_counts()
    axes[0].bar(vc_cls.index, vc_cls.values, color="steelblue")
    axes[0].set_title(f"Phân bố lớp BRISC 2025 (n={len(df)})")
    axes[0].set_ylabel("Số lát cắt MRI")
    axes[0].tick_params(axis="x", rotation=20)

    ct_plane = df.groupby(["class_name", "plane"]).size().unstack(fill_value=0)
    ct_plane.plot(kind="bar", ax=axes[1], colormap="viridis")
    axes[1].set_title("Phân bố theo lớp & mặt phẳng cắt (Axial/Coronal/Sagittal)")
    axes[1].set_ylabel("Số lát cắt MRI")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out / "fig_class_plane_counts.png", dpi=140)
    plt.close()

    # Biểu đồ 2: Kích thước ảnh
    plt.figure(figsize=(6, 5))
    for cname, g in df.groupby("class_name"):
        plt.scatter(g["w"], g["h"], s=8, alpha=0.45, label=cname)
    plt.xlabel("Chiều rộng (px)")
    plt.ylabel("Chiều cao (px)")
    plt.title(f"Phân bố kích thước ảnh BRISC 2025 ({df[['w', 'h']].drop_duplicates().shape[0]} kích thước)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_image_sizes.png", dpi=140)
    plt.close()

    # Biểu đồ 3: Diện tích khối u từ mask
    if not mdf.empty:
        plt.figure(figsize=(7.5, 4.2))
        for cname, g in mdf.groupby("class_name"):
            plt.hist(g["area_frac"] * 100, bins=40, alpha=0.5, label=cname)
        plt.xlabel("Diện tích khối u (% diện tích ảnh)")
        plt.ylabel("Số cặp ảnh - mask")
        plt.title(f"Phân bố diện tích u trên BRISC 2025 Segmentation (n={len(mdf)})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "fig_ds3_tumor_area.png", dpi=140)
        plt.close()

    # Văn bản tổng kết kiểm định chất lượng
    L = [
        "== KIỂM ĐỊNH CHẤT LƯỢNG & THỐNG KÊ DATASET 3 (BRISC 2025) ==",
        f"Tổng số mẫu duy nhất theo tác vụ: {len(df)} lát cắt T1 MRI "
        f"(Segmentation có mask: {len(seg_items)}, Classification no_tumor: {len(cls_no_tumor)})",
        f"Đối chiếu classification_task (6000 ảnh) vs segmentation_task (4793 cặp):",
        f"   - Số ảnh có u (gl, me, pi) lệch tên giữa 2 thư mục: {stem_diff_cls_seg}",
        f"   - Ảnh thiếu mask tương ứng: {len(missing_mask)} | Mask mồ côi thiếu ảnh: {len(orphan_mask)}",
        f"   - Tên file sai quy cách: segmentation={len(seg_bad_name)}, classification={len(cls_bad_name)}",
        f"   - Ảnh lỗi/hỏng không đọc được: {len(corrupted_imgs)} | Mask lỗi/hỏng: {len(corrupted_masks)}",
        f"   - Lệch kích thước 1-1 giữa Ảnh và Mask: {len(size_mismatch)}",
        f"   - Mask rỗng sai quy cách (nhãn có u nhưng vùng u = 0 px): {len(empty_masks)}",
        f"Phát hiện trùng lặp nội dung (SHA256): {len(dup_df)} nhóm ({int(dup_df['count'].sum() - len(dup_df)) if not dup_df.empty else 0} ảnh trùng dư), "
        f"trong đó có {cross_split_groups} nhóm bị rò rỉ trực tiếp giữa train và test gốc của BRISC 2025!",
        "",
        "== THỐNG KÊ HÌNH ẢNH & MẶT NẠ (MASK) ==",
        f"Số kích thước (W, H) khác nhau: {df[['w', 'h']].drop_duplicates().shape[0]} "
        f"(phổ biến nhất: {dict(Counter(zip(df['w'], df['h'])).most_common(5))})",
        f"Số kênh màu: {dict(Counter(df['channels']))} | Độ sáng TB: {df['mean'].mean():.1f} | Tỷ lệ pixel tối (<45): {df['dark_frac'].mean():.2f}",
        f"Phân bố lớp tổng thể: {df['class_name'].value_counts().to_dict()}",
        f"Phân bố mặt phẳng giải phẫu: {df['plane'].value_counts().to_dict()}",
    ]
    if not mdf.empty:
        L.append(
            f"Đặc trưng mức xám của Mask BRISC 2025: nền nhỏ nhất {dict(Counter(mdf['bg_level']).most_common(4))}, "
            f"tỷ lệ mask có pixel biên trung gian (8<v<247): {mdf['uncertain_pixel_frac'].gt(0).mean():.1%} "
            f"=> cần nhị phân hóa bằng ngưỡng 127 (binarize_mask) để tránh nhiễu nén ảnh."
        )
        for cname, g in mdf.groupby("class_name"):
            L.append(
                f"   - {cname}: n={len(g)}, diện tích u TB {g['area_frac'].mean() * 100:.2f}% "
                f"(min {g['area_frac'].min() * 100:.2f}%, max {g['area_frac'].max() * 100:.2f}%), "
                f"số thành phần liên thông TB {g['n_components'].mean():.2f}"
            )

    text = "\n".join(L)
    (out / "eda_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nĐã lưu báo cáo kiểm định vào: {out}")


if __name__ == "__main__":
    main()

