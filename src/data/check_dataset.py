"""
check_dataset.py - Kiem tra chat luong dataset da tao boi build_dataset.py.

    python src/data/check_dataset.py --out_dir "E:/brats_processed"

Kiem tra:
  1. Khong ca nao nam o 2 tap khac nhau (khong ro ri du lieu).
  2. Thong ke so lat u / lanh theo tung tap.
  3. Voi mot mau ngau nhien cac lat: shape anh (H,W,3), gia tri huu han,
     mask chi gom 0/1, has_tumor va tumor_area khop mask, bbox khop mask.
  4. Z-score: mean/std (tren voxel khac 0) cua vai ca ngau nhien.
  5. Ve luoi anh FLAIR + mask + bbox -> check_samples.png de xem bang mat.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_slice(out_dir, rel):
    with np.load(os.path.join(out_dir, rel)) as z:
        return z["image"].astype(np.float32), z["mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Thu muc dataset da xu ly")
    ap.add_argument("--max_check", type=int, default=2000,
                    help="So lat kiem tra chi tiet (ngau nhien)")
    ap.add_argument("--n_samples", type=int, default=12, help="So anh trong luoi minh hoa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta_path = os.path.join(args.out_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        sys.exit(f"Khong thay {meta_path}. Hay chay build_dataset.py truoc.")
    meta = pd.read_csv(meta_path)
    errors = []

    # 1. ro ri du lieu
    nsplit = meta.groupby("case_id")["split"].nunique()
    leak = nsplit[nsplit > 1]
    if len(leak):
        errors.append(f"RO RI: {len(leak)} ca nam o nhieu hon 1 tap: {leak.index[:5].tolist()}")
    else:
        print("[OK] Khong co ca nao nam o 2 tap khac nhau.")

    # 2. thong ke
    tab = meta.groupby("split")["has_tumor"].agg(slices="size", tumor="sum")
    tab["healthy"] = tab["slices"] - tab["tumor"]
    tab["tumor_ratio"] = (tab["tumor"] / tab["slices"]).round(3)
    tab["cases"] = meta.groupby("split")["case_id"].nunique()
    print("\nThong ke theo tap:")
    print(tab[["cases", "slices", "tumor", "healthy", "tumor_ratio"]].to_string())

    # 3. kiem tra chi tiet tren mau ngau nhien
    n_check = min(args.max_check, len(meta))
    sample = meta.sample(n_check, random_state=args.seed)
    shape_set = set()
    for _, r in sample.iterrows():
        try:
            img, mask = load_slice(args.out_dir, r["path"])
        except Exception as e:
            errors.append(f"Khong doc duoc {r['path']}: {e!r}")
            continue
        shape_set.add(img.shape)
        tag = r["path"]
        if img.ndim != 3 or img.shape[2] != 3:
            errors.append(f"{tag}: anh phai la (H,W,3), dang la {img.shape}")
        if img.shape[:2] != mask.shape:
            errors.append(f"{tag}: shape anh {img.shape[:2]} != mask {mask.shape}")
        if not np.isfinite(img).all():
            errors.append(f"{tag}: anh co NaN/Inf")
        elif np.abs(img).max() > 60:
            errors.append(f"{tag}: gia tri bat thuong, max|x|={np.abs(img).max():.1f}")
        if not set(np.unique(mask)).issubset({0, 1}):
            errors.append(f"{tag}: mask khong nhi phan")
        area = int(mask.sum())
        if area != int(r["tumor_area"]):
            errors.append(f"{tag}: tumor_area metadata {r['tumor_area']} != mask {area}")
        if bool(r["has_tumor"]) != (area > 0):
            errors.append(f"{tag}: has_tumor khong khop mask")
        if r["has_tumor"] == 1 and area > 0:
            ys, xs = np.where(mask)
            got = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            exp = tuple(int(r[c]) for c in ("x_min", "y_min", "x_max", "y_max"))
            if got != exp:
                errors.append(f"{tag}: bbox metadata {exp} != tinh lai tu mask {got}")
        elif pd.notna(r["x_min"]):
            errors.append(f"{tag}: lat khong u nhung co bbox")
    print(f"\n[Da kiem tra chi tiet {n_check} lat; shape gap: {sorted(shape_set)}]")

    # 4. Z-score tren vai ca
    rng = np.random.RandomState(args.seed)
    case_ids = meta["case_id"].unique()
    pick = rng.choice(case_ids, size=min(5, len(case_ids)), replace=False)
    print("\nZ-score tren cac lat da luu (mean ~ 0, std ~ 1 la tot; lech nhe do da bo bot lat):")
    for cid in pick:
        vals = [[], [], []]
        for rel in meta.loc[meta["case_id"] == cid, "path"]:
            img, _ = load_slice(args.out_dir, rel)
            for c in range(3):
                ch = img[:, :, c]
                vals[c].append(ch[ch != 0])
        row = []
        for c, name in enumerate(("FLAIR", "T1ce", "T2")):
            v = np.concatenate(vals[c]) if vals[c] else np.array([0.0])
            row.append(f"{name}: mean={v.mean():+.2f} std={v.std():.2f}")
            if abs(v.mean()) > 0.7 or not (0.4 < v.std() < 1.8):
                errors.append(f"{cid} {name}: Z-score lech nhieu (mean={v.mean():.2f}, std={v.std():.2f})")
        print(f"  {cid}: " + " | ".join(row))

    # 5. luoi anh minh hoa
    half = max(args.n_samples // 2, 1)
    tum = meta[meta["has_tumor"] == 1]
    hea = meta[meta["has_tumor"] == 0]
    chosen = pd.concat([
        tum.sample(min(half, len(tum)), random_state=args.seed),
        hea.sample(min(args.n_samples - half, len(hea)), random_state=args.seed),
    ])
    n = len(chosen)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, chosen.iterrows()):
        img, mask = load_slice(args.out_dir, r["path"])
        ax.imshow(img[:, :, 0], cmap="gray")
        if mask.any():
            ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.45)
            ax.add_patch(Rectangle((r["x_min"] - 0.5, r["y_min"] - 0.5),
                                   r["x_max"] - r["x_min"] + 1, r["y_max"] - r["y_min"] + 1,
                                   fill=False, edgecolor="cyan", linewidth=1.6))
        ax.set_title(f"{r['split']} {r['case_id'][-3:]} z={r['z']} "
                     f"{'U' if r['has_tumor'] else 'lanh'}", fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "check_samples.png")
    try:
        plt.savefig(fig_path, dpi=120)
        print(f"\nDa luu anh minh hoa: {fig_path}")
    except Exception as e:
        print(f"[Canh bao] Khong luu duoc anh minh hoa: {e!r}")
    plt.close()

    print("\n" + "=" * 60)
    if errors:
        print(f"PHAT HIEN {len(errors)} VAN DE (hien toi da 25):")
        for e in errors[:25]:
            print("  -", e)
        sys.exit(1)
    print("TAT CA KIEM TRA DEU DAT.")


if __name__ == "__main__":
    main()
