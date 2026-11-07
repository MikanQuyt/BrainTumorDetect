"""
eda_brats.py - Khao sat (EDA) tap du lieu BraTS2020.

Cach chay (trong moi truong da cai nibabel, numpy, pandas, matplotlib, tqdm):

    python eda_brats.py --data_dir "D:/datasets/BraTS2020" --out_dir reports

    # chay thu nhanh tren 5 ca dau:
    python eda_brats.py --data_dir "D:/datasets/BraTS2020" --limit 5

Script tu dong tim moi thu muc ca (thu muc chua file *flair*.nii / .nii.gz),
nen KHONG phu thuoc cau truc thu muc cua Kaggle hay CBICA.

Dau ra (trong --out_dir):
    eda_summary.csv        mot dong / ca
    eda_overview.txt       tom tat toan tap (in ra man hinh luon)
    fig_*.png              cac bieu do phan bo
    fig_sample_case.png    anh minh hoa 1 ca (3 kenh + mask WT)
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nibabel as nib  # noqa: E402
from tqdm import tqdm  # noqa: E402

MODALITIES = ["flair", "t1ce", "t2"]          # 3 kenh dung cho de tai
ALL_MODS = ["flair", "t1", "t1ce", "t2"]      # khao sat ca T1 cho day du
NII_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)


# ---------------------------------------------------------------- tim file
def stem_of(fname):
    return NII_RE.sub("", fname).lower()


def classify_file(fname):
    """Tra ve 'flair' | 't1ce' | 't1' | 't2' | 'seg' | None.
    Xu ly ca dac biet: file seg cua ca 355 (Kaggle) ten 'W39_..._Segm.nii'."""
    if not NII_RE.search(fname):
        return None
    s = stem_of(fname)
    if "seg" in s:           # 'seg' hoac 'segm'
        return "seg"
    for key in ("flair", "t1ce", "t1", "t2"):
        if s.endswith(key):
            return key
    return None


def find_cases(data_dir):
    """Quet de quy, moi thu muc co du file flair duoc coi la 1 ca."""
    cases = {}
    for root, _dirs, files in os.walk(data_dir):
        found = {}
        for f in files:
            kind = classify_file(f)
            if kind and kind not in found:
                found[kind] = os.path.join(root, f)
        if "flair" in found:
            cases[os.path.basename(root)] = found
    return dict(sorted(cases.items()))


# ---------------------------------------------------------------- doc anh
def load_vol(path):
    img = nib.load(path)
    return img, np.asarray(img.dataobj, dtype=np.float32)


def load_seg(path):
    img = nib.load(path)
    arr = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    return img, arr


def brain_stats(vol):
    m = vol > 0
    if not m.any():
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan, nz=0)
    v = vol[m]
    return dict(min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), std=float(v.std()), nz=int(m.sum()))


# ---------------------------------------------------------------- 1 ca
def analyze_case(case_id, paths):
    row = {"case_id": case_id, "has_seg": "seg" in paths}
    shapes, vols = {}, {}

    for mod in ALL_MODS:
        if mod not in paths:
            row[f"{mod}_missing"] = True
            continue
        img, vol = load_vol(paths[mod])
        shapes[mod] = vol.shape
        vols[mod] = vol
        if mod == "flair":
            row["dtype_on_disk"] = str(img.get_data_dtype())
            row["spacing_mm"] = "x".join(f"{z:.2f}" for z in img.header.get_zooms()[:3])
            row["orientation"] = "".join(nib.aff2axcodes(img.affine))
        st = brain_stats(vol)
        for k in ("min", "max", "mean", "std"):
            row[f"{mod}_{k}"] = st[k]

    ref_shape = shapes.get("flair")
    row["shape"] = "x".join(map(str, ref_shape)) if ref_shape else ""
    row["shape_consistent"] = len(set(shapes.values())) == 1
    n_slices = ref_shape[2] if ref_shape else 0
    row["n_slices_z"] = n_slices

    flair = vols["flair"]
    brain_mask = np.zeros(ref_shape, dtype=bool)
    for mod in MODALITIES:
        if mod in vols and vols[mod].shape == ref_shape:
            brain_mask |= vols[mod] > 0
    brain_per_slice = brain_mask.sum(axis=(0, 1))
    row["brain_voxels"] = int(brain_mask.sum())
    row["brain_slices"] = int((brain_per_slice > 0).sum())
    row["empty_slices"] = int((brain_per_slice == 0).sum())

    if "seg" in paths:
        _, seg = load_seg(paths["seg"])
        row["seg_shape_match"] = seg.shape == ref_shape
        labels = sorted(int(x) for x in np.unique(seg))
        row["labels"] = ",".join(map(str, labels))
        row["unexpected_labels"] = ",".join(
            str(x) for x in labels if x not in (0, 1, 2, 4))
        for lb, name in ((1, "ncr_net"), (2, "edema"), (4, "et")):
            row[f"vox_label{lb}_{name}"] = int((seg == lb).sum())
        wt = seg > 0
        voxel_vol = float(np.prod(nib.load(paths["seg"]).header.get_zooms()[:3]))
        row["wt_voxels"] = int(wt.sum())
        row["wt_volume_mm3"] = float(wt.sum()) * voxel_vol
        tumor_per_slice = wt.sum(axis=(0, 1))
        row["tumor_slices"] = int((tumor_per_slice > 0).sum())
        row["max_tumor_area_px"] = int(tumor_per_slice.max())
        row["tumor_slice_ratio"] = (
            row["tumor_slices"] / row["brain_slices"] if row["brain_slices"] else np.nan)
        row["healthy_brain_slices"] = row["brain_slices"] - row["tumor_slices"]
        # tumor ngoai vung nao (bat thuong)?
        row["tumor_outside_brain_vox"] = int((wt & ~brain_mask).sum())
    return row


# ---------------------------------------------------------------- ve hinh
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def make_figures(df, out_dir):
    d = df[df["has_seg"]]
    if d.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(d["tumor_slices"], bins=30)
    plt.xlabel("So lat cat co u / ca")
    plt.ylabel("So ca")
    plt.title("Phan bo so lat cat co u")
    savefig(os.path.join(out_dir, "fig_tumor_slices_per_case.png"))

    plt.figure(figsize=(6, 4))
    plt.hist(d["wt_volume_mm3"] / 1000.0, bins=30)
    plt.xlabel("The tich Whole Tumor (cm3)")
    plt.ylabel("So ca")
    plt.title("Phan bo the tich WT")
    savefig(os.path.join(out_dir, "fig_wt_volume.png"))

    plt.figure(figsize=(6, 4))
    plt.hist(d["tumor_slice_ratio"].dropna(), bins=30)
    plt.xlabel("Ti le lat co u / lat co nao")
    plt.ylabel("So ca")
    plt.title("Ti le lat u so voi lat co nao")
    savefig(os.path.join(out_dir, "fig_tumor_slice_ratio.png"))

    plt.figure(figsize=(7, 4))
    for mod in ALL_MODS:
        col = f"{mod}_mean"
        if col in df:
            plt.hist(df[col].dropna(), bins=30, alpha=0.5, label=mod)
    plt.xlabel("Cuong do trung binh (vung nao) / ca")
    plt.ylabel("So ca")
    plt.legend()
    plt.title("Cuong do trung binh theo modality (can Z-score)")
    savefig(os.path.join(out_dir, "fig_intensity_mean_per_modality.png"))


def sample_figure(cases, df, out_dir):
    d = df[df["has_seg"]]
    if d.empty:
        return
    case_id = d.sort_values("tumor_slices").iloc[len(d) // 2]["case_id"]
    paths = cases[case_id]
    _, seg = load_seg(paths["seg"])
    z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, mod in zip(axes[:3], MODALITIES):
        _, vol = load_vol(paths[mod])
        ax.imshow(np.rot90(vol[:, :, z]), cmap="gray")
        ax.set_title(mod.upper())
        ax.axis("off")
    _, flair = load_vol(paths["flair"])
    axes[3].imshow(np.rot90(flair[:, :, z]), cmap="gray")
    axes[3].imshow(np.ma.masked_where(np.rot90(seg[:, :, z]) == 0,
                                      np.rot90(seg[:, :, z] > 0)),
                   cmap="autumn", alpha=0.5)
    axes[3].set_title("FLAIR + mask WT")
    axes[3].axis("off")
    fig.suptitle(f"{case_id} - lat z={z}")
    savefig(os.path.join(out_dir, "fig_sample_case_full.png"))


# ---------------------------------------------------------------- tong ket
def overview(df):
    L = []
    L.append(f"Tong so ca tim thay: {len(df)}")
    L.append(f"  - co file seg: {int(df['has_seg'].sum())}")
    L.append(f"  - thieu modality: {int(df.filter(like='_missing').any(axis=1).sum())}")
    L.append("Kich thuoc anh (shape) va so ca:")
    for k, v in df["shape"].value_counts().items():
        L.append(f"  {k}: {v}")
    L.append(f"Ca co shape khong dong nhat giua cac modality: "
             f"{int((~df['shape_consistent']).sum())}")
    if "spacing_mm" in df:
        L.append("Voxel spacing (mm):")
        for k, v in df["spacing_mm"].value_counts().items():
            L.append(f"  {k}: {v}")
    if "orientation" in df:
        L.append("Huong truc (orientation):")
        for k, v in df["orientation"].value_counts().items():
            L.append(f"  {k}: {v}")
    L.append(f"So lat z / ca: min={df['n_slices_z'].min()}, max={df['n_slices_z'].max()}")
    L.append(f"So lat co nao / ca: min={df['brain_slices'].min()}, "
             f"mean={df['brain_slices'].mean():.1f}, max={df['brain_slices'].max()}")
    L.append(f"So lat rong (khong co nao) / ca: mean={df['empty_slices'].mean():.1f}")

    d = df[df["has_seg"]]
    if not d.empty:
        L.append("--- Ton thuong (Whole Tumor = seg > 0) ---")
        L.append("Tap nhan xuat hien:")
        for k, v in d["labels"].value_counts().items():
            L.append(f"  {{{k}}}: {v} ca")
        bad = d[d["unexpected_labels"].astype(str) != ""]
        L.append(f"Ca co nhan la (ngoai 0,1,2,4): {len(bad)}")
        L.append(f"So lat co u / ca: min={d['tumor_slices'].min()}, "
                 f"mean={d['tumor_slices'].mean():.1f}, "
                 f"median={d['tumor_slices'].median():.0f}, max={d['tumor_slices'].max()}")
        L.append(f"Tong lat co u: {int(d['tumor_slices'].sum())}  |  "
                 f"tong lat nao khong u: {int(d['healthy_brain_slices'].sum())}")
        tot_t, tot_h = d["tumor_slices"].sum(), d["healthy_brain_slices"].sum()
        L.append(f"Ti le lat u / (u + lanh) toan tap: {tot_t / (tot_t + tot_h):.3f}")
        L.append(f"The tich WT (cm3): min={d['wt_volume_mm3'].min() / 1000:.1f}, "
                 f"median={d['wt_volume_mm3'].median() / 1000:.1f}, "
                 f"max={d['wt_volume_mm3'].max() / 1000:.1f}")
        L.append(f"Dien tich u lon nhat tren 1 lat (px): "
                 f"min={d['max_tumor_area_px'].min()}, max={d['max_tumor_area_px'].max()}")
        L.append(f"Ca co u nam ngoai vung nao: "
                 f"{int((d['tumor_outside_brain_vox'] > 0).sum())}")
        small = d.nsmallest(5, "wt_voxels")[["case_id", "wt_voxels", "tumor_slices"]]
        L.append("5 ca u nho nhat:")
        L.append(small.to_string(index=False))
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="EDA BraTS2020")
    ap.add_argument("--data_dir", required=True, help="Thu muc goc chua du lieu BraTS2020")
    ap.add_argument("--out_dir", default="reports", help="Thu muc luu ket qua")
    ap.add_argument("--limit", type=int, default=None, help="Chi chay N ca dau (de thu)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cases = find_cases(args.data_dir)
    if not cases:
        sys.exit(f"Khong tim thay ca nao (thu muc co file *flair*.nii) trong: {args.data_dir}")
    ids = list(cases)[: args.limit] if args.limit else list(cases)
    print(f"Tim thay {len(cases)} ca, se xu ly {len(ids)} ca.")

    rows, errors = [], []
    for cid in tqdm(ids, desc="EDA"):
        try:
            rows.append(analyze_case(cid, cases[cid]))
        except Exception as e:  # khong de 1 ca loi lam hong ca lo
            errors.append((cid, repr(e)))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "eda_summary.csv")
    df.to_csv(csv_path, index=False)

    text = overview(df)
    if errors:
        text += f"\n--- {len(errors)} ca bi loi khi doc ---\n" + "\n".join(
            f"{c}: {e}" for c, e in errors)
    with open(os.path.join(args.out_dir, "eda_overview.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)

    make_figures(df, args.out_dir)
    try:
        sample_figure({k: cases[k] for k in ids}, df, args.out_dir)
    except Exception as e:
        print(f"[Canh bao] Khong luu duoc anh minh hoa: {e!r}")
    print(f"\nDa luu ket qua vao: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()