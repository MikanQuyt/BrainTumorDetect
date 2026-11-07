"""
build_dataset.py - Pipeline tien xu ly BraTS2020 -> lat cat ngang 2D (ho tro .nii/.nii.gz va .h5).

Moi ca (volume 3D) di qua cac buoc:
  1. Doc FLAIR, T1ce, T2 va mask seg (tu file 3D .nii/.nii.gz hoac ghep tu cac lat .h5).
  2. Chuan hoa Z-score tung modality tren cac voxel nao (voxel > 0); nen giu 0.
  3. Cat lat ngang (axial) theo truc z -> anh 2D.
  4. Loc lat rong o dinh so va nen so (dien tich nao < --min_brain_area va khong co u).
  5. Chi giu lai cac lat cat co dien tich u >= 50 pixels (--min_tumor_area 50) va lay mau
     mot luong can bang lat cat mo nao lanh (khong u) theo --neg_ratio.
  6. Tao mask Whole Tumor (WT) = (seg > 0).astype(uint8) (gop NCR/NET=1, ED=2, ET=4).
  7. Tinh Bounding Box chuan bao toan bo WT tren moi lat.
  8. Ghep 3 kenh [FLAIR, T1ce, T2] da chuan hoa Z-score thanh anh 3 kenh .png (va .npz).
  9. Chia train/val/test THEO CA (khong theo lat) de tranh ro ri du lieu va xuat kem
     dataset/brats2020_annotations.json tuong thich 100% voi UnifiedBrainMRIDataset.

Cach chay:
    python src/data/build_dataset.py ^
        --data_dir "dataset/brats2020" ^
        --out_dir  "dataset/brats2020_png" --limit 5

Dau ra trong --out_dir:
    images/<case>_z<zzz>.png           anh 3 kenh [FLAIR, T1ce, T2] da Z-score (uint8 PNG)
    masks/<case>_z<zzz>.png            mat na Whole Tumor (WT) nhi phan (0 / 255 PNG)
    train|val|test/<case>_z<zzz>.npz   moi file: image (H,W,3) float16 Z-score, mask (H,W) uint8
    metadata.csv                       mot dong / lat DUOC GIU (co bbox xyxy, YOLO, png_path, mask_path)
    splits.csv                         ca nao thuoc tap nao
    all_slices_areas.csv               dien tich nao/u cua MOI lat + ly do giu/bo
    case_summary.csv                   mean/std truoc Z-score, so lat giu/bo, huong truc
    config.json                        tham so da dung
    dataset/brats2020_annotations.json metadata chuan cho UnifiedBrainMRIDataset
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import sys
import time

import cv2
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

CHANNELS = ["flair", "t1ce", "t2"]          # thu tu 3 kenh theo de bai: [FLAIR, T1ce, T2]
H5_MOD_ORDER = ["flair", "t1", "t1ce", "t2"]
NII_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
H5_RE = re.compile(r"^volume_(\d+)_slice_(\d+)\.h5$", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- tim file (.nii & .h5)
def classify_file(fname):
    if not NII_RE.search(fname):
        return None
    s = NII_RE.sub("", fname).lower()
    if "seg" in s:
        return "seg"
    for key in ("flair", "t1ce", "t1", "t2"):
        if s.endswith(key):
            return key
    return None


def find_cases(data_dir, need_seg=True):
    nii_cases = {}
    h5_groups = defaultdict(list)

    for root, _dirs, files in os.walk(data_dir):
        found = {}
        for f in files:
            m_h5 = H5_RE.match(f)
            if m_h5:
                vol_id, sl_id = int(m_h5.group(1)), int(m_h5.group(2))
                h5_groups[vol_id].append((sl_id, os.path.join(root, f)))
                continue
            kind = classify_file(f)
            if kind and kind not in found:
                found[kind] = os.path.join(root, f)
        if all(k in found for k in CHANNELS) and (not need_seg or "seg" in found):
            nii_cases[os.path.basename(root)] = {"format": "nii", "paths": found}

    if nii_cases:
        return dict(sorted(nii_cases.items()))

    h5_cases = {}
    for vol_id in sorted(h5_groups):
        slices = [p for _, p in sorted(h5_groups[vol_id], key=lambda x: x[0])]
        cid = f"BraTS20_Volume_{vol_id:03d}"
        h5_cases[cid] = {"format": "h5", "slices": slices, "volume_id": vol_id}
    return h5_cases


# ---------------------------------------------------------------- xu ly 1 ca
def load_oriented(path, is_seg=False):
    nii = nib.load(path)
    arr = np.asarray(nii.dataobj)
    arr = np.rint(arr).astype(np.int16) if is_seg else arr.astype(np.float32)
    orient = "".join(nib.aff2axcodes(nii.affine))
    return np.swapaxes(arr, 0, 1), orient          # (y, x, z)


def load_h5_case(slice_paths):
    """Doc danh sach lat .h5 cua 1 volume thanh 3 khoi 3D [FLAIR, T1ce, T2] va seg 3D."""
    ch_idx = [H5_MOD_ORDER.index(c) for c in CHANNELS]
    mod_vols = {c: [] for c in CHANNELS}
    seg_list = []

    for sp in slice_paths:
        with h5py.File(sp, "r") as f:
            img = np.asarray(f["image"][()], dtype=np.float32)
            msk = np.asarray(f["mask"][()])
        if img.ndim == 3 and img.shape[0] <= 4 < img.shape[-1]:
            img = np.moveaxis(img, 0, -1)
        if msk.ndim == 3 and msk.shape[0] <= 4 < msk.shape[-1]:
            msk = np.moveaxis(msk, 0, -1)

        for c, idx in zip(CHANNELS, ch_idx):
            mod_vols[c].append(img[:, :, idx])

        if msk.ndim == 3:
            wt2d = (msk > 0).any(axis=-1).astype(np.int16)
        else:
            wt2d = (msk > 0).astype(np.int16)
        seg_list.append(wt2d)

    vols = {c: np.stack(mod_vols[c], axis=-1) for c in CHANNELS}
    seg = np.stack(seg_list, axis=-1)
    return vols, seg, "LPS"


def zscore_brain(vol):
    """Z-score tren voxel nao (vol > 0); nen giu bang 0."""
    m = vol > 0
    out = np.zeros_like(vol, dtype=np.float32)
    if not m.any():
        return out, 0.0, 0.0
    mean = float(vol[m].mean())
    std = float(vol[m].std())
    out[m] = (vol[m] - mean) / (std if std > 1e-8 else 1.0)
    return out, mean, std


def zscore_slice_to_uint8_png(slice_zscore, brain_mask_2d, clip_sigma=3.0):
    """
    Anh xa lat cat 3 kenh [FLAIR, T1ce, T2] da chuan hoa Z-score tu [-clip_sigma, +clip_sigma]
    ve [0, 255] uint8 de luu thanh file .png 3 kenh, giu nen ngoai hop so bang 0.
    """
    scaled = (np.clip(slice_zscore, -clip_sigma, clip_sigma) + clip_sigma) / (2.0 * clip_sigma)
    u8 = np.rint(scaled * 255.0).astype(np.uint8)
    u8[~brain_mask_2d] = 0
    return u8


def bbox_of(mask2d):
    ys, xs = np.where(mask2d > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def process_case(case_id, case_entry, split, args, out_root, rng):
    zs, raw_stats, brain = [], {}, None
    orients = set()

    if case_entry["format"] == "h5":
        vols_dict, seg, orient = load_h5_case(case_entry["slices"])
        orients.add(orient)
        for ch in CHANNELS:
            vol = vols_dict[ch]
            brain = (vol != 0) if brain is None else (brain | (vol != 0))
            z, mean, std = zscore_brain(np.abs(vol) if (vol < 0).any() else vol)
            zs.append(z)
            raw_stats[f"{ch}_mean_raw"] = mean
            raw_stats[f"{ch}_std_raw"] = std
    else:
        paths = case_entry["paths"]
        for ch in CHANNELS:
            vol, orient = load_oriented(paths[ch])
            orients.add(orient)
            brain = (vol > 0) if brain is None else (brain | (vol > 0))
            z, mean, std = zscore_brain(vol)
            zs.append(z)
            raw_stats[f"{ch}_mean_raw"] = mean
            raw_stats[f"{ch}_std_raw"] = std
        seg, _ = load_oriented(paths["seg"], is_seg=True)

    if not all(z.shape == seg.shape for z in zs):
        raise ValueError("shape anh va seg khong khop")

    img = np.stack(zs, axis=-1)                       # (H, W, Z, 3) theo thu tu [FLAIR, T1ce, T2]
    del zs
    wt = (seg > 0).astype(np.uint8)                   # mask Whole Tumor (WT) = (seg > 0).astype(uint8)
    H, W, Z, _ = img.shape
    brain_area = brain.sum(axis=(0, 1))
    tumor_area = wt.sum(axis=(0, 1))

    kind = []
    for z in range(Z):
        ta, ba = int(tumor_area[z]), int(brain_area[z])
        if ta >= args.min_tumor_area:
            kind.append("tumor")
        elif ta > 0:
            kind.append("drop_tiny_tumor")
        elif ba >= args.min_brain_area:
            kind.append("healthy")
        else:
            kind.append("drop_empty")

    # Can bang u / lanh o cac split trong --balance_splits
    if split in args.balance_splits:
        tumor_idx = [z for z in range(Z) if kind[z] == "tumor"]
        healthy_idx = [z for z in range(Z) if kind[z] == "healthy"]
        n_keep = int(round(args.neg_ratio * len(tumor_idx)))
        if len(healthy_idx) > n_keep:
            keep = set(rng.choice(healthy_idx, size=n_keep, replace=False).tolist())
            for z in healthy_idx:
                if z not in keep:
                    kind[z] = "drop_balance"

    img_dir = os.path.join(out_root, "images")
    msk_dir = os.path.join(out_root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    if args.save_npz:
        os.makedirs(os.path.join(out_root, split), exist_ok=True)

    meta, areas, json_records, nbytes = [], [], [], 0
    for z in range(Z):
        kept = kind[z] in ("tumor", "healthy")
        areas.append(dict(case_id=case_id, z=z, split=split,
                          brain_area=int(brain_area[z]), tumor_area=int(tumor_area[z]),
                          kept=kept, reason=kind[z]))
        if not kept:
            continue

        slice_z = img[:, :, z, :]
        mask2d = wt[:, :, z]
        brain2d = brain[:, :, z]
        stem = f"{case_id}_z{z:03d}"

        # 1. Luu anh 3 kenh [FLAIR, T1ce, T2] da chuan hoa Z-score ra .png va mask WT ra .png
        rgb_png = zscore_slice_to_uint8_png(slice_z, brain2d)
        rel_png = f"images/{stem}.png"
        rel_msk = f"masks/{stem}.png"
        abs_png = os.path.join(out_root, rel_png)
        abs_msk = os.path.join(out_root, rel_msk)
        cv2.imwrite(abs_png, cv2.cvtColor(rgb_png, cv2.COLOR_RGB2BGR))
        cv2.imwrite(abs_msk, mask2d * 255)
        nbytes += os.path.getsize(abs_png) + os.path.getsize(abs_msk)

        # 2. Luu them file .npz chua Z-score float16 (mac dinh bat de check_dataset.py kiem tra Z-score)
        rel_npz = f"{split}/{stem}.npz"
        if args.save_npz:
            abs_npz = os.path.join(out_root, rel_npz)
            np.savez_compressed(abs_npz,
                                image=slice_z.astype(args.dtype),
                                mask=mask2d.astype(np.uint8))
            nbytes += os.path.getsize(abs_npz)

        has_tumor = int(kind[z] == "tumor")
        row = dict(split=split, case_id=case_id, z=z,
                   path=rel_npz if args.save_npz else rel_png,
                   png_path=rel_png, mask_path=rel_msk,
                   has_tumor=has_tumor,
                   tumor_area=int(tumor_area[z]), brain_area=int(brain_area[z]),
                   x_min=np.nan, y_min=np.nan, x_max=np.nan, y_max=np.nan,
                   yolo_cx=np.nan, yolo_cy=np.nan, yolo_w=np.nan, yolo_h=np.nan)
        gt_box = None
        if has_tumor:
            x0, y0, x1, y1 = bbox_of(mask2d)
            row.update(x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                       yolo_cx=(x0 + x1 + 1) / 2.0 / W, yolo_cy=(y0 + y1 + 1) / 2.0 / H,
                       yolo_w=(x1 - x0 + 1) / W, yolo_h=(y1 - y0 + 1) / H)
            gt_box = [int(x0), int(y0), int(x1) + 1, int(y1) + 1]
        meta.append(row)

        try:
            img_rel_proj = Path(abs_png).resolve().relative_to(PROJECT_ROOT).as_posix()
            msk_rel_proj = Path(abs_msk).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            img_rel_proj = Path(abs_png).resolve().as_posix()
            msk_rel_proj = Path(abs_msk).resolve().as_posix()

        json_records.append({
            "dataset": "brats2020",
            "case_id": case_id,
            "slice": int(z),
            "img_path": img_rel_proj,
            "mask_path": msk_rel_proj,
            "has_mask": True,
            "has_tumor": has_tumor,
            "sub_class_label": 1 if has_tumor else 0,
            "gt_box": gt_box,
            "split": split,
            "fold": None,
        })

    n = lambda k: sum(1 for x in kind if x == k)  # noqa: E731
    summ = dict(case_id=case_id, split=split, orientation="/".join(sorted(orients)),
                n_slices=Z, kept_tumor=n("tumor"), kept_healthy=n("healthy"),
                drop_empty=n("drop_empty"), drop_tiny_tumor=n("drop_tiny_tumor"),
                drop_balance=n("drop_balance"), **raw_stats)
    return meta, areas, summ, json_records, nbytes


# ---------------------------------------------------------------- chia split
def make_splits(ids, val_ratio, test_ratio, seed):
    N = len(ids)
    perm = np.random.RandomState(seed).permutation(N)
    n_val = max(1, int(round(val_ratio * N))) if N >= 3 else 0
    n_test = max(1, int(round(test_ratio * N))) if N >= 3 else 0
    n_train = N - n_val - n_test
    out = {}
    for rank, i in enumerate(perm):
        out[ids[i]] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Build 2D slice dataset tu BraTS2020 (.nii hoac .h5)")
    ap.add_argument("--data_dir", default="dataset/brats2020", help="Thu muc BraTS2020 (.nii hoac .h5)")
    ap.add_argument("--out_dir", default="dataset/brats2020_png", help="Thu muc luu dataset da xu ly")
    ap.add_argument("--out_json", default="dataset/brats2020_annotations.json",
                    help="File JSON chu thich xuat ra cho UnifiedBrainMRIDataset")
    ap.add_argument("--limit", type=int, default=None, help="Chi xu ly N ca dau (de thu)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--min_brain_area", type=int, default=500,
                    help="Lat khong co u va co dien tich nao (px) < nguong nay bi coi la rong")
    ap.add_argument("--min_tumor_area", type=int, default=50,
                    help="Chi giu lat co dien tich u >= 50 pixels theo dung yeu cau de bai")
    ap.add_argument("--neg_ratio", type=float, default=1.0,
                    help="So lat lanh giu lai / so lat u (cho moi ca, o split can bang)")
    ap.add_argument("--balance_splits", nargs="*", default=["train", "val", "test"],
                    help="Cac split se duoc can bang lat lanh (mac dinh: train val test)")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--save_npz", action="store_true", default=True,
                    help="Luu kem file .npz chua Z-score float16 ben canh file .png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    free_gb = shutil.disk_usage(args.out_dir).free / 1e9
    print(f"Dung luong trong tai noi luu: {free_gb:.1f} GB")

    cases = find_cases(args.data_dir)
    if not cases:
        sys.exit(f"Khong tim thay ca nao (.nii hoac .h5) trong: {args.data_dir}")
    n_all = len(cases)
    ids = list(cases)[: args.limit] if args.limit else list(cases)
    splits = make_splits(ids, args.val_ratio, args.test_ratio, args.seed)
    print(f"Tim thay {n_all} ca, xu ly {len(ids)} ca "
          f"(train {sum(v == 'train' for v in splits.values())}, "
          f"val {sum(v == 'val' for v in splits.values())}, "
          f"test {sum(v == 'test' for v in splits.values())}).")

    all_meta, all_areas, all_summ, all_json, errors, total_bytes = [], [], [], [], [], 0
    t0 = time.time()
    for idx, cid in enumerate(tqdm(ids, desc="Build")):
        rng = np.random.RandomState(args.seed * 1000 + idx)
        try:
            meta, areas, summ, jrecs, nb = process_case(
                cid, cases[cid], splits[cid], args, args.out_dir, rng
            )
            all_meta += meta
            all_areas += areas
            all_summ.append(summ)
            all_json += jrecs
            total_bytes += nb
        except Exception as e:
            errors.append((cid, repr(e)))

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    pd.DataFrame(sorted(splits.items()), columns=["case_id", "split"]).to_csv(
        os.path.join(args.out_dir, "splits.csv"), index=False)
    pd.DataFrame(all_areas).to_csv(os.path.join(args.out_dir, "all_slices_areas.csv"), index=False)
    summ_df = pd.DataFrame(all_summ)
    summ_df.to_csv(os.path.join(args.out_dir, "case_summary.csv"), index=False)
    cfg = {k: v for k, v in vars(args).items()}
    cfg["n_cases_processed"] = len(all_summ)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    if args.out_json:
        json_out_path = PROJECT_ROOT / args.out_json
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2, ensure_ascii=False)
        print(f"Da xuat {len(all_json)} record JSON cho UnifiedBrainMRIDataset -> {json_out_path}")

    # ---- tong ket
    print(f"\nHoan tat sau {time.time() - t0:.0f}s, {len(all_summ)} ca thanh cong, "
          f"{len(errors)} ca loi.")
    if not meta_df.empty:
        print("\nSo lat DUOC GIU theo tung tap (min_tumor_area >= "
              f"{args.min_tumor_area} px):")
        tab = meta_df.groupby("split")["has_tumor"].agg(
            slices="size", tumor="sum")
        tab["healthy"] = tab["slices"] - tab["tumor"]
        tab["tumor_ratio"] = (tab["tumor"] / tab["slices"]).round(3)
        tab["cases"] = meta_df.groupby("split")["case_id"].nunique()
        print(tab[["cases", "slices", "tumor", "healthy", "tumor_ratio"]].to_string())
    if not summ_df.empty:
        print("\nSo lat BI BO (cong don tren tat ca ca):")
        for k in ("drop_empty", "drop_tiny_tumor", "drop_balance"):
            print(f"  {k}: {int(summ_df[k].sum())}")
        non_lps = summ_df[summ_df["orientation"] != "LPS"]
        if len(non_lps):
            print(f"\n[Canh bao] {len(non_lps)} ca khong co huong LPS: "
                  f"{non_lps['case_id'].tolist()[:5]} ...")
    gb = total_bytes / 1e9
    est = gb / max(len(all_summ), 1) * n_all
    print(f"\nDung luong da ghi: {gb:.2f} GB; uoc tinh cho {n_all} ca: ~{est:.1f} GB "
          f"(con trong: {free_gb:.1f} GB)")
    if errors:
        print(f"\n--- {len(errors)} ca bi loi ---")
        for c, e in errors:
            print(f"{c}: {e}")
    print(f"\nDa luu vao: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
