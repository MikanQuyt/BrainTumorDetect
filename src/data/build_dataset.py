"""
build_dataset.py - Pipeline tien xu ly BraTS2020 -> lat cat ngang 2D.

Moi ca (volume 3D) di qua cac buoc:
  1. Doc FLAIR, T1ce, T2 va mask seg.
  2. Chuan hoa Z-score tung modality tren cac voxel nao (voxel > 0); nen giu 0.
  3. Cat lat ngang (axial) theo truc z -> anh 2D.
  4. Loc lat rong (dien tich nao < --min_brain_area va khong co u).
  5. Tao mask Whole Tumor (WT) = seg > 0.
  6. Tinh Bounding Box (mot box bao toan bo WT tren moi lat).
  7. Ghep 3 kenh theo thu tu [FLAIR, T1ce, T2].
  8. Chia train/val/test THEO CA (khong theo lat) de tranh ro ri du lieu.
  9. Can bang u / lanh o tap train (mac dinh): giu moi lat u, lay ngau nhien
     lat lanh voi ti le --neg_ratio (mac dinh 1.0) cho tung ca.

Cach chay (thu 5 ca truoc):

    python src/data/build_dataset.py ^
        --data_dir "C:/TTNT/DoAn/datasets/BraTS2020/archive (4)/BraTS2020_TrainingData" ^
        --out_dir  "E:/brats_processed" --limit 5

Dau ra trong --out_dir:
    train|val|test/<case>_z<zzz>.npz   moi file: image (H,W,3) float16, mask (H,W) uint8
    metadata.csv          mot dong / lat DUOC GIU (co bbox xyxy va dang YOLO)
    splits.csv            ca nao thuoc tap nao
    all_slices_areas.csv  dien tich nao/u cua MOI lat + ly do giu/bo (de chon nguong)
    case_summary.csv      mean/std truoc Z-score, so lat giu/bo, huong truc
    config.json           tham so da dung (de tai lap)

Quy uoc huong anh: du lieu BraTS2020 co huong LPS; ta hoan doi 2 truc dau
(swapaxes 0,1) cho moi volume, nen anh luu ra o dang "thang" (truoc o tren,
trai benh nhan o ben phai anh). Anh va mask dung CHUNG phep bien doi nen
bbox luon khop voi mask.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

CHANNELS = ["flair", "t1ce", "t2"]          # thu tu kenh theo de bai
NII_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)


# ---------------------------------------------------------------- tim file
def classify_file(fname):
    if not NII_RE.search(fname):
        return None
    s = NII_RE.sub("", fname).lower()
    if "seg" in s:                       # 'seg' hoac 'Segm' (ca 355 cua Kaggle)
        return "seg"
    for key in ("flair", "t1ce", "t1", "t2"):
        if s.endswith(key):
            return key
    return None


def find_cases(data_dir, need_seg=True):
    cases = {}
    for root, _dirs, files in os.walk(data_dir):
        found = {}
        for f in files:
            kind = classify_file(f)
            if kind and kind not in found:
                found[kind] = os.path.join(root, f)
        if all(k in found for k in CHANNELS) and (not need_seg or "seg" in found):
            cases[os.path.basename(root)] = found
    return dict(sorted(cases.items()))


# ---------------------------------------------------------------- xu ly 1 ca
def load_oriented(path, is_seg=False):
    nii = nib.load(path)
    arr = np.asarray(nii.dataobj)
    arr = np.rint(arr).astype(np.int16) if is_seg else arr.astype(np.float32)
    orient = "".join(nib.aff2axcodes(nii.affine))
    return np.swapaxes(arr, 0, 1), orient          # (y, x, z)


def zscore_brain(vol):
    """Z-score tren voxel nao (vol > 0); nen giu bang 0."""
    m = vol > 0
    out = np.zeros_like(vol)
    if not m.any():
        return out, 0.0, 0.0
    mean = float(vol[m].mean())
    std = float(vol[m].std())
    out[m] = (vol[m] - mean) / (std if std > 1e-8 else 1.0)
    return out, mean, std


def bbox_of(mask2d):
    ys, xs = np.where(mask2d)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def process_case(case_id, paths, split, args, out_root, rng):
    zs, raw_stats, brain = [], {}, None
    orients = set()
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

    img = np.stack(zs, axis=-1)                       # (H, W, Z, 3)
    del zs
    wt = seg > 0                                      # mask Whole Tumor
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

    # can bang u / lanh (chi o cac split trong --balance_splits)
    if split in args.balance_splits:
        tumor_idx = [z for z in range(Z) if kind[z] == "tumor"]
        healthy_idx = [z for z in range(Z) if kind[z] == "healthy"]
        n_keep = int(round(args.neg_ratio * len(tumor_idx)))
        if len(healthy_idx) > n_keep:
            keep = set(rng.choice(healthy_idx, size=n_keep, replace=False).tolist())
            for z in healthy_idx:
                if z not in keep:
                    kind[z] = "drop_balance"

    os.makedirs(os.path.join(out_root, split), exist_ok=True)
    meta, areas, nbytes = [], [], 0
    for z in range(Z):
        kept = kind[z] in ("tumor", "healthy")
        areas.append(dict(case_id=case_id, z=z, split=split,
                          brain_area=int(brain_area[z]), tumor_area=int(tumor_area[z]),
                          kept=kept, reason=kind[z]))
        if not kept:
            continue
        mask2d = wt[:, :, z]
        rel = f"{split}/{case_id}_z{z:03d}.npz"
        path = os.path.join(out_root, rel)
        np.savez_compressed(path,
                            image=img[:, :, z, :].astype(args.dtype),
                            mask=mask2d.astype(np.uint8))
        nbytes += os.path.getsize(path)
        row = dict(split=split, case_id=case_id, z=z, path=rel,
                   has_tumor=int(kind[z] == "tumor"),
                   tumor_area=int(tumor_area[z]), brain_area=int(brain_area[z]),
                   x_min=np.nan, y_min=np.nan, x_max=np.nan, y_max=np.nan,
                   yolo_cx=np.nan, yolo_cy=np.nan, yolo_w=np.nan, yolo_h=np.nan)
        if kind[z] == "tumor":
            x0, y0, x1, y1 = bbox_of(mask2d)
            row.update(x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                       yolo_cx=(x0 + x1 + 1) / 2.0 / W, yolo_cy=(y0 + y1 + 1) / 2.0 / H,
                       yolo_w=(x1 - x0 + 1) / W, yolo_h=(y1 - y0 + 1) / H)
        meta.append(row)

    n = lambda k: sum(1 for x in kind if x == k)  # noqa: E731
    summ = dict(case_id=case_id, split=split, orientation="/".join(sorted(orients)),
                n_slices=Z, kept_tumor=n("tumor"), kept_healthy=n("healthy"),
                drop_empty=n("drop_empty"), drop_tiny_tumor=n("drop_tiny_tumor"),
                drop_balance=n("drop_balance"), **raw_stats)
    return meta, areas, summ, nbytes


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
    ap = argparse.ArgumentParser(description="Build 2D slice dataset tu BraTS2020")
    ap.add_argument("--data_dir", required=True, help="Thu muc BraTS2020_TrainingData")
    ap.add_argument("--out_dir", required=True, help="Thu muc luu dataset da xu ly")
    ap.add_argument("--limit", type=int, default=None, help="Chi xu ly N ca dau (de thu)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--min_brain_area", type=int, default=500,
                    help="Lat khong co u va co dien tich nao (px) < nguong nay bi coi la rong")
    ap.add_argument("--min_tumor_area", type=int, default=1,
                    help="Lat co u nho hon nguong nay (nhung > 0) bi bo han")
    ap.add_argument("--neg_ratio", type=float, default=1.0,
                    help="So lat lanh giu lai / so lat u (cho moi ca, o split can bang)")
    ap.add_argument("--balance_splits", nargs="*", default=["train"],
                    help="Cac split se duoc can bang (mac dinh: train)")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    free_gb = shutil.disk_usage(args.out_dir).free / 1e9
    print(f"Dung luong trong tai noi luu: {free_gb:.1f} GB")

    cases = find_cases(args.data_dir)
    if not cases:
        sys.exit(f"Khong tim thay ca nao (du FLAIR, T1ce, T2, seg) trong: {args.data_dir}")
    n_all = len(cases)
    ids = list(cases)[: args.limit] if args.limit else list(cases)
    splits = make_splits(ids, args.val_ratio, args.test_ratio, args.seed)
    print(f"Tim thay {n_all} ca, xu ly {len(ids)} ca "
          f"(train {sum(v == 'train' for v in splits.values())}, "
          f"val {sum(v == 'val' for v in splits.values())}, "
          f"test {sum(v == 'test' for v in splits.values())}).")

    all_meta, all_areas, all_summ, errors, total_bytes = [], [], [], [], 0
    t0 = time.time()
    for idx, cid in enumerate(tqdm(ids, desc="Build")):
        rng = np.random.RandomState(args.seed * 1000 + idx)
        try:
            meta, areas, summ, nb = process_case(cid, cases[cid], splits[cid],
                                                 args, args.out_dir, rng)
            all_meta += meta
            all_areas += areas
            all_summ.append(summ)
            total_bytes += nb
        except Exception as e:                     # 1 ca loi khong lam hong ca lo
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

    # ---- tong ket
    print(f"\nHoan tat sau {time.time() - t0:.0f}s, {len(all_summ)} ca thanh cong, "
          f"{len(errors)} ca loi.")
    if not meta_df.empty:
        print("\nSo lat DUOC GIU theo tung tap:")
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
