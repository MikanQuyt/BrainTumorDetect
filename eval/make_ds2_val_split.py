"""
make_ds2_val_split.py - Dinh nghia tap Validation cho DS2 (indk214 Segmentation) va loc lai ket qua zero-shot.

Cot `split` cua dataset/ds2_annotations.json dang rong (null), nen thi nghiem Tuan 2 ban dau chay tren toan bo 2.192 cap.
Script nay chia Stratified 70/15/15 theo lop u (glioma / meningioma / pituitary), seed co dinh, roi loc
oracle_results.csv (KHONG chay lai mo hinh) de lay phan Validation.

    python eval/make_ds2_val_split.py
    python eval/analyze_week2.py --csv ds4=reports/week2/ds4/oracle_results.csv \
        ds2=reports/week2/ds2_val/oracle_results.csv --out_dir reports/week2/val_protocol

Dau ra:
    reports/week2/ds2/ds2_split.csv            group, class, split (train/val/test)
    reports/week2/ds2_val/oracle_results.csv   chi cac dong thuoc tap val
    reports/week2/ds2_val/run_info.json        run_info goc + ghi chu giao thuc loc
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("glioma", "meningioma", "pituitary")


def class_of(group):
    g = str(group).lower()
    for c in CLASSES:
        if g.startswith(c):
            return c
    return "unknown"


def stratified_split(df, val_ratio, test_ratio, seed):
    rng = np.random.RandomState(seed)
    split = pd.Series("train", index=df.index)
    for c, sub in df.groupby("cls"):
        idx = rng.permutation(sub.index.to_numpy())
        n_val = int(round(len(idx) * val_ratio))
        n_test = int(round(len(idx) * test_ratio))
        split.loc[idx[:n_val]] = "val"
        split.loc[idx[n_val:n_val + n_test]] = "test"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="reports/week2/ds2/oracle_results.csv")
    ap.add_argument("--out_split", default="reports/week2/ds2/ds2_split.csv")
    ap.add_argument("--out_dir", default="reports/week2/ds2_val")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    assert df["group"].is_unique, "group phai la duy nhat (moi anh = 1 nhom)"
    df["cls"] = df["group"].map(class_of)
    assert (df["cls"] != "unknown").all(), "co anh khong xac dinh duoc lop u"
    df["split"] = stratified_split(df, args.val_ratio, args.test_ratio, args.seed)

    Path(args.out_split).parent.mkdir(parents=True, exist_ok=True)
    df[["group", "cls", "split"]].rename(columns={"cls": "class"}).to_csv(args.out_split, index=False)

    print("Phan bo split x lop:")
    print(pd.crosstab(df["cls"], df["split"], margins=True))

    val = df[df["split"] == "val"].drop(columns=["cls"]).copy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val.to_csv(out / "oracle_results.csv", index=False)

    info_path = Path(args.src).parent / "run_info.json"
    info = json.load(open(info_path, encoding="utf-8")) if info_path.exists() else {}
    info["split_filter"] = {
        "note": "Loc tu ket qua zero-shot tren toan bo 2.192 anh; khong chay lai mo hinh.",
        "split_file": str(args.out_split).replace("\\", "/"),
        "strategy": "stratified theo lop u, seed=%d, ratio train/val/test = %.2f/%.2f/%.2f" % (
            args.seed, 1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio),
        "n_val": int(len(val)),
    }
    info["split"] = "val"
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nVal: {len(val)} anh -> {out / 'oracle_results.csv'}")


if __name__ == "__main__":
    main()
