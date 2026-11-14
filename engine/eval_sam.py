"""Đánh giá Zero-shot SAM nguyên bản và MedSAM với hộp chuẩn (Oracle GT Box) bằng mô-đun đo lường chung.

    python -m engine.eval_sam --datasets brisc2025 --split val --models sam medsam --out runs/zeroshot_ds3_val
    python -m engine.eval_sam --limit 5 --models sam --out runs/smoke_sam          # chạy thử nhanh

Mọi chỉ số (Dice, IoU, HD95) tính trên mask hệ 1024x1024 do UnifiedBrainMRIDataset trả về, nên SAM và MedSAM
được so sánh cùng một lưới pixel và cùng một hộp (`gt_box` đã đổi sang hệ 1024). Chỉ lấy mẫu có mask và có hộp.
Trọng số tải bằng `python -m utils.checkpoints`.

Đầu ra trong --out: per_image.csv (mỗi dòng một ảnh x mô hình), summary.json (tóm tắt theo mô hình).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.unified_dataset import UnifiedBrainMRIDataset
from models.sam_wrappers import load_segmenter
from utils.metrics import SegmentationMeter


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ann",
        nargs="+",
        default=[
            "dataset/ds1_annotations.json",
            "dataset/ds2_annotations.json",
            "dataset/ds3_annotations.json",
            "dataset/brats2020_annotations.json",
        ],
    )
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="+", default=["sam", "medsam"], choices=["sam", "medsam"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", "--out-dir", dest="out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = UnifiedBrainMRIDataset(args.ann, train=False, datasets=args.datasets, split=args.split)
    ds.records = [r for r in ds.records if r.get("has_mask") and r.get("mask_path") and int(r["has_tumor"]) == 1]
    if args.limit and len(ds.records) > args.limit:
        rng = np.random.RandomState(args.seed)
        keep = sorted(rng.choice(len(ds.records), args.limit, replace=False))
        ds.records = [ds.records[i] for i in keep]
    if not ds.records:
        raise SystemExit("Không có mẫu có u + mask cho cấu hình đã chọn.")
    print(f"Đánh giá {len(ds)} ảnh | split={args.split} | models={args.models} | device={args.device}")

    segmenters = {m: load_segmenter(m, args.ckpt_dir, args.device) for m in args.models}
    meters = {m: SegmentationMeter() for m in args.models}
    rows = []
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        if not bool(item["has_box"]) or not bool(item["has_mask"]):
            continue
        image = (item["image_1024"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        gt = item["gt_mask"].numpy().astype(bool)
        box = item["gt_box"].numpy()
        for name, seg in segmenters.items():
            pred = seg.segment(image, box)
            r = meters[name].update(pred, gt, index=i)
            rows.append({"index": i, "model": name, "dice": r["dice"], "iou": r["iou"], "hd95": r["hd95"],
                         "pred_empty": r["pred_empty"]})
        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            print(f"  {i + 1}/{len(ds)}  ({time.time() - t0:.0f}s)", flush=True)

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "model", "dice", "iou", "hd95", "pred_empty"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"config": vars(args), "n": len(ds), "models": {m: meters[m].summary() for m in args.models}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nTóm tắt (mean ± sd):")
    for m in args.models:
        s = meters[m].summary()
        print(f"  {m:7s} Dice={s['dice']['mean']:.4f}±{s['dice']['sd']:.4f}  IoU={s['iou']['mean']:.4f}  "
              f"HD95={s['hd95']['mean']:.2f}px (n_undefined={s['hd95_n_undefined']})")
    print(f"Đã lưu vào {out.resolve()}")


if __name__ == "__main__":
    main()
