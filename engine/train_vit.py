"""Huấn luyện ViT-Base/16 baseline (Linear Probing hoặc Full Fine-tuning) trên UnifiedBrainMRIDataset.

    # Linear Probing: chỉ huấn luyện đầu phân loại, backbone đóng băng
    python -m engine.train_vit --mode linear_probe --datasets brisc2025 --epochs 10 --out runs/vit_lp_ds3

    # Full Fine-tuning: huấn luyện toàn bộ tham số
    python -m engine.train_vit --mode full_finetune --datasets brisc2025 --epochs 10 --out runs/vit_ft_ds3

    # Chạy thử nhanh (vài chục ảnh) để kiểm tra pipeline
    python -m engine.train_vit --mode linear_probe --limit 64 --epochs 1 --out runs/smoke

Dùng chia tập sẵn có trong tệp JSON (cột `split`): huấn luyện trên `train`, chọn epoch tốt nhất theo F1 trên `val`,
báo cáo cuối cùng trên `test` (chạy một lần với trọng số tốt nhất). Mặc định dùng DS3 (BRISC 2025) vì đã có chia tập
chống rò rỉ bệnh nhân; truyền `--datasets` khác chỉ khi bộ đó đã có cột `split`.

Đầu ra trong --out: config.json, history.csv, val_best.json, test_metrics.json, test_predictions.npz, best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.unified_dataset import UnifiedBrainMRIDataset
from models.vit_baseline import MODES, ViTBaseline
from utils.metrics import ClassificationMeter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {4: ["no_tumor", "glioma", "meningioma", "pituitary"], 2: ["no_tumor", "tumor"]}
DEFAULT_ANN = ["dataset/ds3_annotations.json"]
DEFAULT_LR = {"linear_probe": (1e-3, 1e-3), "full_finetune": (2e-5, 1e-3)}   # (backbone, head)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(args, split: str, train: bool) -> UnifiedBrainMRIDataset:
    ds = UnifiedBrainMRIDataset(args.ann, train=train, datasets=args.datasets, split=split)
    key = "sub_class_label" if args.task == "multiclass" else "has_tumor"
    valid = {0, 1, 2, 3} if args.task == "multiclass" else {0, 1}
    ds.records = [r for r in ds.records if int(r[key]) in valid]
    if not ds.records:
        raise SystemExit(f"Không có mẫu nào cho split='{split}', datasets={args.datasets}. "
                         f"Bộ dữ liệu cần có cột `split` hợp lệ và nhãn phù hợp tác vụ '{args.task}'.")
    if args.limit and len(ds.records) > args.limit:        # lấy mẫu cố định, giữ ngẫu nhiên có seed
        rng = np.random.RandomState(args.seed)
        keep = sorted(rng.choice(len(ds.records), args.limit, replace=False))
        ds.records = [ds.records[i] for i in keep]
    return ds


def make_loader(ds, args, shuffle: bool):
    g = torch.Generator().manual_seed(args.seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      generator=g, pin_memory=args.device.startswith("cuda"), drop_last=False)


def label_key(task: str) -> str:
    return "sub_class_label" if task == "multiclass" else "binary_label"


def run_epoch(model, loader, device, task, optimizer=None, scaler=None, amp=False, desc=""):
    training = optimizer is not None
    model.train(training)
    meter = ClassificationMeter(CLASS_NAMES[4 if task == "multiclass" else 2])
    loss_sum, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, 1):
            x = batch["image_224"].to(device, non_blocking=True)
            y = batch[label_key(task)].to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.float(), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            n += len(y)
            meter.update(y.cpu().numpy(), logits.detach().float().softmax(1).cpu().numpy())
            if step % 20 == 0:
                print(f"  {desc} step {step}/{len(loader)} loss={loss_sum / n:.4f}", flush=True)
    metrics = meter.compute(with_detection=(task == "multiclass"))
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics, meter


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--ann", nargs="+", default=DEFAULT_ANN)
    ap.add_argument("--datasets", nargs="+", default=["brisc2025"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=None)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5, help="dừng sớm nếu F1 val không tăng sau N epoch (0 = tắt)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="FP16 mixed precision (chỉ có tác dụng trên CUDA)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số mẫu mỗi split (chạy thử)")
    ap.add_argument("--model-name", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--offline", action="store_true", help="chỉ dùng checkpoint Hugging Face đã có trong cache")
    ap.add_argument("--out", "--out-dir", dest="out", required=True)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_classes = 4 if args.task == "multiclass" else 2
    amp = args.amp and args.device.startswith("cuda")

    train_ds, val_ds, test_ds = (build_dataset(args, s, train=(s == "train")) for s in ("train", "val", "test"))
    train_loader = make_loader(train_ds, args, True)
    val_loader, test_loader = make_loader(val_ds, args, False), make_loader(test_ds, args, False)
    print(f"Mẫu: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | mode={args.mode} task={args.task}")

    model = ViTBaseline(n_classes, args.mode, model_name=args.model_name, local_files_only=args.offline).to(args.device)
    counts = model.param_counts()
    print(f"Tham số: tổng={counts['total']:,} huấn luyện={counts['trainable']:,} ({counts['trainable_pct']:.2f}%)")

    lr_b, lr_h = DEFAULT_LR[args.mode]
    lr_b, lr_h = args.lr_backbone or lr_b, args.lr_head or lr_h
    optimizer = torch.optim.AdamW(model.param_groups(lr_b, lr_h, args.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    save_json(out / "config.json", {**vars(args), "n_classes": n_classes, "lr_backbone": lr_b, "lr_head": lr_h,
                                    "params": counts, "n_train": len(train_ds), "n_val": len(val_ds),
                                    "n_test": len(test_ds)})

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr, _ = run_epoch(model, train_loader, args.device, args.task, optimizer, scaler, amp, f"[E{epoch} train]")
        va, _ = run_epoch(model, val_loader, args.device, args.task, amp=amp, desc=f"[E{epoch} val]")
        sched.step()
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"], "val_loss": va["loss"],
               "val_acc": va["accuracy"], "val_f1": va["f1"], "val_recall": va["recall"], "val_auc": va["auc"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"val_acc={va['accuracy']:.4f} val_f1={va['f1']:.4f} val_auc={va['auc']:.4f}", flush=True)
        if va["f1"] > best_f1:
            best_f1, best_epoch, bad = va["f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.trainable_state_dict().items()}
            save_json(out / "val_best.json", {"epoch": epoch, **va})
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"Dừng sớm tại epoch {epoch} (best epoch {best_epoch}).")
                break

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(best_state, strict=False)           # linear_probe: chỉ nạp đầu phân loại
    test, meter = run_epoch(model, test_loader, args.device, args.task, amp=amp, desc="[test]")
    test["best_epoch"] = best_epoch
    save_json(out / "test_metrics.json", test)
    y, p = meter.arrays()
    np.savez_compressed(out / "test_predictions.npz", y_true=y, y_prob=p)
    torch.save({"state_dict": best_state, "mode": args.mode, "n_classes": n_classes, "epoch": best_epoch},
               out / "best.pt")
    print(f"\nTEST (epoch {best_epoch}): acc={test['accuracy']:.4f} recall={test['recall']:.4f} "
          f"precision={test['precision']:.4f} f1={test['f1']:.4f} auc={test['auc']:.4f}")
    if "detection" in test:
        d = test["detection"]
        print(f"Sàng lọc có/không u: recall={d['recall']:.4f} specificity={d['specificity']:.4f} auc={d['auc']:.4f}")
    print(f"Đã lưu vào {out.resolve()}")


if __name__ == "__main__":
    main()
