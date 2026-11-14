"""Huấn luyện và đánh giá nhánh 1: ViT-LoRA + ViTDetBoxHead + AutoPromptGenerator (Tuần 3).

Ví dụ sử dụng:
    # 1. Huấn luyện ViT-LoRA (r=8, alpha=16) + Box Regression Head (GIoU + L1) trên Dataset 3 (BRISC 2025)
    python -m engine.train_vit_lora --datasets brisc2025 --ann dataset/ds3_annotations.json --epochs 10 --out runs/vit_lora_ds3

    # 2. Huấn luyện 5-Fold CV trên Dataset 1 (Navoneel - không có nhãn Box, dùng Weakly-Supervised Heatmap)
    python -m engine.train_vit_lora --datasets navoneel --ann dataset/ds1_annotations.json --task binary --no-box-head --cv-5fold --epochs 15 --out runs/vit_lora_ds1

    # 3. Khảo sát hạng LoRA r ∈ {4, 8, 16, 32}
    python -m engine.train_vit_lora --lora-r 16 --lora-alpha 32 --datasets brisc2025 --epochs 10 --out runs/vit_lora_r16
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
from torch.utils.data import DataLoader

from data.unified_dataset import UnifiedBrainMRIDataset
from models.auto_prompt import AutoPromptGenerator
from models.vit_lora import ViTLoRA
from utils.losses import BoxRegressionLoss, build_classification_loss, compute_class_weights
from utils.metrics import (
    BoxLocalizationMeter,
    ClassificationMeter,
    detection_metrics,
    optimize_detection_threshold,
)

CLASS_NAMES = {4: ["no_tumor", "glioma", "meningioma", "pituitary"], 2: ["no_tumor", "tumor"]}
DEFAULT_ANN = ["dataset/ds3_annotations.json"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_key(task: str) -> str:
    return "sub_class_label" if task == "multiclass" else "binary_label"


def record_label_key(task: str) -> str:
    return "sub_class_label" if task == "multiclass" else "has_tumor"


def build_split_dataset(
    args,
    split: str | None = None,
    fold: int | None = None,
    exclude_fold: int | None = None,
    train: bool = True,
) -> UnifiedBrainMRIDataset:
    ds = UnifiedBrainMRIDataset(
        args.ann,
        train=train,
        datasets=args.datasets,
        split=split,
        fold=fold,
        exclude_fold=exclude_fold,
    )
    rkey = record_label_key(args.task)
    valid = {0, 1, 2, 3} if args.task == "multiclass" else {0, 1}
    ds.records = [r for r in ds.records if int(r[rkey]) in valid]
    if not ds.records:
        raise SystemExit(
            f"Không tìm thấy mẫu nào cho datasets={args.datasets}, split={split}, fold={fold}, exclude_fold={exclude_fold}."
        )
    if args.limit and len(ds.records) > args.limit:
        rng = np.random.RandomState(args.seed)
        keep = sorted(rng.choice(len(ds.records), args.limit, replace=False))
        ds.records = [ds.records[i] for i in keep]
    return ds


def make_loader(ds: UnifiedBrainMRIDataset, args, shuffle: bool) -> DataLoader:
    g = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        generator=g,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )


def run_epoch(
    model: ViTLoRA,
    loader: DataLoader,
    cls_criterion,
    box_criterion: BoxRegressionLoss | None,
    device: str,
    task: str,
    box_loss_weight: float = 1.0,
    threshold: float = 0.35,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    desc: str = "",
):
    training = optimizer is not None
    model.train(training)
    n_classes = 4 if task == "multiclass" else 2
    cls_meter = ClassificationMeter(CLASS_NAMES[n_classes])
    loc_meter = BoxLocalizationMeter()

    total_loss_sum, cls_loss_sum, box_loss_sum, n = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    use_box = (box_criterion is not None) and (model.box_head is not None)

    with ctx:
        for step, batch in enumerate(loader, 1):
            x = batch["image_224"].to(device, non_blocking=True)
            y = batch[label_key(task)].to(device, non_blocking=True)
            gt_box_norm = batch["gt_box_norm"].to(device, non_blocking=True)
            has_box = batch["has_box"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                out = model(x, return_dict=True)
                logits = out["logits"]
                cls_loss = cls_criterion(logits.float(), y)
                if use_box and out["pred_boxes"] is not None:
                    box_res = box_criterion(out["pred_boxes"], gt_box_norm, has_box=has_box)
                    box_loss = box_res["loss"]
                    loss = cls_loss + box_loss_weight * box_loss
                else:
                    box_loss = logits.sum() * 0.0
                    loss = cls_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = len(y)
            total_loss_sum += float(loss.detach()) * bs
            cls_loss_sum += float(cls_loss.detach()) * bs
            box_loss_sum += float(box_loss.detach()) * bs
            n += bs

            probs = logits.detach().float().softmax(dim=1).cpu().numpy()
            cls_meter.update(y.cpu().numpy(), probs)

            if use_box and out["pred_boxes"] is not None:
                pred_b = out["pred_boxes"].detach().cpu().numpy()
                gt_b = gt_box_norm.detach().cpu().numpy()
                hb = has_box.detach().cpu().numpy()
                for i in range(bs):
                    if hb[i]:
                        loc_meter.update(pred_box=pred_b[i], gt_box=gt_b[i])

            if step % 20 == 0:
                print(
                    f"  {desc} step {step}/{len(loader)} loss={total_loss_sum / n:.4f} "
                    f"(cls={cls_loss_sum / n:.4f}, box={box_loss_sum / n:.4f})",
                    flush=True,
                )

    metrics = cls_meter.compute(with_detection=True)
    y_arr, p_arr = cls_meter.arrays()
    metrics["detection_tau_default"] = detection_metrics(y_arr, p_arr, threshold=0.50)
    metrics["detection_tau_tuned"] = detection_metrics(y_arr, p_arr, threshold=threshold)
    metrics["loss"] = total_loss_sum / max(n, 1)
    metrics["cls_loss"] = cls_loss_sum / max(n, 1)
    metrics["box_loss"] = box_loss_sum / max(n, 1)
    if loc_meter.ious:
        metrics["box_head_localization"] = loc_meter.summary()
    return metrics, cls_meter


def evaluate_auto_prompts(
    model: ViTLoRA,
    dataset: UnifiedBrainMRIDataset,
    device: str,
    max_samples: int = 32,
    threshold: float = 0.35,
    box_padding: float = 0.08,
) -> dict:
    """Đánh giá cả 2 cơ chế Auto-Prompt (Weakly-Supervised Heatmap & Box Head) về Mean Box IoU & Pointing Game."""
    prompt_gen = AutoPromptGenerator(
        model=model,
        mode="auto",
        threshold=threshold,
        box_padding=box_padding,
    )
    hm_meter = BoxLocalizationMeter()
    reg_meter = BoxLocalizationMeter()

    tumor_indices = [i for i, r in enumerate(dataset.records) if int(r.get("has_tumor", 0)) == 1]
    if not tumor_indices:
        tumor_indices = list(range(len(dataset)))
    if max_samples > 0 and len(tumor_indices) > max_samples:
        rng = np.random.RandomState(42)
        tumor_indices = sorted(rng.choice(tumor_indices, max_samples, replace=False))

    for idx in tumor_indices:
        sample = dataset[idx]
        x = sample["image_224"].unsqueeze(0).to(device)
        res = prompt_gen.generate(x)

        has_box = bool(sample["has_box"])
        has_mask = bool(sample["has_mask"])
        gt_box_1024 = sample["gt_box"].numpy() if has_box else None
        gt_mask_1024 = sample["gt_mask"].numpy() if has_mask else None

        peak_pt = res["peak_points_1024"][0].detach().cpu().numpy()
        hm_box = res["heatmap_boxes_1024"][0].detach().cpu().numpy()
        hm_meter.update(
            pred_box=hm_box if has_box else None,
            gt_box=gt_box_1024,
            peak_point_xy=peak_pt,
            gt_mask=gt_mask_1024,
        )
        if res["regressed_boxes_1024"] is not None:
            reg_box = res["regressed_boxes_1024"][0].detach().cpu().numpy()
            reg_meter.update(
                pred_box=reg_box if has_box else None,
                gt_box=gt_box_1024,
                peak_point_xy=peak_pt,
                gt_mask=gt_mask_1024,
            )

    out = {"weakly_supervised_heatmap": hm_meter.summary()}
    if reg_meter.ious or reg_meter.hits:
        out["supervised_box_head"] = reg_meter.summary()
    return out


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_single_split(args, train_ds, val_ds, test_ds, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = 4 if args.task == "multiclass" else 2
    amp = args.amp and args.device.startswith("cuda")

    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False) if test_ds is not None else val_loader

    rkey = record_label_key(args.task)
    train_labels = [int(r[rkey]) for r in train_ds.records]
    class_weights = compute_class_weights(train_labels, num_classes=n_classes)
    cls_criterion = build_classification_loss(
        loss_type=args.loss,
        class_weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    ).to(args.device)
    box_criterion = (
        BoxRegressionLoss(giou_weight=args.giou_weight, l1_weight=args.l1_weight).to(args.device)
        if args.use_box_head
        else None
    )

    model = ViTLoRA(
        num_classes=n_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_box_head=args.use_box_head,
        dropout=args.dropout,
        model_name=args.model_name,
        local_files_only=args.offline,
    ).to(args.device)

    counts = model.param_counts()
    print(
        f"Mẫu: train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else len(val_ds)} | "
        f"LoRA r={args.lora_r}, alpha={args.lora_alpha} | "
        f"Tham số khả huấn: {counts['trainable']:,}/{counts['total']:,} ({counts['trainable_pct']:.2f}%)"
    )

    optimizer = torch.optim.AdamW(
        model.param_groups(
            lr_lora=args.lr_lora,
            lr_head=args.lr_head,
            lr_box=args.lr_box,
            weight_decay=args.weight_decay,
        )
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    save_json(
        out_dir / "config.json",
        {
            **vars(args),
            "n_classes": n_classes,
            "class_weights": class_weights.tolist(),
            "params": counts,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds) if test_ds else len(val_ds),
        },
    )

    history, best_score, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr, _ = run_epoch(
            model,
            train_loader,
            cls_criterion,
            box_criterion,
            args.device,
            args.task,
            box_loss_weight=args.box_loss_weight,
            threshold=args.threshold,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            desc=f"[E{epoch} train]",
        )
        va, _ = run_epoch(
            model,
            val_loader,
            cls_criterion,
            box_criterion,
            args.device,
            args.task,
            box_loss_weight=args.box_loss_weight,
            threshold=args.threshold,
            amp=amp,
            desc=f"[E{epoch} val]",
        )
        sched.step()
        val_box_iou = va.get("box_head_localization", {}).get("mean_box_iou", float("nan"))
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_cls_loss": tr["cls_loss"],
            "train_box_loss": tr["box_loss"],
            "train_acc": tr["accuracy"],
            "val_loss": va["loss"],
            "val_cls_loss": va["cls_loss"],
            "val_box_loss": va["box_loss"],
            "val_acc": va["accuracy"],
            "val_f1": va["f1"],
            "val_recall": va["recall"],
            "val_auc": va["auc"],
            "val_det_recall_tau": va["detection_tau_tuned"]["recall"],
            "val_mean_box_iou": val_box_iou,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_f1={va['f1']:.4f} val_recall={va['recall']:.4f} val_box_iou={val_box_iou:.4f}",
            flush=True,
        )

        score = va["f1"] + (0.2 * val_box_iou if not np.isnan(val_box_iou) else 0.0)
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = model.trainable_state_dict()
            save_json(out_dir / "val_best.json", {"epoch": epoch, **va})
            model.save_lora_checkpoint(out_dir / "best_lora.pt", extra={"epoch": epoch, "val_f1": va["f1"]})
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"Dừng sớm tại epoch {epoch} (best epoch {best_epoch}).")
                break

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(best_state, strict=False)
    test_metrics, test_meter = run_epoch(
        model,
        test_loader,
        cls_criterion,
        box_criterion,
        args.device,
        args.task,
        box_loss_weight=args.box_loss_weight,
        threshold=args.threshold,
        amp=amp,
        desc="[test]",
    )
    y_true, y_prob = test_meter.arrays()
    thresh_opt = optimize_detection_threshold(y_true, y_prob, target_recall=0.95)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["threshold_optimization"] = thresh_opt["best"]

    if args.prompt_samples > 0:
        eval_ds = test_ds if test_ds is not None else val_ds
        test_metrics["auto_prompt_evaluation"] = evaluate_auto_prompts(
            model,
            eval_ds,
            device=args.device,
            max_samples=args.prompt_samples,
            threshold=args.threshold,
            box_padding=args.box_padding,
        )

    save_json(out_dir / "test_metrics.json", test_metrics)
    np.savez_compressed(out_dir / "test_predictions.npz", y_true=y_true, y_prob=y_prob)
    print(
        f"\nTEST (epoch {best_epoch}): acc={test_metrics['accuracy']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}"
    )
    det_t = test_metrics["detection_tau_tuned"]
    print(
        f"Sàng lọc (τ={args.threshold:.2f}): recall={det_t['recall']:.4f} "
        f"specificity={det_t['specificity']:.4f} f1={det_t['f1']:.4f}"
    )
    return test_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--ann", nargs="+", default=DEFAULT_ANN)
    ap.add_argument("--datasets", nargs="+", default=["brisc2025"])
    ap.add_argument("--lora-r", type=int, default=8, help="Hạng ma trận LoRA r (ví dụ: 4, 8, 16, 32)")
    ap.add_argument("--lora-alpha", type=int, default=16, help="Hệ số tỷ lệ LoRA alpha (mặc định 16)")
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--target-modules", nargs="+", default=["query", "value"])
    ap.add_argument("--loss", choices=["focal", "weighted_ce"], default="focal")
    ap.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing ε (mặc định 0.05)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--use-box-head", dest="use_box_head", action="store_true", default=True)
    ap.add_argument("--no-box-head", dest="use_box_head", action="store_false")
    ap.add_argument("--box-loss-weight", type=float, default=1.0)
    ap.add_argument("--giou-weight", type=float, default=1.0)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.35, help="Ngưỡng xác suất phát hiện khối u τ (mặc định 0.35)")
    ap.add_argument("--box-padding", type=float, default=0.08, help="Hệ số mở rộng hộp δ (mặc định 0.08 = +8%%)")
    ap.add_argument("--prompt-samples", type=int, default=16, help="Số mẫu đánh giá Auto-Prompt (0 = tắt)")
    ap.add_argument("--cv-5fold", action="store_true", help="Chạy 5-Fold Cross-Validation (dành cho Dataset 1)")
    ap.add_argument("--fold", type=int, default=None, help="Chỉ chạy 1 fold cụ thể trong 5-Fold CV (0..4)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-box", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-name", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", "--out-dir", dest="out", required=True)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    out_root = Path(args.out)

    if args.cv_5fold or args.fold is not None:
        folds = [args.fold] if args.fold is not None else list(range(5))
        fold_results = []
        for f in folds:
            print(f"\n==================== FOLD {f} ====================")
            train_ds = build_split_dataset(args, exclude_fold=f, train=True)
            val_ds = build_split_dataset(args, fold=f, train=False)
            res = train_single_split(args, train_ds, val_ds, None, out_root / f"fold_{f}")
            fold_results.append(res)
        if len(fold_results) > 1:
            summary = {
                k: {
                    "mean": float(np.mean([r[k] for r in fold_results])),
                    "std": float(np.std([r[k] for r in fold_results], ddof=1)),
                }
                for k in ("accuracy", "precision", "recall", "f1", "auc")
            }
            save_json(out_root / "cv_summary.json", {"n_folds": len(fold_results), "metrics": summary})
            print(f"\nTổng hợp {len(fold_results)}-Fold CV lưu tại {out_root / 'cv_summary.json'}")
    else:
        train_ds = build_split_dataset(args, split="train", train=True)
        val_ds = build_split_dataset(args, split="val", train=False)
        test_ds = build_split_dataset(args, split="test", train=False)
        train_single_split(args, train_ds, val_ds, test_ds, out_root)


if __name__ == "__main__":
    main()
