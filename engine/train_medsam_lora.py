"""Huấn luyện và đánh giá Nhánh 2: MedSAM-LoRA (Tuần 4).

Kiến trúc & chiến lược huấn luyện:
  * Đóng băng Image Encoder (ViT-B) và Prompt Encoder của MedSAM (`medsam_vit_b.pth`).
  * Chèn ma trận hạng thấp LoRA (mặc định r = 8, α = 16) vào Query & Value của cả 12 khối Attention.
  * Huấn luyện đầy đủ Mask Decoder.
  * Hàm mất mát phân vùng: L_Seg = 0.5 * L_Dice + 0.5 * L_Focal (hoặc BCE).
  * Box Jittering ±10px (hệ tọa độ 1024x1024) trong pha huấn luyện để tăng độ bền vững trước sai số lời nhắc.

Ví dụ sử dụng:
    # 1. Huấn luyện MedSAM-LoRA (r=8) với Focal + Dice Loss và Box Jittering ±10px trên BRISC 2025
    python -m engine.train_medsam_lora --datasets brisc2025 --epochs 15 --batch-size 4 --grad-accum 4 --amp --out runs/medsam_lora_ds3

    # 2. Huấn luyện trên toàn bộ các bộ dữ liệu có nhãn Mask (DS2 + DS3 + DS4) dùng BCE + Dice Loss
    python -m engine.train_medsam_lora --datasets indk214 brisc2025 brats2020 --loss-mode bce --box-jitter 10 --amp --out runs/medsam_lora_all

    # 3. Chạy thử nhanh (smoke test 1 epoch trên 16 mẫu)
    python -m engine.train_medsam_lora --datasets brisc2025 --limit 16 --epochs 1 --batch-size 2 --out runs/smoke_medsam_lora
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

from data.unified_dataset import DEFAULT_ANNOTATION_FILES, UnifiedBrainMRIDataset
from models.medsam_lora import MedSamLoRA, jitter_boxes
from utils.checkpoints import checkpoint_path
from utils.losses import SegmentationLoss
from utils.metrics import SegmentationMeter

DEFAULT_ANN = list(DEFAULT_ANNOTATION_FILES)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_seg_dataset(args, split: str, train: bool) -> UnifiedBrainMRIDataset:
    ds_filter = None if (args.datasets and "all" in [d.lower() for d in args.datasets]) else args.datasets
    ds = UnifiedBrainMRIDataset(args.ann, train=train, datasets=ds_filter, split=split)
    ds.records = [
        r for r in ds.records
        if bool(r.get("has_mask")) and bool(r.get("mask_path")) and int(r.get("has_tumor", 0)) == 1
    ]
    if not ds.records:
        raise SystemExit(
            f"Không tìm thấy mẫu có nhãn Mask + khối u cho datasets={args.datasets}, split='{split}'."
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


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_epoch(
    model: MedSamLoRA,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: str,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    box_jitter_px: float = 0.0,
    grad_accum: int = 1,
    compute_hd95: bool = False,
    threshold: float = 0.5,
    desc: str = "",
) -> dict:
    training = optimizer is not None
    model.train(training)
    meter = SegmentationMeter()

    loss_sum, dice_loss_sum, pixel_loss_sum, n_samples = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    grad_accum = max(int(grad_accum), 1)
    if training:
        optimizer.zero_grad(set_to_none=True)

    with ctx:
        for step, batch in enumerate(loader, 1):
            valid = batch["has_box"].bool() & batch["has_mask"].bool()
            if not valid.any():
                continue

            images = batch["image_1024"][valid].to(device, non_blocking=True)
            boxes = batch["gt_box"][valid].to(device, non_blocking=True)
            masks = batch["gt_mask"][valid].to(device, non_blocking=True).float().unsqueeze(1)

            if training and box_jitter_px > 0.0:
                boxes = jitter_boxes(boxes, max_px=box_jitter_px, img_size=images.shape[-1])

            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                logits = model(images, boxes, upsample=True)
                res = criterion(logits, masks)
                loss = res["loss"]

            if training:
                scaled_loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % grad_accum == 0 or step == len(loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            bs = int(images.shape[0])
            loss_sum += float(loss.detach()) * bs
            dice_loss_sum += float(res["dice_loss"]) * bs
            pixel_loss_sum += float(res["pixel_loss"]) * bs
            n_samples += bs

            if not training or compute_hd95:
                preds_np = (torch.sigmoid(logits.detach().float()) > threshold).squeeze(1).cpu().numpy()
                gts_np = (masks.detach().squeeze(1).cpu().numpy() > 0.5)
                for p_m, g_m in zip(preds_np, gts_np):
                    meter.update(p_m, g_m)
            else:
                # Nhanh trong lúc train: chỉ ghi soft-threshold Dice/IoU trên batch
                preds_b = (torch.sigmoid(logits.detach().float()) > threshold).float().flatten(1)
                gts_b = (masks.detach() > 0.5).float().flatten(1)
                inter = (preds_b * gts_b).sum(dim=1)
                union = (preds_b + gts_b - preds_b * gts_b).sum(dim=1)
                dice_b = (2.0 * inter + 1e-8) / (preds_b.sum(dim=1) + gts_b.sum(dim=1) + 1e-8)
                iou_b = (inter + 1e-8) / (union + 1e-8)
                for d_v, i_v in zip(dice_b.cpu().tolist(), iou_b.cpu().tolist()):
                    meter.rows.append(
                        {"dice": d_v, "iou": i_v, "hd95": float("nan"), "pred_empty": False, "gt_empty": False}
                    )

            if step % 20 == 0:
                print(
                    f"  {desc} step {step}/{len(loader)} loss={loss_sum / n_samples:.4f} "
                    f"(dice_loss={dice_loss_sum / n_samples:.4f}, pixel_loss={pixel_loss_sum / n_samples:.4f})",
                    flush=True,
                )

    summary = meter.summary() if meter.rows else {}
    return {
        "n": n_samples,
        "loss": loss_sum / max(n_samples, 1),
        "dice_loss": dice_loss_sum / max(n_samples, 1),
        "pixel_loss": pixel_loss_sum / max(n_samples, 1),
        "dice": summary.get("dice", {}).get("mean", float("nan")),
        "iou": summary.get("iou", {}).get("mean", float("nan")),
        "hd95": summary.get("hd95", {}).get("mean", float("nan")),
        "segmentation": summary,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", nargs="+", default=DEFAULT_ANN)
    ap.add_argument("--datasets", nargs="+", default=["brisc2025"])
    ap.add_argument("--ckpt", default=None, help="Đường dẫn tới medsam_vit_b.pth (mặc định checkpoints/medsam_vit_b.pth)")
    ap.add_argument("--no-verify-md5", action="store_true", help="Bỏ qua kiểm tra MD5 của checkpoint")
    ap.add_argument("--lora-r", type=int, default=8, help="Hạng LoRA r chèn vào 12 khối Attention (mặc định 8)")
    ap.add_argument("--lora-alpha", type=float, default=16.0, help="Hệ số scaling LoRA alpha (mặc định 16)")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--loss-mode", choices=["focal", "bce"], default="focal",
                    help="Thành phần pixel loss trong L_Seg = 0.5*Dice + 0.5*Focal/BCE")
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--pixel-weight", type=float, default=0.5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--box-jitter", type=float, default=10.0,
                    help="Biên độ Box Jittering ±px trên hệ 1024x1024 khi huấn luyện (0 = tắt)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-decoder", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", choices=["auto", "never"], default="auto",
                    help="Tự động khôi phục từ latest_medsam_lora.pt nếu có (mặc định: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", "--out-dir", dest="out", required=True)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = args.amp and args.device.startswith("cuda")

    train_ds = build_seg_dataset(args, "train", train=True)
    val_ds = build_seg_dataset(args, "val", train=False)
    test_ds = build_seg_dataset(args, "test", train=False)

    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    test_loader = make_loader(test_ds, args, shuffle=False)

    ckpt = Path(args.ckpt) if args.ckpt else checkpoint_path("medsam")
    model = MedSamLoRA.from_checkpoint(
        ckpt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=args.device,
        verify=not args.no_verify_md5,
    )
    counts = model.param_counts()
    print(
        f"MedSAM-LoRA (r={args.lora_r}, α={args.lora_alpha}, blocks={model.n_lora_blocks}) | "
        f"Trainable: {counts['trainable']:,}/{counts['total']:,} ({counts['trainable_pct']:.2f}%) "
        f"[LoRA={counts['lora_params']:,}, MaskDecoder={counts['mask_decoder_params']:,}]"
    )
    print(
        f"Mẫu phân vùng: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | "
        f"Loss: {args.dice_weight}*Dice + {args.pixel_weight}*{args.loss_mode.upper()} | "
        f"Box Jittering: ±{args.box_jitter:.1f}px"
    )

    criterion = SegmentationLoss(
        mode=args.loss_mode,
        dice_weight=args.dice_weight,
        pixel_weight=args.pixel_weight,
        gamma=args.focal_gamma,
        alpha=args.focal_alpha,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.param_groups(lr_lora=args.lr_lora, lr_decoder=args.lr_decoder, weight_decay=args.weight_decay)
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if amp else None

    save_json(
        out_dir / "config.json",
        {
            **vars(args),
            "ckpt": str(ckpt),
            "params": counts,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
    )

    history, best_dice, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    start_epoch = 1
    elapsed_offset = 0.0
    latest_path = out_dir / "latest_medsam_lora.pt"
    if args.resume == "auto" and latest_path.is_file():
        ck = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        best_dice = float(ck["best_dice"])
        best_epoch = int(ck["best_epoch"])
        bad = int(ck["bad"])
        best_state = ck["best_state"]
        history = list(ck.get("history", []))
        elapsed_offset = float(history[-1]["seconds"]) if history else 0.0
        print(
            f"[RESUME] Khôi phục từ {latest_path.name}: tiếp tục từ epoch {start_epoch}/{args.epochs} "
            f"(best_epoch={best_epoch}, best_val_dice={best_dice:.4f}, bad={bad}/{args.patience})",
            flush=True,
        )

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.patience and bad >= args.patience:
            print(f"Đã đạt ngưỡng dừng sớm trước đó tại epoch {start_epoch - 1} (best epoch {best_epoch}).")
            break
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            args.device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            box_jitter_px=args.box_jitter,
            grad_accum=args.grad_accum,
            threshold=args.threshold,
            desc=f"[E{epoch} train]",
        )
        va = run_epoch(
            model,
            val_loader,
            criterion,
            args.device,
            amp=amp,
            box_jitter_px=0.0,
            threshold=args.threshold,
            desc=f"[E{epoch} val]",
        )
        sched.step()
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_dice_loss": tr["dice_loss"],
            "train_pixel_loss": tr["pixel_loss"],
            "train_dice": tr["dice"],
            "val_loss": va["loss"],
            "val_dice_loss": va["dice_loss"],
            "val_pixel_loss": va["pixel_loss"],
            "val_dice": va["dice"],
            "val_iou": va["iou"],
            "val_hd95": va["hd95"],
            "seconds": round(elapsed_offset + (time.time() - t0), 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_dice={va['dice']:.4f} val_iou={va['iou']:.4f} val_hd95={va['hd95']:.2f}px",
            flush=True,
        )
        if va["dice"] > best_dice:
            best_dice, best_epoch, bad = va["dice"], epoch, 0
            best_state = model.trainable_state_dict()
            save_json(out_dir / "val_best.json", {"epoch": epoch, **va})
            model.save_checkpoint(out_dir / "best_medsam_lora.pt", extra={"epoch": epoch, "val_dice": va["dice"]})
        else:
            bad += 1

        # Lưu lịch sử & checkpoint resume sau mỗi epoch để an toàn trên Google Colab
        with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(history)
        tmp_latest = latest_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "bad": bad,
                "model_state": model.trainable_state_dict(),
                "best_state": best_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "history": history,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            tmp_latest,
        )
        tmp_latest.replace(latest_path)

        if args.patience and bad >= args.patience:
            print(f"Dừng sớm tại epoch {epoch} (best epoch {best_epoch}).")
            break

    if best_state is None:
        best_state = model.trainable_state_dict()
    model.load_state_dict(best_state, strict=False)
    test_res = run_epoch(
        model,
        test_loader,
        criterion,
        args.device,
        amp=amp,
        box_jitter_px=0.0,
        compute_hd95=True,
        threshold=args.threshold,
        desc="[test]",
    )
    test_res["best_epoch"] = best_epoch
    save_json(out_dir / "test_metrics.json", test_res)
    print(
        f"\nTEST (epoch {best_epoch}): Dice={test_res['dice']:.4f} "
        f"IoU={test_res['iou']:.4f} HD95={test_res['hd95']:.2f}px"
    )
    print(f"Đã lưu kết quả vào {out_dir.resolve()}")
    return test_res


if __name__ == "__main__":
    main()
