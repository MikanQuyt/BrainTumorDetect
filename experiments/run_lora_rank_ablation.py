"""Script tự động khảo sát hạng LoRA (Ablation Study 1): r ∈ {4, 8, 16, 32} (Tuần 3).

Ví dụ sử dụng:
    # 1. Thống kê nhanh số lượng tham số khả huấn của r ∈ {4, 8, 16, 32} so với Linear Probing & Full Fine-tuning
    python -m experiments.run_lora_rank_ablation --summary-only

    # 2. Chạy huấn luyện và đánh giá tuần tự toàn bộ lưới r ∈ {4, 8, 16, 32} trên BRISC 2025 (Dataset 3)
    python -m experiments.run_lora_rank_ablation --datasets brisc2025 --epochs 10 --out runs/ablation_lora_rank
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from transformers import ViTConfig, ViTModel

from engine.train_vit_lora import main as train_vit_lora_main
from models.vit_baseline import ViTBaseline
from models.vit_lora import ViTLoRA

DEFAULT_RANKS = (4, 8, 16, 32)


def build_base_vit_config() -> ViTConfig:
    return ViTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=224,
        patch_size=16,
    )


def summarize_rank_parameters(
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    num_classes: int = 4,
    use_box_head: bool = True,
) -> list[dict]:
    """Bảng đối sánh số lượng tham số giữa Linear Probe, LoRA r ∈ {4, 8, 16, 32} và Full Fine-tune."""
    cfg = build_base_vit_config()
    cfg._attn_implementation = "eager"
    rows = []

    for mode in ("linear_probe", "full_finetune"):
        backbone = ViTModel(cfg, add_pooling_layer=False)
        m = ViTBaseline(num_classes=num_classes, mode=mode, pretrained=False, backbone=backbone)
        c = m.param_counts()
        rows.append(
            {
                "config": mode,
                "lora_r": 0,
                "lora_alpha": 0,
                "total_params": c["total"],
                "trainable_params": c["trainable"],
                "trainable_pct": round(c["trainable_pct"], 4),
                "lora_params": 0,
                "lora_pct_of_backbone": 0.0,
                "box_head_params": 0,
            }
        )

    for r in ranks:
        alpha = 2 * r
        backbone = ViTModel(cfg, add_pooling_layer=False)
        m = ViTLoRA(
            num_classes=num_classes,
            lora_r=r,
            lora_alpha=alpha,
            use_box_head=use_box_head,
            pretrained=False,
            backbone=backbone,
        )
        c = m.param_counts()
        rows.append(
            {
                "config": f"vit_lora_r{r}",
                "lora_r": r,
                "lora_alpha": alpha,
                "total_params": c["total"],
                "trainable_params": c["trainable"],
                "trainable_pct": round(c["trainable_pct"], 4),
                "lora_params": c["lora_params"],
                "lora_pct_of_backbone": round(c["lora_pct_of_backbone"], 4),
                "box_head_params": c["box_head_params"],
            }
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    ap.add_argument("--summary-only", action="store_true", help="Chỉ tính và in bảng tham số của các cấu hình")
    ap.add_argument("--no-box-head", action="store_true", help="Tắt nhánh ViTDetBoxHead khi thống kê/huấn luyện")
    ap.add_argument("--datasets", nargs="+", default=["brisc2025"])
    ap.add_argument("--ann", nargs="+", default=["dataset/ds3_annotations.json"])
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="runs/ablation_lora_rank")
    args = ap.parse_args(argv)

    n_classes = 4 if args.task == "multiclass" else 2
    use_box_head = not args.no_box_head
    param_rows = summarize_rank_parameters(tuple(args.ranks), num_classes=n_classes, use_box_head=use_box_head)

    print(f"{'Config':<16} | {'r':>3} | {'Trainable':>12} | {'Total':>12} | {'Trainable %':>11} | {'LoRA Only':>10}")
    print("-" * 78)
    for row in param_rows:
        print(
            f"{row['config']:<16} | {row['lora_r']:>3} | {row['trainable_params']:>12,} | "
            f"{row['total_params']:>12,} | {row['trainable_pct']:>10.4f}% | {row['lora_params']:>10,}"
        )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "parameter_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(param_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(param_rows)

    if args.summary_only:
        return param_rows

    ablation_results = []
    for r in args.ranks:
        alpha = 2 * r
        run_dir = out_root / f"r{r}"
        t0 = time.time()
        cli = [
            "--task", args.task,
            "--ann", *args.ann,
            "--datasets", *args.datasets,
            "--lora-r", str(r),
            "--lora-alpha", str(alpha),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--out", str(run_dir),
        ]
        if args.no_box_head:
            cli.append("--no-box-head")
        if args.limit > 0:
            cli.extend(["--limit", str(args.limit)])
        if args.offline:
            cli.append("--offline")

        train_vit_lora_main(cli)
        elapsed = round(time.time() - t0, 2)
        ablation_results.append({"lora_r": r, "lora_alpha": alpha, "elapsed_sec": elapsed, "run_dir": str(run_dir)})

    return ablation_results


if __name__ == "__main__":
    main()
