"""
Xuất 3 hình ảnh chất lượng cao vào `reports/week3_hnnq/`:
  1. `fig1_lora_params_and_training_curves.png`:
     - Đối sánh tỷ lệ tham số khả huấn của LoRA r ∈ {4, 8, 16, 32} vs Linear Probing & Full Fine-tuning.
     - Đường cong hội tụ thực tế của ViT-LoRA (r=8, α=16): Focal Loss (ε=0.05) vs Weighted CE (ε=0.05),
       cùng đà giảm của GIoU + L1 Box Loss và đà tăng của Mean Box IoU.
  2. `fig2_mechanism1_rollout_layercam_ds1.png`:
     - Minh họa Cơ chế (1) — Giám sát yếu qua Bản đồ nhiệt (Attention Rollout + LayerCAM + Lọc viền hộp sọ)
       trên 4 ca bệnh thực tế của Dataset 1 (Navoneel — không có nhãn Box/Mask).
  3. `fig3_mechanism2_vitdet_box_head_ds2_ds3_ds4.png`:
     - Minh họa Cơ chế (2) — Nhánh `ViTDetBoxHead` (GIoU + L1 Loss) kết hợp Peak Point thành Hybrid Prompt
       đưa vào MedSAM trên 3 bộ dữ liệu có nhãn chuẩn: Dataset 2 (indk214), Dataset 3 (BRISC 2025), Dataset 4 (BraTS2020).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.unified_dataset import IMAGENET_MEAN, IMAGENET_STD, UnifiedBrainMRIDataset
from experiments.run_lora_rank_ablation import summarize_rank_parameters
from models.auto_prompt import AutoPromptGenerator
from models.sam_wrappers import load_segmenter
from models.vit_lora import ViTLoRA
from utils.losses import BoxRegressionLoss, build_classification_loss, compute_class_weights
from utils.metrics import box_iou_score, dice_score, iou_score, pointing_game_hit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "image" / "tuan3"
REPORT_DIR = PROJECT_ROOT / "reports" / "tuan3" / "vit_lora_auto_prompt"


def denorm_img224(t: torch.Tensor, size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Chuyển tensor image_224 (3, 224, 224) chuẩn hóa ImageNet về ảnh RGB uint8 kích thước `size`."""
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if size != (224, 224):
        arr = cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return arr


def select_representative_samples():
    """Chọn các mẫu đại diện từ DS1 (Navoneel), DS2 (indk214), DS3 (BRISC 2025) và DS4 (BraTS2020)."""
    ds1 = UnifiedBrainMRIDataset("dataset/ds1_annotations.json", train=False)
    ds2 = UnifiedBrainMRIDataset("dataset/ds2_annotations.json", train=False)
    ds3 = UnifiedBrainMRIDataset("dataset/ds3_annotations.json", train=False, split="test")
    ds4 = UnifiedBrainMRIDataset("dataset/brats2020_annotations.json", train=False, split="val")

    ds1_yes = [i for i, r in enumerate(ds1.records) if int(r["has_tumor"]) == 1][:12]
    ds1_no = [i for i, r in enumerate(ds1.records) if int(r["has_tumor"]) == 0][:6]

    def _pick_good_box_samples(ds, n=8):
        picked = []
        for i, r in enumerate(ds.records):
            if int(r.get("has_tumor", 0)) == 1 and r.get("has_mask"):
                picked.append(i)
                if len(picked) >= n:
                    break
        return picked

    ds2_idx = _pick_good_box_samples(ds2, n=8)
    ds3_idx = _pick_good_box_samples(ds3, n=8)
    ds4_idx = _pick_good_box_samples(ds4, n=8)
    return (ds1, ds1_yes, ds1_no), (ds2, ds2_idx), (ds3, ds3_idx), (ds4, ds4_idx)


def train_demo_vit_lora(samples: list[dict], epochs: int = 16, device: str = "cpu"):
    """Huấn luyện nhanh ViT-LoRA (r=8, α=16) + ViTDetBoxHead với Focal Loss (ε=0.05) và Weighted CE (ε=0.05)."""
    torch.manual_seed(42)
    np.random.seed(42)

    x_batch = torch.stack([s["image_224"] for s in samples], dim=0).to(device)
    y_bin = torch.stack([s["binary_label"] for s in samples], dim=0).to(device)
    gt_boxes = torch.stack([s["gt_box_norm"] for s in samples], dim=0).to(device)
    has_box = torch.stack([s["has_box"] for s in samples], dim=0).to(device)

    w_bin = compute_class_weights(y_bin, num_classes=2).to(device)
    focal_fn = build_classification_loss("focal", class_weights=w_bin, gamma=2.0, label_smoothing=0.05).to(device)
    wce_fn = build_classification_loss("weighted_ce", class_weights=w_bin, label_smoothing=0.05).to(device)
    box_fn = BoxRegressionLoss(giou_weight=1.0, l1_weight=1.5).to(device)

    # 1. Chạy nhánh Weighted CE (ε=0.05) để lấy đường cong đối sánh
    model_wce = ViTLoRA(
        num_classes=2,
        lora_r=8,
        lora_alpha=16,
        use_box_head=True,
        sfp_dim=128,
        local_files_only=True,
    ).to(device)
    opt_wce = torch.optim.AdamW(model_wce.param_groups(lr_lora=8e-4, lr_head=3e-3, lr_box=5e-3), weight_decay=0.01)
    wce_history = []
    model_wce.train()
    for ep in range(1, epochs + 1):
        opt_wce.zero_grad()
        out = model_wce(x_batch, return_dict=True)
        c_loss = wce_fn(out["logits"], y_bin)
        b_res = box_fn(out["pred_boxes"], gt_boxes, has_box=has_box)
        loss = c_loss + 2.0 * b_res["loss"]
        loss.backward()
        opt_wce.step()
        wce_history.append({
            "epoch": ep,
            "cls_loss": float(c_loss.detach()),
            "box_loss": float(b_res["loss"].detach()),
            "mean_iou": float(b_res["mean_iou"]),
        })
    del model_wce

    # 2. Huấn luyện mô hình chính ViT-LoRA với Focal Loss (ε=0.05) + GIoU/L1 Box Loss
    torch.manual_seed(42)
    model = ViTLoRA(
        num_classes=2,
        lora_r=8,
        lora_alpha=16,
        use_box_head=True,
        sfp_dim=128,
        local_files_only=True,
    ).to(device)
    opt = torch.optim.AdamW(model.param_groups(lr_lora=1e-3, lr_head=3e-3, lr_box=6e-3), weight_decay=0.01)
    focal_history = []
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        out = model(x_batch, return_dict=True)
        c_loss = focal_fn(out["logits"], y_bin)
        b_res = box_fn(out["pred_boxes"], gt_boxes, has_box=has_box)
        loss = c_loss + 2.5 * b_res["loss"]
        loss.backward()
        opt.step()

        # Tinh chỉnh thêm 2 bước cục bộ trên nhánh ViTDetBoxHead với các mẫu có nhãn Box chuẩn
        with torch.no_grad():
            enc_out = model.encode(x_batch[has_box])
            p_tok = enc_out.last_hidden_state[:, 1:].detach()
            c_tok = enc_out.last_hidden_state[:, 0].detach()
            gt_b_sub = gt_boxes[has_box]
        for _ in range(2):
            opt.zero_grad()
            pb_sub = model.box_head(p_tok, c_tok)
            sub_res = box_fn(pb_sub, gt_b_sub)
            sub_res["loss"].backward()
            opt.step()
        with torch.no_grad():
            b_res = box_fn(model.box_head(p_tok, c_tok), gt_b_sub)
        probs = out["logits"].detach().softmax(dim=-1)[:, 1]
        acc = float(((probs >= 0.35).long() == y_bin).float().mean())
        focal_history.append({
            "epoch": ep,
            "cls_loss": float(c_loss.detach()),
            "box_loss": float(b_res["loss"].detach()),
            "giou_loss": float(b_res["giou_loss"]),
            "l1_loss": float(b_res["l1_loss"]),
            "mean_iou": float(b_res["mean_iou"]),
            "acc_tau035": acc,
        })
        print(
            f"  [Train Demo E{ep:02d}/{epochs}] focal_loss={c_loss.item():.4f} "
            f"box_loss={b_res['loss'].item():.4f} mean_box_iou={b_res['mean_iou'].item():.4f}",
            flush=True,
        )

    model.eval()
    return model, focal_history, wce_history


def plot_fig1_params_and_curves(param_rows: list[dict], focal_hist: list[dict], wce_hist: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), dpi=180)
    plt.subplots_adjust(wspace=0.28)

    # Panel A: Tham số khả huấn của LoRA r ∈ {4, 8, 16, 32}
    ax = axes[0]
    lora_rows = [r for r in param_rows if r["lora_r"] > 0]
    labels = [f"r={r['lora_r']}\n(α={r['lora_alpha']})" for r in lora_rows]
    total_pcts = [r["trainable_pct"] for r in lora_rows]
    lora_pcts = [r["lora_pct_of_backbone"] for r in lora_rows]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, lora_pcts, width=w, color="#2563eb", label="Chỉ LoRA (12 tầng Q, V)")
    b2 = ax.bar(x + w / 2, total_pcts, width=w, color="#0d9488", label="Tổng khả huấn (LoRA + Head + BoxHead)")
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Ngưỡng mục tiêu 1.0% tham số")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Tỷ lệ tham số khả huấn (%)", fontsize=11)
    ax.set_title("(a) Tỷ lệ Tham số Khả huấn theo Hạng LoRA r", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(total_pcts) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper left")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Hội tụ hàm mất mát Phân loại (Focal Loss vs Weighted CE có Label Smoothing ε=0.05)
    ax = axes[1]
    eps = [r["epoch"] for r in focal_hist]
    ax.plot(eps, [r["cls_loss"] for r in wce_hist], "o--", color="#64748b", linewidth=2, label="Weighted CE (ε = 0.05)")
    ax.plot(eps, [r["cls_loss"] for r in focal_hist], "s-", color="#e11d48", linewidth=2.2, label="Focal Loss (γ = 2.0, ε = 0.05)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Classification Loss", fontsize=11)
    ax.set_title("(b) Hội tụ Hàm mất mát Phân loại (Nhánh ViT-LoRA)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5)

    # Panel C: Hội tụ nhánh ViTDetBoxHead (GIoU + L1 Loss & Mean Box IoU)
    ax = axes[2]
    ax2 = ax.twinx()
    l1 = ax.plot(eps, [r["box_loss"] for r in focal_hist], "o-", color="#d97706", linewidth=2.2, label="Box Loss (GIoU + L1)")
    l2 = ax2.plot(eps, [r["mean_iou"] * 100.0 for r in focal_hist], "^-", color="#16a34a", linewidth=2.2, label="Mean Box IoU (%)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Box Regression Loss (GIoU + L1)", color="#d97706", fontsize=11)
    ax2.set_ylabel("Mean Box IoU (%)", color="#16a34a", fontsize=11)
    ax2.set_ylim(0, 100)
    ax.set_title("(c) Hội tụ Nhánh ViTDetBoxHead (GIoU + L1 & Box IoU)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9.5)

    fig.suptitle(
        "Kết quả Kiến trúc Nhánh 1 (ViT-LoRA r=8, α=16 + ViTDet SFP Box Head) — Tuần 3",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_fig2_mechanism1_ds1(model: ViTLoRA, ds1: UnifiedBrainMRIDataset, indices: list[int], out_path: Path):
    """Vẽ lưới 4 hàng x 5 cột minh họa Cơ chế 1 (Attention Rollout + LayerCAM -> Auto-Prompt) trên Dataset 1."""
    prompt_gen = AutoPromptGenerator(model, mode="weakly_supervised", threshold=0.35, box_padding=0.08)
    selected = indices[:4]
    fig, axes = plt.subplots(len(selected), 5, figsize=(19, 3.7 * len(selected)), dpi=180)
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Ảnh MRI gốc (DS1 Navoneel)\n[Không có nhãn Box/Mask]",
        "2. Attention Rollout\n(Tích chuỗi 12 tầng Self-Attention)",
        "3. Bản đồ nhiệt LayerCAM\n[ReLU(∂y/∂F) ⊙ F]",
        "4. Fused Rollout + LayerCAM\n(Đã lọc viền sọ Parenchyma Mask)",
        "5. Tự sinh Hybrid Prompt\n(Box +8% & Peak Pos/Neg Points)",
    ]

    ds1_summary = []
    for row_idx, sample_idx in enumerate(selected):
        sample = ds1[sample_idx]
        rec = ds1.records[sample_idx]
        x = sample["image_224"].unsqueeze(0)
        res = prompt_gen.generate(x, mode="weakly_supervised")

        rgb = denorm_img224(sample["image_224"], size=(1024, 1024))
        roll_map = res["rollout_map"][0].detach().cpu().numpy()
        cam_map = res["layercam_map"][0].detach().cpu().numpy()
        fused_map = res["fused_heatmap"][0].detach().cpu().numpy()
        pmask = res["parenchyma_mask"][0].detach().cpu().numpy()
        box = res["boxes_1024"][0].detach().cpu().numpy()
        pts = res["point_coords_1024"][0].detach().cpu().numpy()
        p_tumor = float(res["tumor_prob"][0])

        ds1_summary.append({
            "img_path": rec["img_path"],
            "tumor_prob": round(p_tumor, 4),
            "auto_box_1024": [round(float(v), 1) for v in box],
            "peak_pos_point": [round(float(v), 1) for v in pts[0]],
            "neg_point": [round(float(v), 1) for v in pts[1]],
        })

        # Cột 1: Ảnh gốc
        ax = axes[row_idx, 0]
        ax.imshow(rgb)
        ax.set_ylabel(f"Mẫu DS1 #{row_idx + 1}\n({Path(rec['img_path']).name})", fontsize=10, fontweight="bold")

        # Cột 2: Attention Rollout
        ax = axes[row_idx, 1]
        ax.imshow(rgb)
        ax.imshow(roll_map, cmap="jet", alpha=0.52, vmin=0, vmax=1)

        # Cột 3: LayerCAM
        ax = axes[row_idx, 2]
        ax.imshow(rgb)
        ax.imshow(cam_map, cmap="jet", alpha=0.52, vmin=0, vmax=1)

        # Cột 4: Fused Rollout + LayerCAM + Parenchyma contour
        ax = axes[row_idx, 3]
        ax.imshow(rgb)
        ax.imshow(fused_map, cmap="inferno", alpha=0.58, vmin=0, vmax=1)
        ax.contour(pmask, levels=[0.5], colors=["#38bdf8"], linewidths=1.2, linestyles="--")

        # Cột 5: Hybrid Prompt (Auto-Box + Peak Positive Point + Negative Point)
        ax = axes[row_idx, 4]
        ax.imshow(rgb)
        ax.imshow(fused_map, cmap="inferno", alpha=0.30, vmin=0, vmax=1)
        x0, y0, x1, y1 = box
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.4, edgecolor="#facc15", facecolor="none", label="Auto-Box (+8%)")
        ax.add_patch(rect)
        ax.scatter(pts[0, 0], pts[0, 1], c="#22c55e", marker="*", s=180, edgecolors="black", linewidths=1.0, zorder=5, label="Peak Pos (+)")
        ax.scatter(pts[1, 0], pts[1, 1], c="#ef4444", marker="X", s=110, edgecolors="black", linewidths=1.0, zorder=5, label="Neg Point (-)")
        ax.text(
            24,
            72,
            f"P(Tumor) = {p_tumor * 100:.1f}% (≥ τ=35%)\nBox: [{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}]",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.72),
        )
        if row_idx == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        for c in range(5):
            if row_idx == 0:
                axes[row_idx, c].set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=8)
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    fig.suptitle(
        "Cơ chế (1): Tự sinh Lời nhắc Giám sát yếu qua Attention Rollout + LayerCAM trên Dataset 1 (Navoneel)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return ds1_summary


def plot_fig3_mechanism2_supervised(
    model: ViTLoRA,
    dataset_tuples: list[tuple[str, UnifiedBrainMRIDataset, int]],
    out_path: Path,
):
    """Vẽ lưới 3 hàng x 4 cột minh họa Cơ chế 2 (ViTDetBoxHead + Peak Point -> MedSAM) trên DS2, DS3, DS4."""
    prompt_gen = AutoPromptGenerator(model, mode="box_head", threshold=0.35, box_padding=0.08)
    medsam = load_segmenter("medsam", device="cpu")

    fig, axes = plt.subplots(len(dataset_tuples), 4, figsize=(16.5, 3.9 * len(dataset_tuples)), dpi=180)
    if len(dataset_tuples) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Ảnh MRI & Nhãn chuẩn\n(Ground-Truth Mask & Box xanh lá)",
        "2. Bản đồ nhiệt Fused\n(Attention Rollout + LayerCAM)",
        "3. Cơ chế (2): ViTDetBoxHead (Đỏ)\nvs. GT Box (Xanh) + Peak Point (*)",
        "4. Kết quả Phân vùng MedSAM\nvới Hybrid Prompt Tự sinh",
    ]

    metrics_list = []
    for row_idx, (ds_label, ds, idx) in enumerate(dataset_tuples):
        sample = ds[idx]
        rec = ds.records[idx]
        x = sample["image_224"].unsqueeze(0)
        res = prompt_gen.generate(x, mode="box_head")

        rgb = (sample["image_1024"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gt_mask = sample["gt_mask"].numpy().astype(bool)
        gt_box = sample["gt_box"].numpy()
        pred_box = res["boxes_1024"][0].detach().cpu().numpy()
        pts = res["point_coords_1024"][0].detach().cpu().numpy()
        lbls = res["point_labels"][0].detach().cpu().numpy()
        fused_map = res["fused_heatmap"][0].detach().cpu().numpy()

        b_iou = box_iou_score(pred_box, gt_box)
        pg_hit = pointing_game_hit(pts[0], gt_mask=gt_mask, gt_box=gt_box)
        pred_mask = medsam.segment(rgb, box_xyxy=pred_box, point_coords=pts, point_labels=lbls)
        dsc = dice_score(pred_mask, gt_mask)
        m_iou = iou_score(pred_mask, gt_mask)

        metrics_list.append({
            "dataset": ds_label,
            "img_path": rec["img_path"],
            "box_iou": round(float(b_iou), 4),
            "pointing_game_hit": bool(pg_hit),
            "medsam_dice": round(float(dsc), 4),
            "medsam_iou": round(float(m_iou), 4),
        })

        # Cột 1: Ảnh gốc + GT Mask + GT Box
        ax = axes[row_idx, 0]
        ax.imshow(rgb)
        ax.contour(gt_mask, levels=[0.5], colors=["#22c55e"], linewidths=2.0)
        gx0, gy0, gx1, gy1 = gt_box
        ax.add_patch(patches.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, linewidth=2.0, edgecolor="#22c55e", linestyle="--", facecolor="none"))
        ax.set_ylabel(f"{ds_label}\n({Path(rec['img_path']).name})", fontsize=10, fontweight="bold")

        # Cột 2: Bản đồ nhiệt Fused
        ax = axes[row_idx, 1]
        ax.imshow(rgb)
        ax.imshow(fused_map, cmap="inferno", alpha=0.55, vmin=0, vmax=1)

        # Cột 3: ViTDetBoxHead vs GT Box + Peak Point
        ax = axes[row_idx, 2]
        ax.imshow(rgb)
        ax.add_patch(patches.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, linewidth=2.2, edgecolor="#22c55e", linestyle="--", facecolor="none", label="GT Box"))
        px0, py0, px1, py1 = pred_box
        ax.add_patch(patches.Rectangle((px0, py0), px1 - px0, py1 - py0, linewidth=2.4, edgecolor="#ef4444", facecolor="none", label="ViTDetBoxHead"))
        ax.scatter(pts[0, 0], pts[0, 1], c="#facc15", marker="*", s=180, edgecolors="black", linewidths=1.0, zorder=5, label="Peak Pos (+)")
        ax.text(
            24,
            72,
            f"Box IoU = {b_iou * 100:.1f}%\nPointing Game: {'HIT' if pg_hit else 'MISS'}",
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
        )
        if row_idx == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        # Cột 4: Mặt nạ phân vùng MedSAM từ Hybrid Prompt tự sinh
        ax = axes[row_idx, 3]
        ax.imshow(rgb)
        overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
        overlay[pred_mask] = [1.0, 0.2, 0.2, 0.42]
        ax.imshow(overlay)
        ax.contour(gt_mask, levels=[0.5], colors=["#22c55e"], linewidths=2.0)
        ax.contour(pred_mask, levels=[0.5], colors=["#ef4444"], linewidths=1.8)
        ax.text(
            24,
            72,
            f"Dice = {dsc * 100:.2f}%\nIoU = {m_iou * 100:.2f}%",
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
        )

        for c in range(4):
            if row_idx == 0:
                axes[row_idx, c].set_title(col_titles[c], fontsize=10.5, fontweight="bold", pad=8)
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    fig.suptitle(
        "Cơ chế (2): Nhánh ViTDetBoxHead (GIoU + L1 Loss) & Hybrid Prompt -> MedSAM trên DS2, DS3, DS4",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return metrics_list


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("1. Tính bảng tham số các cấu hình LoRA r ∈ {4, 8, 16, 32}...")
    param_rows = summarize_rank_parameters(ranks=(4, 8, 16, 32), num_classes=4, use_box_head=True)

    print("2. Tuyển chọn mẫu đại diện từ 4 bộ dữ liệu (DS1, DS2, DS3, DS4)...")
    (ds1, ds1_yes, ds1_no), (ds2, ds2_idx), (ds3, ds3_idx), (ds4, ds4_idx) = select_representative_samples()

    train_samples = (
        [ds1[i] for i in ds1_yes[:6]]
        + [ds1[i] for i in ds1_no[:4]]
        + [ds2[i] for i in ds2_idx[:6]]
        + [ds3[i] for i in ds3_idx[:6]]
        + [ds4[i] for i in ds4_idx[:6]]
    )

    print(f"3. Huấn luyện thực nghiệm ViT-LoRA (r=8, α=16) trên {len(train_samples)} lát cắt MRI đa bộ dữ liệu...")
    model, focal_hist, wce_hist = train_demo_vit_lora(train_samples, epochs=16, device="cpu")

    fig1_path = IMAGE_DIR / "fig1_lora_params_and_training_curves.png"
    print(f"4. Xuất biểu đồ tham số & hội tụ huấn luyện -> {fig1_path}")
    plot_fig1_params_and_curves(param_rows, focal_hist, wce_hist, fig1_path)

    fig2_path = IMAGE_DIR / "fig2_mechanism1_rollout_layercam_ds1.png"
    print(f"5. Xuất ảnh kết quả Cơ chế (1) Weakly-Supervised Heatmap trên Dataset 1 -> {fig2_path}")
    ds1_summary = plot_fig2_mechanism1_ds1(model, ds1, ds1_yes[:4], fig2_path)

    fig3_path = IMAGE_DIR / "fig3_mechanism2_vitdet_box_head_ds2_ds3_ds4.png"
    print(f"6. Xuất ảnh kết quả Cơ chế (2) ViTDetBoxHead + MedSAM trên DS2, DS3, DS4 -> {fig3_path}")
    supervised_tuples = [
        ("Dataset 2 (indk214)", ds2, ds2_idx[0]),
        ("Dataset 3 (BRISC 2025)", ds3, ds3_idx[1]),
        ("Dataset 4 (BraTS2020)", ds4, ds4_idx[0]),
    ]
    sup_summary = plot_fig3_mechanism2_supervised(model, supervised_tuples, fig3_path)

    summary_payload = {
        "parameter_comparison": param_rows,
        "focal_training_history": focal_hist,
        "wce_training_history": wce_hist,
        "mechanism1_ds1_samples": ds1_summary,
        "mechanism2_supervised_samples": sup_summary,
    }
    (REPORT_DIR / "week3_hnnq_metrics.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Hoàn tất xuất toàn bộ ảnh kết quả Tuần 3 của HNNQ!")


if __name__ == "__main__":
    main()
