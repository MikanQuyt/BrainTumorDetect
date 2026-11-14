"""
Experiment 2: Zero-shot MedSAM Segmentation with Oracle Box Prompt
  - Uses GT segmentation masks to derive tight bounding-box prompts (Oracle Box)
  - Runs official MedSAM (medsam_vit_b.pth, ViT-B) inference on BRISC 2025 Test set
  - Computes Dice, IoU, Precision, Recall, and HD95 per image, per class, and per view

Outputs (in results/segmentation/):
  - medsam_oracle_box_metrics.json
  - medsam_oracle_box_per_image.csv
  - medsam_tumor_class_summary.csv
  - medsam_view_plane_summary.csv
  - medsam_qualitative_grid.png   (4-col grid: Image | GT Mask | Pred Mask | TP/FP/FN Overlay)
"""

import sys
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from utils import (seed_everything, get_device, load_binary_mask, mask_to_bbox,
                   compute_seg_metrics, save_json, save_csv, timestamp)

log = logging.getLogger("medsam_seg")

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from segment_anything import sam_model_registry


def load_medsam_model(device):
    ckpt_path = MEDSAM_CKPT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing MedSAM checkpoint at {ckpt_path}")
    log.info(f"Loading official MedSAM (ViT-B) from {ckpt_path}...")
    model = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def medsam_infer(model, img_np: np.ndarray, bbox, device) -> np.ndarray:
    """Official MedSAM inference pipeline: 1024x1024 min-max [0,1], box in 1024 coords, sigmoid > 0.5."""
    H, W = img_np.shape[:2]
    img_1024 = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_tensor = torch.from_numpy(img_1024).permute(2, 0, 1).unsqueeze(0).to(device)

    box_np = np.array([bbox], dtype=np.float32)
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0
    box_tensor = torch.as_tensor(box_1024, dtype=torch.float32, device=device)[:, None, :]

    image_embedding = model.image_encoder(img_tensor)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    return (low_res_pred.squeeze().cpu().numpy() > 0.5)


def iter_test_samples(limit=None):
    """
    Scan segmentation_task/test to retrieve pairs of (image, mask, tumor_class, view).
    """
    img_files = sorted(SEG_TEST_IMG.glob("*.jpg"))
    if limit:
        step = max(1, len(img_files) // limit)
        img_files = img_files[::step][:limit]

    samples = []
    tumor_map = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor"}
    for img_p in img_files:
        mask_p = SEG_TEST_MASK / (img_p.stem + ".png")
        if not mask_p.exists():
            continue
        parts = img_p.stem.split("_")
        t_code = parts[3] if len(parts) > 3 else "unk"
        v_code = parts[4] if len(parts) > 4 else "unk"
        samples.append({
            "img_path": img_p,
            "mask_path": mask_p,
            "tumor_class": tumor_map.get(t_code, t_code),
            "view": v_code,
            "stem": img_p.stem,
        })
    return samples


def save_qualitative_grid(samples: list, out_path: Path):
    if not samples:
        return
    n_show = min(len(samples), 10)
    fig, axes = plt.subplots(n_show, 4, figsize=(15, n_show * 3.5))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_show):
        s = samples[i]
        img = s["img_np"]
        gt  = s["gt_mask"]
        prd = s["pred_mask"]
        cls = s["tumor_class"]
        box = s["box"]
        dice = s["metrics"]["dice"]
        iou  = s["metrics"]["iou"]
        hd95 = s["metrics"]["hd95"]

        # 1. Original MRI + Oracle Box
        axes[i, 0].imshow(img)
        x0, y0, x1, y1 = box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=2)
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f"MRI + Oracle Box\n({cls.upper()})", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. GT Mask
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(gt, alpha=0.5, cmap="Greens")
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # 3. MedSAM Prediction
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(prd, alpha=0.5, cmap="Blues")
        axes[i, 2].set_title("MedSAM (ViT-B) Prediction", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

        # 4. Error Overlay
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        tp = gt & prd
        fp = (~gt) & prd
        fn = gt & (~prd)
        overlay[tp] = [0, 220, 0]    # Green = TP
        overlay[fp] = [220, 0, 0]    # Red = FP
        overlay[fn] = [255, 165, 0]  # Orange = FN

        axes[i, 3].imshow(img)
        axes[i, 3].imshow(overlay, alpha=0.55)
        axes[i, 3].set_title(f"Overlap (TP/FP/FN)\nDice={dice*100:.1f}% | IoU={iou*100:.1f}% | HD95={hd95:.1f}px", fontsize=10, fontweight="bold")
        axes[i, 3].axis("off")

    patches = [
        mpatches.Patch(color="lime", label="True Positive (TP)"),
        mpatches.Patch(color="red", label="False Positive (FP)"),
        mpatches.Patch(color="orange", label="False Negative (FN)"),
        mpatches.Patch(color="cyan", label="Oracle Bounding Box Prompt"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=11, frameon=True)
    fig.suptitle("Zero-shot MedSAM (medsam_vit_b.pth) with Oracle Box Prompt -- BRISC 2025 Benchmark",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Qualitative grid saved -> {out_path}")


def main():
    seed_everything(SEED)
    device = get_device("cpu")

    model = load_medsam_model(device)

    # Evaluate on a representative set of 50 test slices covering all classes and planes
    eval_limit = SEG_EVAL_LIMIT or 50
    test_samples = iter_test_samples(limit=eval_limit)
    log.info(f"Evaluating Zero-shot MedSAM (medsam_vit_b.pth) on {len(test_samples)} slices...")

    results = []
    qual_samples = []

    t0 = time.time()
    for idx, item in enumerate(test_samples):
        img_p = item["img_path"]
        mask_p = item["mask_path"]
        tumor_cls = item["tumor_class"]
        view = item["view"]

        pil_img = Image.open(img_p).convert("RGB")
        img_np  = np.array(pil_img)
        gt_mask = load_binary_mask(mask_p)

        # Oracle BBox from Ground Truth
        bbox = mask_to_bbox(gt_mask, padding=BOX_PADDING)
        if bbox is None:
            continue

        t_start = time.time()
        pred_mask = medsam_infer(model, img_np, bbox, device)
        inf_time = time.time() - t_start

        m = compute_seg_metrics(pred_mask, gt_mask)
        rec = {
            "image": img_p.name,
            "tumor_class": tumor_cls,
            "view": view,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "hd95": m["hd95"],
            "gt_area": int(gt_mask.sum()),
            "pred_area": int(pred_mask.sum()),
            "inference_time_sec": round(inf_time, 3),
            "box": bbox,
        }
        results.append(rec)

        if len(qual_samples) < 10:
            qual_samples.append({
                "img_np": img_np,
                "gt_mask": gt_mask,
                "pred_mask": pred_mask,
                "tumor_class": tumor_cls,
                "box": bbox,
                "metrics": m,
            })

        if (idx + 1) % 10 == 0:
            running_dice = np.mean([r["dice"] for r in results])
            log.info(f"[{idx+1:02d}/{len(test_samples)}] Running Mean Dice = {running_dice*100:.2f}%")

    total_time = time.time() - t0

    # Aggregations
    mean_dice = float(np.mean([r["dice"] for r in results]))
    mean_iou  = float(np.mean([r["iou"] for r in results]))
    mean_prec = float(np.mean([r["precision"] for r in results]))
    mean_rec  = float(np.mean([r["recall"] for r in results]))
    mean_hd95 = float(np.nanmean([r["hd95"] for r in results]))

    class_stats = {}
    for c in ["glioma", "meningioma", "pituitary"]:
        sub = [r for r in results if r["tumor_class"] == c]
        if sub:
            class_stats[c] = {
                "n_slices": len(sub),
                "mean_dice": float(np.mean([r["dice"] for r in sub])),
                "mean_iou": float(np.mean([r["iou"] for r in sub])),
                "mean_precision": float(np.mean([r["precision"] for r in sub])),
                "mean_recall": float(np.mean([r["recall"] for r in sub])),
                "mean_hd95": float(np.nanmean([r["hd95"] for r in sub])),
            }

    view_stats = {}
    for v in ["ax", "co", "sa"]:
        v_name = {"ax": "axial", "co": "coronal", "sa": "sagittal"}.get(v, v)
        sub = [r for r in results if r["view"] == v]
        if sub:
            view_stats[v_name] = {
                "n_slices": len(sub),
                "mean_dice": float(np.mean([r["dice"] for r in sub])),
                "mean_iou": float(np.mean([r["iou"] for r in sub])),
                "mean_hd95": float(np.nanmean([r["hd95"] for r in sub])),
            }

    summary = {
        "model": "Zero-shot MedSAM (medsam_vit_b.pth, ViT-B)",
        "checkpoint": str(MEDSAM_CKPT.name),
        "checkpoint_md5": "3bb6db55bd0c9ca30b61248bca72f8d6",
        "prompt": "Oracle Bounding Box (derived from GT mask)",
        "n_evaluated_slices": len(results),
        "total_eval_time_sec": round(total_time, 2),
        "avg_time_per_slice_sec": round(total_time / len(results), 3),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_hd95": mean_hd95,
        "per_class": class_stats,
        "per_view": view_stats,
    }

    save_json(summary, SEG_RESULTS / "medsam_oracle_box_metrics.json")
    save_csv(results, SEG_RESULTS / "medsam_oracle_box_per_image.csv")

    cls_rows = [{"tumor_class": k, **v} for k, v in class_stats.items()]
    save_csv(cls_rows, SEG_RESULTS / "medsam_tumor_class_summary.csv")

    view_rows = [{"view_plane": k, **v} for k, v in view_stats.items()]
    save_csv(view_rows, SEG_RESULTS / "medsam_view_plane_summary.csv")

    save_qualitative_grid(qual_samples, SEG_RESULTS / "medsam_qualitative_grid.png")

    print("\n" + "="*60)
    print("  Zero-shot MedSAM (medsam_vit_b.pth, Oracle Box) Results")
    print("="*60)
    print(f"  Slices Evaluated : {len(results)}")
    print(f"  Overall Mean Dice: {mean_dice*100:.2f}%")
    print(f"  Overall Mean IoU : {mean_iou*100:.2f}%")
    print(f"  Mean Precision   : {mean_prec*100:.2f}%")
    print(f"  Mean Recall      : {mean_rec*100:.2f}%")
    print(f"  Mean HD95 (px)   : {mean_hd95:.2f}")
    print("\n  Per-class Breakdown:")
    for c, s in class_stats.items():
        print(f"    {c.upper():<12}: Dice = {s['mean_dice']*100:.2f}% | IoU = {s['mean_iou']*100:.2f}% | HD95 = {s['mean_hd95']:.2f}px | Slices = {s['n_slices']}")
    print("\n  Per-view Plane Breakdown:")
    for v, s in view_stats.items():
        print(f"    {v.upper():<12}: Dice = {s['mean_dice']*100:.2f}% | IoU = {s['mean_iou']*100:.2f}% | HD95 = {s['mean_hd95']:.2f}px | Slices = {s['n_slices']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
