"""
Experiment 1: ViT Baseline Classification on BRISC 2025
  Mode A: Linear Probing  (freeze backbone, train head only)
  Mode B: Full Fine-tuning (train all layers)

Outputs (in results/classification/):
  - vit_linear_probe_metrics.json
  - vit_full_finetune_metrics.json
  - training_curves_linear_probe.png
  - training_curves_full_finetune.png
  - confusion_matrix_linear_probe.png
  - confusion_matrix_full_finetune.png
  - per_class_linear_probe.csv
  - per_class_full_finetune.csv
"""

import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from utils import seed_everything, get_device, compute_cls_metrics, save_json, save_csv, timestamp

log = logging.getLogger("vit_cls")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def build_transforms(train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_stratified_subset(dataset, samples_per_class, seed=42, exclude_indices=None):
    """Select exactly samples_per_class from each class (disjoint from exclude_indices)."""
    rng = np.random.RandomState(seed)
    excluded = set(exclude_indices or [])
    targets = np.array(dataset.targets)
    indices = []
    for c in range(len(dataset.classes)):
        c_idx = [i for i in np.where(targets == c)[0] if i not in excluded]
        chosen = rng.choice(c_idx, min(len(c_idx), samples_per_class), replace=False)
        indices.extend(chosen.tolist())
    return Subset(dataset, indices)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_curves(history: dict, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    ax1.plot(history["train_loss"], label="Train Loss", color="royalblue", marker='o', lw=2)
    ax1.plot(history["val_loss"],   label="Val Loss",   color="crimson",   marker='s', lw=2)
    ax1.set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train Acc", color="royalblue", marker='o', lw=2)
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val Acc",   color="crimson",   marker='s', lw=2)
    ax2.set_title("Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Curves saved -> {out_path}")


def plot_confusion_matrix(cm, class_names, title: str, out_path: Path):
    import itertools
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    thresh = cm_arr.max() / 2.0
    for i, j in itertools.product(range(cm_arr.shape[0]), range(cm_arr.shape[1])):
        ax.text(j, i, f"{cm_arr[i, j]:d}",
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black", fontsize=11)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved -> {out_path}")


def run_linear_probing(device):
    """
    Linear Probing: Freeze backbone, train classification head.
    Pre-extracts features once using torch.no_grad() -> runs in seconds!
    """
    log.info("\n" + "="*60 + "\n ViT Baseline -- Linear Probing (Frozen Backbone)\n" + "="*60)
    seed_everything(SEED)

    train_raw = datasets.ImageFolder(str(CLS_TRAIN_DIR), transform=build_transforms(False))
    test_raw  = datasets.ImageFolder(str(CLS_TEST_DIR), transform=build_transforms(False))
    classes   = train_raw.classes

    train_sub = get_stratified_subset(train_raw, samples_per_class=250, seed=SEED)  # 1000 train
    val_sub   = get_stratified_subset(train_raw, samples_per_class=50, seed=SEED+1, exclude_indices=train_sub.indices)  # 200 val (disjoint)
    test_sub  = get_stratified_subset(test_raw,  samples_per_class=100, seed=SEED)  # 400 test

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_sub, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_sub, batch_size=32, shuffle=False)

    # Load pretrained ViT backbone
    base_model = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0).to(device)
    base_model.eval()
    embed_dim = base_model.num_features

    def extract_features(loader, name=""):
        log.info(f"Extracting ViT features for {name} ({len(loader.dataset)} samples)...")
        feats, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                f = base_model(x)
                feats.append(f.cpu())
                targets.append(y)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    t0 = time.time()
    X_train, y_train = extract_features(train_loader, "train")
    X_val, y_val     = extract_features(val_loader, "val")
    X_test, y_test   = extract_features(test_loader, "test")

    # Linear classifier head
    head = nn.Linear(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR_LP, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_feat_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_feat_ds   = torch.utils.data.TensorDataset(X_val, y_val)
    test_feat_ds  = torch.utils.data.TensorDataset(X_test, y_test)

    train_feat_loader = DataLoader(train_feat_ds, batch_size=32, shuffle=True)
    val_feat_loader   = DataLoader(val_feat_ds, batch_size=32, shuffle=False)
    test_feat_loader  = DataLoader(test_feat_ds, batch_size=32, shuffle=False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_head_state = 0.0, None

    for epoch in range(1, EPOCHS_LP + 1):
        tr_loss, tr_acc = train_one_epoch(head, train_feat_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate_model(head, val_feat_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        log.info(f"[LP Epoch {epoch:02d}/{EPOCHS_LP}] Train Loss={tr_loss:.4f} Acc={tr_acc*100:.2f}% | Val Loss={vl_loss:.4f} Acc={vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_head_state = head.state_dict().copy()

    elapsed = time.time() - t0
    head.load_state_dict(best_head_state)
    tst_loss, tst_acc, preds, labels = evaluate_model(head, test_feat_loader, criterion, device)
    log.info(f"[LP TEST RESULT] Acc: {tst_acc*100:.2f}%, Loss: {tst_loss:.4f}")

    metrics = compute_cls_metrics(labels, preds, class_names=classes)
    metrics.update({
        "mode": "Linear Probing",
        "model": VIT_MODEL_NAME,
        "epochs": EPOCHS_LP,
        "best_val_acc": best_val_acc,
        "test_acc": tst_acc,
        "test_loss": tst_loss,
        "elapsed_sec": elapsed,
        "history": history,
    })

    save_json(metrics, CLS_RESULTS / "vit_linear_probe_metrics.json")
    plot_curves(history, f"ViT ({VIT_MODEL_NAME}) Linear Probing -- BRISC 2025",
                CLS_RESULTS / "training_curves_linear_probe.png")
    plot_confusion_matrix(metrics["confusion_matrix"], classes,
                          "Confusion Matrix -- ViT Linear Probing",
                          CLS_RESULTS / "confusion_matrix_linear_probe.png")

    rows = [{"class": c, "precision": f"{metrics['report'][c]['precision']:.4f}",
             "recall": f"{metrics['report'][c]['recall']:.4f}",
             "f1-score": f"{metrics['report'][c]['f1-score']:.4f}",
             "support": metrics['report'][c]['support']} for c in classes]
    save_csv(rows, CLS_RESULTS / "per_class_linear_probe.csv")

    torch.save(best_head_state, CKPT_DIR / "vit_linear_probe_head.pth")
    return metrics


def run_full_finetuning(device):
    """
    Full Fine-tuning: Unfreeze all ViT layers with cosine annealing and smaller lr.
    """
    log.info("\n" + "="*60 + "\n ViT Baseline -- Full Fine-Tuning (All Layers Trainable)\n" + "="*60)
    seed_everything(SEED)

    train_raw = datasets.ImageFolder(str(CLS_TRAIN_DIR), transform=build_transforms(True))
    test_raw  = datasets.ImageFolder(str(CLS_TEST_DIR), transform=build_transforms(False))
    classes   = train_raw.classes

    train_sub = get_stratified_subset(train_raw, samples_per_class=40, seed=SEED)  # 160 train (fast & effective on CPU)
    val_sub   = get_stratified_subset(train_raw, samples_per_class=15, seed=SEED+1, exclude_indices=train_sub.indices)  # 60 val (disjoint)
    test_sub  = get_stratified_subset(test_raw,  samples_per_class=50, seed=SEED)   # 200 test

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_sub, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_sub, batch_size=16, shuffle=False)

    model = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=WEIGHT_DECAY)
    epochs_ft = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_ft)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    t0 = time.time()
    for epoch in range(1, epochs_ft + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        log.info(f"[FT Epoch {epoch:02d}/{epochs_ft}] Train Loss={tr_loss:.4f} Acc={tr_acc*100:.2f}% | Val Loss={vl_loss:.4f} Acc={vl_acc*100:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = model.state_dict().copy()

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    tst_loss, tst_acc, preds, labels = evaluate_model(model, test_loader, criterion, device)
    log.info(f"[FT TEST RESULT] Acc: {tst_acc*100:.2f}%, Loss: {tst_loss:.4f}")

    metrics = compute_cls_metrics(labels, preds, class_names=classes)
    metrics.update({
        "mode": "Full Fine-tuning",
        "model": VIT_MODEL_NAME,
        "epochs": epochs_ft,
        "best_val_acc": best_val_acc,
        "test_acc": tst_acc,
        "test_loss": tst_loss,
        "elapsed_sec": elapsed,
        "history": history,
    })

    save_json(metrics, CLS_RESULTS / "vit_full_finetune_metrics.json")
    plot_curves(history, f"ViT ({VIT_MODEL_NAME}) Full Fine-Tuning -- BRISC 2025",
                CLS_RESULTS / "training_curves_full_finetune.png")
    plot_confusion_matrix(metrics["confusion_matrix"], classes,
                          "Confusion Matrix -- ViT Full Fine-Tuning",
                          CLS_RESULTS / "confusion_matrix_full_finetune.png")

    rows = [{"class": c, "precision": f"{metrics['report'][c]['precision']:.4f}",
             "recall": f"{metrics['report'][c]['recall']:.4f}",
             "f1-score": f"{metrics['report'][c]['f1-score']:.4f}",
             "support": metrics['report'][c]['support']} for c in classes]
    save_csv(rows, CLS_RESULTS / "per_class_full_finetune.csv")

    if best_state is not None:
        torch.save(best_state, CKPT_DIR / "vit_full_finetune_best.pth")
    return metrics


def main():
    device = get_device("cpu")
    lp_json = CLS_RESULTS / "vit_linear_probe_metrics.json"
    if lp_json.exists():
        log.info("Linear Probing metrics already computed, loading cached result...")
        with open(lp_json, encoding="utf-8") as f:
            m_lp = json.load(f)
    else:
        m_lp = run_linear_probing(device)

    m_ft = run_full_finetuning(device)

    print("\n" + "="*60)
    print("  ViT Classification Summary (BRISC 2025)")
    print("="*60)
    print(f"  Linear Probing  : Test Acc = {m_lp['test_acc']*100:.2f}% | Macro F1 = {m_lp['macro_f1']*100:.2f}%")
    print(f"  Full Fine-tuning: Test Acc = {m_ft['test_acc']*100:.2f}% | Macro F1 = {m_ft['macro_f1']*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
