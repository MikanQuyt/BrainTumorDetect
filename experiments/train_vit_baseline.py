"""Run the same ViT backbone as a linear probe and with all weights trainable."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import sklearn
import timm
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.vit_data import collect_samples, make_splits, prepare_images

MODEL = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
SCALARS = ("accuracy", "precision", "recall", "f1", "auc", "tumor_recall")


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_loader(values, labels, indices, batch_size, *, train=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(TensorDataset(values, labels), indices.tolist()),
        batch_size=batch_size,
        shuffle=train,
        generator=generator,
    )


def normalize(batch, norm):
    mean, std = norm
    return (batch.float() / 255 - mean) / std


@torch.no_grad()
def extract_features(backbone, images, norm, device, batch_size, bf16):
    backbone.eval()
    features = []
    for batch in images.split(batch_size):
        x = normalize(batch.to(device), norm)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            features.append(backbone(x).float().cpu())
    return torch.cat(features)


def run_epoch(model, loader, device, norm, bf16, *, optimizer=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    probabilities = []
    with torch.set_grad_enabled(training):
        for batch, target in loader:
            x, target = batch.to(device), target.to(device)
            if norm is not None:
                x = normalize(x, norm)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and norm is not None):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            correct += int((logits.argmax(dim=1) == target).sum())
            count += len(target)
            if collect:
                probabilities.append(logits.float().softmax(dim=1).detach().cpu().numpy())
    stats = {"loss": loss_sum / count, "accuracy": correct / count}
    return stats, np.concatenate(probabilities) if collect else None


def classification_metrics(target, probabilities, classes):
    prediction = probabilities.argmax(axis=1)
    class_ids = np.arange(len(classes))
    matrix = confusion_matrix(target, prediction, labels=class_ids)
    average = "binary" if len(classes) == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, labels=class_ids, average=average, zero_division=0,
    )
    if len(classes) == 2:
        auc = float(roc_auc_score(target, probabilities[:, 1]))
    else:
        auc = float(roc_auc_score(target, probabilities, multi_class="ovr", average="macro"))
    return {
        "accuracy": float((target == prediction).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "tumor_recall": float(((target > 0) & (prediction > 0)).sum() / (target > 0).sum()),
        "class_recall": dict(zip(classes, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
        "confusion_matrix": matrix.tolist(),
    }


def run_trial(backbone, values, labels, classes, split, mode, norm, device, args):
    seed = args.seed + split.fold
    torch.manual_seed(seed)
    head = nn.Linear(backbone.num_features, len(classes)).to(device)
    linear = mode == "linear_probe"
    if linear:
        model, norm, epochs, lr = head, None, args.epochs_linear, args.lr_linear
    else:
        model = nn.Sequential(copy.deepcopy(backbone), head).to(device)
        epochs, lr = args.epochs_full, args.lr_full

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_loader = make_loader(values, labels, split.train, args.batch_size, train=True, seed=seed)
    val_loader = make_loader(values, labels, split.val, args.batch_size)
    test_loader = make_loader(values, labels, split.test, args.batch_size)
    history = []
    best = None
    best_state = None
    for epoch in range(1, epochs + 1):
        train, _ = run_epoch(model, train_loader, device, norm, args.bf16, optimizer=optimizer)
        val, _ = run_epoch(model, val_loader, device, norm, args.bf16)
        row = {
            "mode": mode,
            "fold": split.fold,
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
        }
        history.append(row)
        if best is None or val["loss"] < best["val_loss"]:
            best = row
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"{mode} fold {split.fold} epoch {epoch}/{epochs} "
                f"train_loss={train['loss']:.4f} val_loss={val['loss']:.4f}",
                flush=True,
            )

    model.load_state_dict(best_state)
    test, probabilities = run_epoch(model, test_loader, device, norm, args.bf16, collect=True)
    target = labels[split.test].numpy()
    result = {
        "fold": split.fold,
        "train": len(split.train),
        "validation": len(split.val),
        "test": len(split.test),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch_by_val_loss": best["epoch"],
        "best_epoch": best,
        "last_epoch": history[-1],
        "train_loss_down_val_loss_up_after_best": (
            history[-1]["train_loss"] < best["train_loss"]
            and history[-1]["val_loss"] > best["val_loss"]
        ),
        "test_loss": test["loss"],
        **classification_metrics(target, probabilities, classes),
    }
    print(
        f"{mode} fold {split.fold} test accuracy={result['accuracy']:.4f} "
        f"recall={result['recall']:.4f} best_epoch={best['epoch']}",
        flush=True,
    )
    return history, result, probabilities


def run(args):
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select --device cpu or use a CUDA-enabled PyTorch.")
    if args.bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16. Run without --bf16.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)

    samples, classes = collect_samples(args.dataset, args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [sample.path.relative_to(args.data).as_posix() for sample in samples]
    images, labels, groups = prepare_images(samples, args.image_size)
    splits = make_splits(args.dataset, labels, groups, args.seed)
    split_rows = []
    for split in splits:
        for name, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
            for index in indices:
                split_rows.append({
                    "fold": split.fold,
                    "partition": name,
                    "image": paths[index],
                    "label": classes[labels[index]],
                    "original_split": samples[index].original_split,
                    "duplicate_group": groups[index],
                })
    save_csv(args.output / "splits.csv", split_rows)
    print("Split counts (train/val/test):", [
        (len(s.train), len(s.val), len(s.test)) for s in splits
    ], flush=True)

    weights = {"file": str(args.weights.resolve())} if args.weights else None
    backbone = timm.create_model(
        args.model, pretrained=True, num_classes=0, img_size=args.image_size,
        pretrained_cfg_overlay=weights,
    ).to(device)
    norm = tuple(
        torch.tensor(backbone.pretrained_cfg[key], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        for key in ("mean", "std")
    )
    weights_hash = None
    if args.weights is not None:
        with args.weights.open("rb") as file:
            weights_hash = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "dataset": args.dataset,
        "classes": classes,
        "images": len(samples),
        "class_counts": dict(zip(classes, np.bincount(labels).tolist())),
        "exact_grayscale_groups": len(set(groups)),
        "model": args.model,
        "weights_sha256": weights_hash,
        "seed": args.seed,
        "image_size": args.image_size,
        "device": str(device),
        "bf16_backbone": args.bf16,
        "batch_size": args.batch_size,
        "epochs_linear": args.epochs_linear,
        "epochs_full": args.epochs_full,
        "lr_linear": args.lr_linear,
        "lr_full": args.lr_full,
        "weight_decay": 0.01,
        "checkpoint_selection": "minimum validation cross entropy",
        "recall_average": "binary, tumor is positive" if len(classes) == 2 else "macro over four classes",
        "versions": {"torch": str(torch.__version__), "timm": timm.__version__,
                     "scikit_learn": sklearn.__version__, "opencv": cv2.__version__},
    }
    save_json(args.output / "config.json", metadata)
    image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    label_tensor = torch.from_numpy(labels)
    features = extract_features(backbone, image_tensor, norm, device, args.batch_size, args.bf16)
    summary = {}
    for mode, values in (("linear_probe", features), ("full_finetune", image_tensor)):
        history, results, predictions, all_probabilities, all_indices = [], [], [], [], []
        for split in splits:
            rows, result, probabilities = run_trial(
                backbone, values, label_tensor, classes, split, mode, norm, device, args,
            )
            history.extend(rows)
            results.append(result)
            all_probabilities.append(probabilities)
            all_indices.append(split.test)
            for index, scores in zip(split.test, probabilities):
                predictions.append({
                    "fold": split.fold,
                    "image": paths[index],
                    "true_label": classes[labels[index]],
                    "predicted_label": classes[scores.argmax()],
                    **{f"p_{label}": float(score) for label, score in zip(classes, scores)},
                })

            # Keep completed folds available if an experiment is interrupted.
            save_csv(args.output / f"{mode}_history.csv", history)
            save_csv(args.output / f"{mode}_predictions.csv", predictions)
            save_json(args.output / f"{mode}_folds.json", results)

        summary[mode] = {
            "fold_mean": {key: float(np.mean([row[key] for row in results])) for key in SCALARS},
            "fold_std": {
                key: float(np.std([row[key] for row in results], ddof=1)) if len(results) > 1 else None
                for key in SCALARS
            },
            "pooled_test": classification_metrics(
                labels[np.concatenate(all_indices)], np.concatenate(all_probabilities), classes,
            ),
            "folds": results,
        }
        save_json(args.output / "summary.json", summary)
    print(f"Completed both training modes. Results: {args.output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ds1", "ds2"), required=True)
    parser.add_argument("--data", type=Path, help="Extracted navoneel or indk214 folder.")
    parser.add_argument("--output", type=Path, help="New directory for this run.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--weights", type=Path, help="Optional local pretrained safetensors file.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-linear", type=int, default=25)
    parser.add_argument("--epochs-full", type=int)
    parser.add_argument("--lr-linear", type=float, default=0.001)
    parser.add_argument("--lr-full", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.data is None:
        name = "navoneel" if args.dataset == "ds1" else "indk214"
        args.data = Path.home() / "Downloads" / name
    if args.output is None:
        args.output = Path("runs") / args.dataset
    if args.epochs_full is None:
        args.epochs_full = 20 if args.dataset == "ds1" else 8
    if min(args.epochs_linear, args.epochs_full, args.batch_size, args.threads) <= 0:
        parser.error("Epochs, batch size and threads must be positive.")
    if args.image_size < 32 or args.image_size % 16:
        parser.error("Image size must be at least 32 and divisible by 16.")
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
