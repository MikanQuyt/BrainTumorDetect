"""Huy's Week 3: honest DS1 5-fold CV, DS2 classification + box supervision, resumable training.

python -m engine.train_vit_lora_huy --dataset ds1 --out runs/huy_week3
python -m engine.train_vit_lora_huy --dataset ds2 --out runs/huy_week3
See experiments/HUONG_DAN_TUAN3_HUY.md and notebooks/train.ipynb.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.prepare_huy_week3 import validate_split
from data.unified_dataset import UnifiedBrainMRIDataset
from engine.train_vit_lora import run_epoch
from models.auto_prompt import AutoPromptGenerator
from models.vit_lora import ViTLoRA
from utils.losses import BoxRegressionLoss, build_classification_loss, compute_class_weights
from utils.metrics import box_iou_score, classification_metrics, detection_metrics

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {2: ["no_tumor", "tumor"], 4: ["no_tumor", "glioma", "meningioma", "pituitary"]}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        torch.save(payload, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if hasattr(info.dataset.transform, "set_random_seed"):
        info.dataset.transform.set_random_seed(seed)


class HuyDataset(UnifiedBrainMRIDataset):
    """Use the shared synchronized transforms; omit 1024 tensors from training batches."""

    def __init__(self, records, root=ROOT, train=False):
        super().__init__(annotation_files=[], root=root, train=train)
        self.records = list(records)

    def full_sample(self, index):
        return super().__getitem__(index)

    def __getitem__(self, index):
        sample = self.full_sample(index)
        return {k: sample[k] for k in ("image_224", "binary_label", "sub_class_label",
                                       "gt_box_norm", "has_box")}


def loader(dataset, args, train=False, epoch=0):
    seed = args.seed + epoch
    if hasattr(dataset.transform, "set_random_seed"):
        dataset.transform.set_random_seed(seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=args.device.startswith("cuda"))


def frozen_digest(model):
    """Checkpoint adapters are usable only with these exact frozen backbone weights."""
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name in frozen:
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])


def load_trainable(model, state):
    expected = set(model.trainable_state_dict())
    if expected != set(state):
        raise ValueError("Checkpoint trainable parameter keys do not match this model")
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint parameters: {unexpected}")


def validate_checkpoint(payload, identity, base_hash):
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if payload["identity"] != identity:
        raise ValueError("Run config/data/code changed. Use the original setup or a NEW output directory.")
    if payload["frozen_sha256"] != base_hash:
        raise ValueError("Frozen pretrained backbone differs from the checkpoint")


def best_payload(latest):
    return {"format_version": 1, "identity": latest["identity"],
            "frozen_sha256": latest["frozen_sha256"],
            "epoch": latest["best_epoch"], "val_loss": latest["best_loss"],
            "model_state": latest["best_state"],
            "checkpoint_kind": "inference: LoRA + classification/box heads; resume with loss_latest.pth"}


def fit(model, train_ds, val_ds, args, out, identity):
    """Save a complete epoch transaction; latest also contains the best adapter snapshot."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out / "loss_latest.pth", out / "loss_best.pth"
    if latest_path.exists() and args.resume == "never":
        raise ValueError(f"Run exists: {out}. Choose a new directory or --resume auto.")
    base_hash = frozen_digest(model)
    optimizer = torch.optim.AdamW(model.param_groups(args.lr_lora, args.lr_head, args.lr_box,
                                                    args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=1e-7)
    amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if amp else None
    labels = [r["has_tumor"] if args.task == "binary" else r["sub_class_label"]
              for r in train_ds.records]
    weights = compute_class_weights(labels, num_classes=model.num_classes).to(args.device)
    cls_loss = build_classification_loss(args.loss, class_weights=weights,
                                         label_smoothing=args.label_smoothing).to(args.device)
    box_loss = BoxRegressionLoss().to(args.device) if model.box_head is not None else None
    start, best_loss, best_epoch, bad, history, best_state = 1, float("inf"), 0, 0, [], None
    if latest_path.exists():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_checkpoint(latest, identity, base_hash)
        load_trainable(model, latest["model_state"])
        optimizer.load_state_dict(latest["optimizer"])
        scheduler.load_state_dict(latest["scheduler"])
        if scaler is not None and latest["scaler"] is not None:
            scaler.load_state_dict(latest["scaler"])
        restore_rng(latest["rng"])
        start = latest["epoch"] + 1
        best_loss, best_epoch, bad = latest["best_loss"], latest["best_epoch"], latest["bad_epochs"]
        history, best_state = latest["history"], latest["best_state"]
        atomic_checkpoint(best_path, best_payload(latest))
        print(f"RESUME {out}: next epoch {start}; best={best_epoch}; bad_epochs={bad}", flush=True)
    for epoch in range(start, args.epochs + 1):
        if bad >= args.patience:
            break
        # Epoch-derived seeds make augmentation and shuffle reproducible on epoch-level resume.
        seed_everything(args.seed + epoch)
        tr, _ = run_epoch(model, loader(train_ds, args, True, epoch), cls_loss, box_loss,
                          args.device, args.task, box_loss_weight=args.box_loss_weight,
                          optimizer=optimizer, scaler=scaler, amp=amp, desc=f"E{epoch} train")
        va, _ = run_epoch(model, loader(val_ds, args), cls_loss, box_loss,
                          args.device, args.task, box_loss_weight=args.box_loss_weight,
                          amp=amp, desc=f"E{epoch} val")
        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            raise ValueError("Non-finite loss; the previous complete checkpoint was preserved")
        scheduler.step(va["loss"])
        # The BEST file is the true minimum validation loss; min_delta controls only patience.
        significant = va["loss"] < best_loss - args.min_delta
        improved = va["loss"] < best_loss
        bad = 0 if significant else bad + 1
        if improved:
            best_loss, best_epoch = va["loss"], epoch
            best_state = model.trainable_state_dict()
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
               "train_cls_loss": tr["cls_loss"], "val_cls_loss": va["cls_loss"],
               "train_box_loss": tr["box_loss"], "val_box_loss": va["box_loss"],
               "train_accuracy": tr["accuracy"], "val_accuracy": va["accuracy"],
               "val_tumor_recall_050": va["detection_tau_default"]["recall"],
               "lr_lora": optimizer.param_groups[0]["lr"], "bad_epochs": bad}
        history.append(row)
        latest = {"format_version": 1, "identity": identity, "frozen_sha256": base_hash,
                  "epoch": epoch, "model_state": model.trainable_state_dict(),
                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                  "scaler": scaler.state_dict() if scaler else None, "rng": rng_state(),
                  "best_loss": best_loss, "best_epoch": best_epoch, "bad_epochs": bad,
                  "best_state": best_state, "history": history}
        atomic_checkpoint(latest_path, latest)
        if improved:
            atomic_checkpoint(best_path, best_payload(latest))
        atomic_csv(out / "history.csv", history)
        print(f"E{epoch}/{args.epochs} train={tr['loss']:.5f} val={va['loss']:.5f} "
              f"best={best_epoch} patience={bad}/{args.patience} lr={row['lr_lora']:.2g}", flush=True)
    if best_state is None:
        raise ValueError("No complete epoch/checkpoint available")
    load_trainable(model, best_state)
    atomic_csv(out / "history.csv", history)
    return {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_completed": len(history), "early_stopped": bad >= args.patience}


def select_threshold(y_val, p_val, target=0.96):
    """Strict Recall >96%; select ONLY on validation within [0.35, 0.40]."""
    curve = [detection_metrics(y_val, p_val, threshold=t / 100) for t in range(35, 41)]
    feasible = [r for r in curve if r["recall"] > target]
    if feasible:
        best = max(feasible, key=lambda r: (r["specificity"], r["f1"], r["threshold"]))
    else:
        best = max(curve, key=lambda r: (r["recall"], r["specificity"], r["threshold"]))
    return {"selected_on": "validation classification samples only", "target_recall_strictly_above": target,
            "validation_target_met": bool(feasible), "best": best,
            "curve": [detection_metrics(y_val, p_val, threshold=0.50)] + curve}


@torch.no_grad()
def predict_classification(model, dataset, args, threshold, fold):
    model.eval()
    labels, probabilities = [], []
    for batch in loader(dataset, args):
        logits = model(batch["image_224"].to(args.device))
        key = "binary_label" if args.task == "binary" else "sub_class_label"
        labels.extend(batch[key].tolist())
        probabilities.extend(logits.float().softmax(1).cpu().tolist())
    y, p = np.array(labels), np.array(probabilities)
    rows = []
    for record, label, scores in zip(dataset.records, y, p):
        pt = float(1 - scores[0])
        rows.append({"sample_id": record["sample_id"], "fold": fold,
                     "true_label": int(label), "predicted_class": int(scores.argmax()),
                     "p_tumor": pt, "threshold": threshold,
                     "predicted_tumor": int(pt >= threshold),
                     **{f"p_{name}": float(v) for name, v in zip(CLASS_NAMES[model.num_classes], scores)}})
    return y, p, rows


def point_in_mask(point, mask):
    """Pointing Game uses predicted peak INSIDE GT tumor mask, not equality to its centroid."""
    x, y = (int(round(float(v))) for v in point)
    h, w = mask.shape
    return bool(0 <= x < w and 0 <= y < h and mask[y, x] > 0)


def summarize_localization(rows):
    if not rows:
        return {"n": 0, "mean_box_iou": None, "pointing_game_accuracy": None}
    return {"n": len(rows), "n_detected": sum(r["detected"] for r in rows),
            "mean_box_iou": float(np.mean([r["box_iou"] for r in rows])),
            "pointing_game_accuracy": float(np.mean([r["point_hit"] for r in rows])),
            "end_to_end_mean_box_iou": float(np.mean([r["box_iou"] if r["detected"] else 0 for r in rows])),
            "end_to_end_pointing_game_accuracy": float(np.mean([bool(r["detected"] and r["point_hit"]) for r in rows]))}


def evaluate_localization(model, dataset, args, threshold, out):
    """Evaluate EVERY held-out DS2 segmentation image, including detector misses."""
    model.eval()
    generator = AutoPromptGenerator(model, mode="box_head", threshold=threshold, box_padding=0.08)
    rows = []
    for i, record in enumerate(dataset.records):
        sample = dataset.full_sample(i)
        # No GT mask/box is passed to generate(). LayerCAM needs gradients at inference.
        result = generator.generate(sample["image_224"].unsqueeze(0).to(args.device))
        box = result["boxes_1024"][0].detach().cpu().numpy()
        point = result["peak_points_1024"][0].detach().cpu().numpy()
        mask = sample["gt_mask"].numpy()
        gt = sample["gt_box"].numpy()
        row = {"sample_id": record["sample_id"], "sub_class_label": record["sub_class_label"],
               "p_tumor": float(result["tumor_prob"][0].detach()),
               "detected": bool(result["has_tumor_pred"][0]),
               "box_iou": box_iou_score(box, gt), "point_hit": point_in_mask(point, mask),
               "pred_box_1024": json.dumps(box.tolist()), "gt_box_1024": json.dumps(gt.tolist()),
               "peak_point_1024": json.dumps(point.tolist())}
        rows.append(row)
        # Incremental evidence is useful if Colab disconnects during the slower CAM evaluation.
        atomic_csv(out / "localization_predictions.csv", rows)
        if (i + 1) % 25 == 0:
            print(f"DS2 localization: {i + 1}/{len(dataset)}", flush=True)
    summary = {"scope": "all held-out DS2 segmentation samples; no GT prompts",
               "box_method": "shared AutoPromptGenerator mode=box_head, padding=0.08",
               "point_method": "predicted Rollout/LayerCAM peak inside GT mask (1024 coordinates)",
               "ungated_vs_end_to_end": "end_to_end assigns zero IoU and a missed point to detector-negative tumors",
               "overall": summarize_localization(rows),
               "per_class": {CLASS_NAMES[4][c]: summarize_localization(
                   [r for r in rows if r['sub_class_label'] == c]) for c in (1, 2, 3)}}
    atomic_json(out / "localization_summary.json", summary)
    return summary


def identity_for(args, dataset, split, records):
    excluded = {"out", "root", "data", "resume", "dataset", "fold", "device", "workers", "task"}
    config = {k: v for k, v in vars(args).items() if k not in excluded}
    fingerprint = hashlib.sha256(json.dumps({"records": records, "split": split},
                                            sort_keys=True).encode()).hexdigest()
    code = hashlib.sha256()
    for filename in ("engine/train_vit_lora_huy.py", "engine/train_vit_lora.py",
                     "data/prepare_huy_week3.py", "data/unified_dataset.py", "data/transforms.py",
                     "models/vit_lora.py", "models/auto_prompt.py", "utils/losses.py", "utils/metrics.py"):
        code.update((ROOT / filename).read_bytes())
    records_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"dataset": dataset, "fold": split["fold"], "task": args.task,
            "config": clean_json(config), "data_sha256": fingerprint,
            "records_sha256": records_hash, "code_sha256": code.hexdigest()}


def preflight(records, split, root, dataset):
    validate_split(records, split)
    paths = [r["img_path"] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Repeated image paths (including the old Y92 collision). Re-run preparation.")
    for record in records:
        for key in ("img_path", "mask_path"):
            if record.get(key) and not (root / record[key]).is_file():
                raise FileNotFoundError(f"Missing {record[key]}; run data.prepare_huy_week3")
    for part in ("train", "val", "test"):
        cls = [records[i] for i in split[part] if records[i]["source"] == "classification"]
        labels = {r['has_tumor'] if dataset == 'ds1' else r['sub_class_label'] for r in cls}
        if labels != ({0, 1} if dataset == 'ds1' else {0, 1, 2, 3}):
            raise ValueError(f"Missing classification class in {part}")
        if dataset == "ds2" and not any(records[i]["has_mask"] for i in split[part]):
            raise ValueError(f"No segmentation masks in DS2 {part}")


def plot_history(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (out / "history.csv").open() as file:
        rows = list(csv.DictReader(file))
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("train", "val"):
        axes[0].plot(epochs, [float(r[f'{name}_loss']) for r in rows], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Total loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(r['lr_lora']) for r in rows])
    axes[1].set(xlabel="Epoch", ylabel="LoRA learning rate", yscale="log")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def run_split(args, dataset, records, split):
    args.task = "binary" if dataset == "ds1" else "multiclass"
    out = args.out / dataset / f"fold_{split['fold']:02d}"
    identity = identity_for(args, dataset, split, records)
    preflight(records, split, args.root, dataset)
    finished_path = out / "test_metrics.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text())
        if finished["identity"] != identity or args.resume == "never":
            raise ValueError(f"Existing completed run has a different config: {out}")
        print(f"Already complete: {out}", flush=True)
        return finished
    seed_everything(args.seed + split["fold"])
    model = ViTLoRA(num_classes=2 if dataset == "ds1" else 4, lora_r=8, lora_alpha=16,
                    use_box_head=dataset == "ds2", sfp_dim=128, model_name=args.model_name,
                    local_files_only=args.offline).to(args.device)
    counts = model.param_counts()
    if counts["num_lora_modules"] != 24:
        raise ValueError("Expected LoRA on query/value in all 12 ViT attention blocks")
    print(f"{dataset} fold {split['fold']}: {counts}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("torch", "transformers", "peft", "albumentations", "numpy"):
        versions[package] = importlib.metadata.version(package)
    atomic_json(out / "config.json", {"identity": identity, "versions": versions,
                                      "params": counts, "model_name": args.model_name,
                                      "base_revision": getattr(model.backbone.config, '_commit_hash', None),
                                      "n_train": len(split['train']), "n_val": len(split['val']),
                                      "n_test": len(split['test'])})
    training = fit(model, HuyDataset([records[i] for i in split['train']], args.root, True),
                   HuyDataset([records[i] for i in split['val']], args.root), args, out, identity)
    cls_val = HuyDataset([records[i] for i in split['val']
                         if records[i]['source'] == 'classification'], args.root)
    cls_test = HuyDataset([records[i] for i in split['test']
                          if records[i]['source'] == 'classification'], args.root)
    y_val, p_val, val_rows = predict_classification(model, cls_val, args, 0.50, split['fold'])
    threshold_result = select_threshold(y_val, p_val)
    tau = threshold_result["best"]["threshold"]
    atomic_json(out / "threshold_selection.json", threshold_result)
    atomic_csv(out / "validation_threshold_curve.csv", threshold_result["curve"])
    atomic_csv(out / "validation_predictions.csv", val_rows)
    # No test data enters checkpoint selection, LR scheduling, early stopping or threshold moving.
    y_test, p_test, test_rows = predict_classification(model, cls_test, args, tau, split['fold'])
    result = {"identity": identity, "training": training, "params": counts,
              "classification": classification_metrics(y_test, p_test, CLASS_NAMES[model.num_classes]),
              "detection_050": detection_metrics(y_test, p_test, threshold=0.50),
              "detection_selected": detection_metrics(y_test, p_test, threshold=tau),
              "threshold_selection": threshold_result,
              "test_target_met": bool(detection_metrics(y_test, p_test, threshold=tau)['recall'] > 0.96)}
    atomic_csv(out / "test_predictions.csv", test_rows)
    if dataset == "ds2":
        loc_ds = HuyDataset([records[i] for i in split['test'] if records[i]['has_mask']], args.root)
        result["localization"] = evaluate_localization(model, loc_ds, args, tau, out)
    plot_history(out)
    # Written last: this marker means ALL classification/localization evaluation completed.
    result = clean_json(result)
    atomic_json(finished_path, result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(clean_json(result), ensure_ascii=False, indent=2), flush=True)
    return result


def aggregate_ds1(out):
    paths = [out / 'ds1' / f'fold_{f:02d}' / 'test_metrics.json' for f in range(1, 6)]
    if not all(path.exists() for path in paths):
        return
    results = [json.loads(path.read_text()) for path in paths]
    rows = []
    for path in paths:
        with path.with_name('test_predictions.csv').open() as file:
            rows.extend(csv.DictReader(file))
    if len(rows) != 253 or len({r['sample_id'] for r in rows}) != 253:
        raise ValueError("DS1 OOF must contain exactly 253 DISTINCT source images")
    identities = [r['identity'] for r in results]
    common = ('config', 'records_sha256', 'code_sha256')
    if any(any(x[key] != identities[0][key] for key in common) for x in identities):
        raise ValueError("Do not pool folds from different configurations/data/code")
    y = np.array([int(r['true_label']) for r in rows])
    p = np.array([[float(r['p_no_tumor']), float(r['p_tumor'])] for r in rows])
    pred = np.array([int(r['predicted_tumor']) for r in rows])
    tp, tn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())
    fp, fn = int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    detection = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "recall": tp / (tp + fn),
                 "specificity": tn / (tn + fp), "precision": tp / (tp + fp) if tp + fp else None,
                 "accuracy": (tp + tn) / len(y), "threshold_policy": "each outer fold uses its own inner-validation threshold"}
    summary = {"n_folds": 5, "n_oof_images": 253,
               "classification_pooled": classification_metrics(y, p, CLASS_NAMES[2]),
               "detection_050_pooled": detection_metrics(y, p, threshold=0.50),
               "detection_selected_pooled": detection,
               "target_met": detection['recall'] > 0.96,
               "fold_recall_mean": float(np.mean([r['detection_selected']['recall'] for r in results])),
               "fold_recall_sample_std": float(np.std([r['detection_selected']['recall'] for r in results], ddof=1)),
               "per_fold_thresholds": [r['threshold_selection']['best']['threshold'] for r in results]}
    atomic_csv(out / 'ds1' / 'oof_predictions.csv', rows)
    atomic_json(out / 'ds1' / 'cv_summary.json', summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ds1', 'ds2', 'all'], default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, default=Path('dataset/huy_week3'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 6), help='Run just one DS1 outer fold')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr-lora', type=float, default=3e-4)
    parser.add_argument('--lr-head', type=float, default=1e-3)
    parser.add_argument('--lr-box', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--box-loss-weight', type=float, default=1.0)
    parser.add_argument('--loss', choices=['focal', 'weighted_ce'], default='focal')
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--model-name', default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--resume', choices=['auto', 'never'], default='auto')
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.lr_patience, args.batch_size) < 1 or args.workers < 0:
        parser.error('epochs/patience/batch-size must be positive; workers must be nonnegative')
    if args.min_delta < 0 or min(args.lr_lora, args.lr_head, args.lr_box) <= 0:
        parser.error('min-delta must be nonnegative; learning rates must be positive')
    if args.fold and args.dataset != 'ds1':
        parser.error('--fold is only valid with --dataset ds1')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; select a GPU runtime or explicitly use --device cpu')
    args.root = args.root.resolve()
    args.data = args.data if args.data.is_absolute() else args.root / args.data
    args.out = args.out.resolve()
    plans = json.loads((args.data / 'splits.json').read_text())
    for dataset in (['ds1', 'ds2'] if args.dataset == 'all' else [args.dataset]):
        records = json.loads((args.data / f'{dataset}_annotations.json').read_text())
        for split in plans[dataset]:
            if args.fold is None or split['fold'] == args.fold:
                run_split(args, dataset, records, split)
        if dataset == 'ds1':
            aggregate_ds1(args.out)


if __name__ == '__main__':
    main()
