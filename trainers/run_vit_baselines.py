"""ViT-Base baseline: Linear Probing và Full Fine-tuning trên DS1 (5-Fold CV), DS2, DS3.

Chạy từ thư mục gốc dự án:
    python -m trainers.run_vit_baselines --epochs 10
Log từng epoch ghi ở eval/vit_baselines/epoch_log.csv, tổng kết ở summary.json.
Script chỉ ghi lại kết quả thực đo; phần cuối in ra việc các giả thuyết
(overfit khi fine-tune DS1, recall < 88% khi linear probing) có được thỏa hay không.
"""
import os

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # ẩn LOAD REPORT (classifier MISSING / pooler UNEXPECTED là bình thường)

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from datasets.unified_loader import UnifiedBrainMRIDataset, IMAGENET_MEAN, IMAGENET_STD
from utils.metrics import classification_metrics

MODEL_NAME = 'google/vit-base-patch16-224-in21k'
ROOT = Path(__file__).resolve().parents[1]
JSON_FILES = ['dataset/unified_2d_annotations.json']


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_vit_input(x):
    """Loader chuẩn hóa theo ImageNet; ViT in21k dùng mean=std=0.5 nên đổi lại."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x * std + mean) - 0.5) / 0.5


def build_model(mode, num_labels):
    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    for name, p in model.named_parameters():
        # Linear Probing: chỉ mở khóa classifier; Full Fine-tuning: mở toàn bộ
        p.requires_grad = True if mode == 'full' else name.startswith('classifier')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f'  [{mode}] tham số huấn luyện: {n_train:,} / {n_all:,}')
    return model


def indices(records, dataset, split=None, fold=None, not_fold=None):
    out = []
    for i, r in enumerate(records):
        if r['dataset'] != dataset:
            continue
        if split is not None and r.get('split') != split:
            continue
        if fold is not None and r.get('fold') != fold:
            continue
        if not_fold is not None and r.get('fold') == not_fold:
            continue
        out.append(i)
    return out


@torch.no_grad()
def evaluate(model, loader, device, label_key, num_labels):
    model.eval()
    losses, y_cls, y_bin, preds, scores = [], [], [], [], []
    for batch in loader:
        x = to_vit_input(batch['image_224'].to(device))
        y = batch[label_key].to(device)
        logits = model(pixel_values=x).logits
        losses.append(F.cross_entropy(logits, y, reduction='sum').item())
        prob = logits.softmax(-1)
        y_cls.append(y.cpu())
        y_bin.append(batch['binary_label'])
        preds.append(logits.argmax(-1).cpu())
        scores.append((prob[:, 1] if num_labels == 2 else 1 - prob[:, 0]).cpu())
    y_cls, y_bin = torch.cat(y_cls), torch.cat(y_bin)
    preds, scores = torch.cat(preds), torch.cat(scores)
    bin_pred = preds if num_labels == 2 else (preds > 0).long()
    m = classification_metrics(y_bin.numpy(), bin_pred.numpy(), scores.numpy())
    m['loss'] = sum(losses) / len(y_cls)
    m['multiclass_acc'] = float((preds == y_cls).float().mean())
    return m


def run_one(tag, mode, train_idx, val_idx, test_idx, num_labels, label_key, args, device, writer):
    set_seed(args.seed)
    if args.max_train and len(train_idx) > args.max_train:
        train_idx = list(np.random.default_rng(args.seed).choice(train_idx, args.max_train, replace=False))
    records = ds_train.data_records
    model = build_model(mode, num_labels).to(device)
    lr = args.lr_linear if mode == 'linear' else args.lr_full
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    mk = lambda ds, idx, shuffle: DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, drop_last=False)
    train_loader = mk(ds_train, train_idx, True)
    val_loader = mk(ds_eval, val_idx, False)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        t_ep = time.time()
        for batch in tqdm(train_loader, desc=f'  {tag} {mode} ep{epoch:02d}', leave=False):
            x = to_vit_input(batch['image_224'].to(device))
            y = batch[label_key].to(device)
            logits = model(pixel_values=x).logits
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
            loss_sum += loss.item() * len(y)
        val = evaluate(model, val_loader, device, label_key, num_labels)
        row = {'tag': tag, 'mode': mode, 'epoch': epoch, 'train_loss': loss_sum / total,
               'train_acc': correct / total, 'val_loss': val['loss'], 'val_acc': val['multiclass_acc'],
               'val_recall': val['sensitivity'], 'val_specificity': val['specificity'],
               'val_f1': val['f1'], 'val_auc': val['roc_auc']}
        history.append(row)
        writer.writerow(row)
        print(f"  {tag} {mode} ep{epoch:02d} train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
              f"val_recall={row['val_recall']:.4f} val_spec={row['val_specificity']:.4f} "
              f"({time.time() - t_ep:.0f}s)", flush=True)

    result = {'tag': tag, 'mode': mode, 'history': history}
    if test_idx:
        result['test'] = evaluate(model, mk(ds_eval, test_idx, False), device, label_key, num_labels)
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def summarize(results):
    """Gom kết quả theo (dataset, mode): trung bình qua các fold, độ dao động Val Acc."""
    groups = {}
    for r in results:
        groups.setdefault((r['tag'].split('/')[0], r['mode']), []).append(r)
    summary = {}
    for (ds, mode), rs in groups.items():
        last = [r['history'][-1] for r in rs]
        val_curves = [[h['val_acc'] for h in r['history']] for r in rs]
        summary[f'{ds}/{mode}'] = {
            'n_runs': len(rs),
            'final_train_acc': float(np.mean([h['train_acc'] for h in last])),
            'final_val_acc': float(np.mean([h['val_acc'] for h in last])),
            'final_val_recall': float(np.nanmean([h['val_recall'] for h in last])),
            'final_val_specificity': float(np.nanmean([h['val_specificity'] for h in last])),
            'val_acc_std_over_epochs': float(np.mean([np.std(c) for c in val_curves])),
            'val_acc_drop_from_peak': float(np.mean([max(c) - c[-1] for c in val_curves])),
        }
    return summary


def main():
    global ds_train, ds_eval
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['navoneel', 'indk214', 'brisc2025'])
    ap.add_argument('--modes', nargs='+', default=['linear', 'full'], choices=['linear', 'full'])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr_linear', type=float, default=1e-4)
    ap.add_argument('--lr_full', type=float, default=2e-5)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--max_train', type=int, default=0, help='giới hạn số mẫu train (0 = dùng hết), hữu ích khi chạy CPU')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='eval/vit_baselines')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Thiết bị: {device}')
    if device.type == 'cpu':
        print('Cảnh báo: đang chạy CPU, rất chậm với ViT-Base. Nên dùng GPU hoặc thêm --epochs 3 --max_train 200.')
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ds_train = UnifiedBrainMRIDataset(JSON_FILES, train=True)
    ds_eval = UnifiedBrainMRIDataset(JSON_FILES, train=False)
    records = ds_train.data_records

    fields = ['tag', 'mode', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
              'val_recall', 'val_specificity', 'val_f1', 'val_auc']
    results = []
    t0 = time.time()
    with open(out / 'epoch_log.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ds in args.datasets:
            for mode in args.modes:
                if ds == 'navoneel':  # 5-Fold CV, nhãn nhị phân
                    for k in range(args.folds):
                        tr = indices(records, ds, not_fold=k)
                        va = indices(records, ds, fold=k)
                        print(f'== {ds} fold {k} | {mode} | train={len(tr)} val={len(va)}')
                        results.append(run_one(f'{ds}/fold{k}', mode, tr, va, [], 2,
                                               'binary_label', args, device, writer))
                        f.flush()
                else:  # DS2/DS3: 4 lớp, chia train/val/test cố định
                    tr, va, te = (indices(records, ds, split=s) for s in ('train', 'val', 'test'))
                    print(f'== {ds} | {mode} | train={len(tr)} val={len(va)} test={len(te)}')
                    results.append(run_one(f'{ds}/split', mode, tr, va, te, 4,
                                           'sub_class_label', args, device, writer))
                    f.flush()

    summary = summarize(results)
    tests = {f"{r['tag']}/{r['mode']}": r['test'] for r in results if 'test' in r}
    (out / 'summary.json').write_text(json.dumps({'summary': summary, 'test': tests}, indent=2), encoding='utf-8')

    print(f'\n===== TỔNG KẾT ({(time.time() - t0) / 60:.1f} phút) =====')
    for k, v in summary.items():
        print(f"{k:22s} train_acc={v['final_train_acc']:.4f} val_acc={v['final_val_acc']:.4f} "
              f"recall={v['final_val_recall']:.4f} spec={v['final_val_specificity']:.4f} "
              f"val_std={v['val_acc_std_over_epochs']:.4f} drop={v['val_acc_drop_from_peak']:.4f}")

    # Kiểm tra giả thuyết bằng số liệu thực đo (không ép kết quả)
    full, lin = summary.get('navoneel/full'), summary.get('navoneel/linear')
    if full:
        print(f"\n[DS1 Full FT] Train Acc >= 99%: {full['final_train_acc'] >= 0.99} | "
              f"Val thấp hơn Train >= 5 điểm: {full['final_train_acc'] - full['final_val_acc'] >= 0.05} | "
              f"Val dao động (std qua epoch): {full['val_acc_std_over_epochs']:.4f}")
    if lin:
        print(f"[DS1 Linear] Recall < 88%: {lin['final_val_recall'] < 0.88} "
              f"(recall = {lin['final_val_recall']:.4f})")
    print(f'Log chi tiết: {out / "epoch_log.csv"}')


if __name__ == '__main__':
    main()

