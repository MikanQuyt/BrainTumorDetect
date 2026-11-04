"""Baseline SAM vs MedSAM.

Thí nghiệm 1: SAM Everything Mode (SamAutomaticMaskGenerator, không prompt) trên DS3 và DS1.
Thí nghiệm 2: Zero-shot SAM vs MedSAM với Oracle Box (hộp suy ra từ mask thật) trên DS2, DS3, DS4.

Cài đặt:  python -m pip install git+https://github.com/facebookresearch/segment-anything.git scipy
Chạy từ thư mục gốc dự án:  python -m eval.run_sam_baselines
"""
import argparse
import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from datasets.preprocess_2d_all import mask_to_box
from utils.metrics import dice_score, hd95, iou_score

ROOT = Path(__file__).resolve().parents[1]
CKPTS = {
    'sam': ('sam_vit_b_01ec64.pth', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'),
    'medsam': ('medsam_vit_b.pth', 'https://zenodo.org/records/10689643/files/medsam_vit_b.pth'),
}
JSON_FILES = ['dataset/unified_2d_annotations.json', 'dataset/brats2020_annotations.json']
CLASS_NAMES = {-1: 'tumor', 0: 'no_tumor', 1: 'glioma', 2: 'meningioma', 3: 'pituitary'}


def download_checkpoints(ckpt_dir):
    """Tải sam_vit_b_01ec64.pth và medsam_vit_b.pth vào checkpoints/ (bỏ qua nếu đã có)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, url) in CKPTS.items():
        dst = ckpt_dir / name
        if not dst.exists():
            print(f'Tải {name} ...')
            urllib.request.urlretrieve(url, dst)
        paths[key] = dst
    return paths


def load_model(path, device):
    model = sam_model_registry['vit_b'](checkpoint=str(path))
    return model.to(device).eval()


def read_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return (cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)


@torch.no_grad()
def medsam_predict(model, image, box, device):
    """Suy luận MedSAM theo cách chính thức: resize thẳng 1024x1024, chuẩn hóa [0,1], box theo hệ 1024."""
    h, w = image.shape[:2]
    img = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = (img - img.min()) / max(float(img.max() - img.min()), 1e-8)
    tensor = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
    emb = model.image_encoder(tensor)
    b = np.array(box, dtype=np.float32) / np.array([w, h, w, h], dtype=np.float32) * 1024
    b = torch.as_tensor(b[None, None], dtype=torch.float32, device=device)
    sparse, dense = model.prompt_encoder(points=None, boxes=b, masks=None)
    low, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                multimask_output=False)
    prob = torch.sigmoid(F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False))
    return (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def sam_predict(predictor, image, box):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
    return masks[0].astype(np.uint8)


# ---------------------------------------------------------------- Thí nghiệm 1
def experiment_everything(records, sam, device, args, rng):
    out_dir = ROOT / 'eval/baselines/sam_everything_fail'
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = SamAutomaticMaskGenerator(sam)
    rows = []
    for ds, need_mask in (('brisc2025', True), ('navoneel', False)):
        pool = [r for r in records if r['dataset'] == ds and r['has_tumor']
                and (r['has_mask'] or not need_mask)]
        if ds == 'brisc2025':
            pool = [r for r in pool if r.get('split') == 'test'] or pool
        pick = rng.choice(len(pool), min(args.n_everything, len(pool)), replace=False)
        print(f'[Everything] {ds}: {len(pick)} ảnh')
        for n, i in enumerate(pick):
            r = pool[i]
            image = read_rgb(ROOT / r['img_path'])
            masks = gen.generate(image)
            overlay = image.copy().astype(np.float32)
            for m in masks:
                color = rng.integers(0, 255, 3)
                overlay[m['segmentation']] = 0.5 * overlay[m['segmentation']] + 0.5 * color
            overlay = overlay.astype(np.uint8)

            best_iou = float('nan')
            if r['has_mask']:
                gt = read_mask(ROOT / r['mask_path'])
                best_iou = max((iou_score(m['segmentation'], gt) for m in masks), default=0.0)
                x0, y0, x1, y1 = mask_to_box(gt)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)  # hộp u thật (xanh lá)
            name = f"{ds}_{n:02d}_{Path(r['img_path']).stem}.png"
            cv2.imwrite(str(out_dir / name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            rows.append({'dataset': ds, 'image': r['img_path'], 'n_masks': len(masks),
                         'best_iou_with_tumor': best_iou})
    with open(out_dir / 'everything_stats.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image', 'n_masks', 'best_iou_with_tumor'])
        w.writeheader()
        w.writerows(rows)
    ious = [r['best_iou_with_tumor'] for r in rows if r['best_iou_with_tumor'] == r['best_iou_with_tumor']]
    if ious:
        print(f'[Everything] DS3: IoU tốt nhất giữa mọi mask tự động và khối u: '
              f'trung bình {np.mean(ious):.3f}, tỷ lệ ảnh có IoU < 0.5: {np.mean(np.array(ious) < 0.5):.2%}')
    print(f'[Everything] Ảnh kết quả: {out_dir}')


# ---------------------------------------------------------------- Thí nghiệm 2
def experiment_oracle_box(records, sam, medsam, device, args, rng):
    out_dir = ROOT / 'eval/baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = SamPredictor(sam)
    cases = []
    for ds in ('indk214', 'brisc2025', 'brats2020'):
        pool = [r for r in records if r['dataset'] == ds and r['has_tumor'] and r['has_mask']
                and r.get('split') in ('val', 'test')]
        if len(pool) > args.max_per_dataset:
            pool = [pool[i] for i in rng.choice(len(pool), args.max_per_dataset, replace=False)]
        print(f'[Oracle Box] {ds}: {len(pool)} ảnh val/test')
        for n, r in enumerate(pool):
            image = read_rgb(ROOT / r['img_path'])
            gt = read_mask(ROOT / r['mask_path'])
            box = mask_to_box(gt)
            if box is None:
                continue
            for model_name, pred in (('SAM', sam_predict(predictor, image, box)),
                                     ('MedSAM', medsam_predict(medsam, image, box, device))):
                cases.append({'dataset': ds, 'split': r['split'], 'image': r['img_path'],
                              'class': CLASS_NAMES.get(r['sub_class_label'], str(r['sub_class_label'])),
                              'model': model_name, 'dice': dice_score(pred, gt),
                              'iou': iou_score(pred, gt), 'hd95': hd95(pred, gt)})
            if (n + 1) % 100 == 0:
                print(f'  {ds}: {n + 1}/{len(pool)}', flush=True)

    fields = ['dataset', 'split', 'image', 'class', 'model', 'dice', 'iou', 'hd95']
    with open(out_dir / 'oracle_box_per_case.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cases)

    # Bảng tổng hợp theo (dataset, lớp, mô hình); Dice trung bình của mô hình tốt nhất = mức trần
    groups = defaultdict(list)
    for c in cases:
        groups[(c['dataset'], c['class'], c['model'])].append(c)
    rows = []
    for (ds, cl, model_name), cs in sorted(groups.items()):
        rows.append({'dataset': ds, 'class': cl, 'model': model_name, 'n': len(cs),
                     'dice_mean': np.mean([c['dice'] for c in cs]), 'dice_std': np.std([c['dice'] for c in cs]),
                     'iou_mean': np.mean([c['iou'] for c in cs]),
                     'hd95_mean': np.nanmean([c['hd95'] for c in cs])})
    with open(out_dir / 'oracle_box_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('\n===== ORACLE BOX (Dice trung bình) =====')
    for r in rows:
        print(f"{r['dataset']:10s} {r['class']:12s} {r['model']:7s} n={r['n']:5d} dice={r['dice_mean']:.4f} "
              f"iou={r['iou_mean']:.4f} hd95={r['hd95_mean']:.2f}")
    # Kiểm tra giả thuyết bằng số liệu thực đo, ghép cặp theo ảnh
    for ds in ('indk214', 'brisc2025'):
        sam_d = {c['image']: c['dice'] for c in cases if c['dataset'] == ds and c['class'] == 'glioma' and c['model'] == 'SAM'}
        med_d = {c['image']: c['dice'] for c in cases if c['dataset'] == ds and c['class'] == 'glioma' and c['model'] == 'MedSAM'}
        keys = sorted(set(sam_d) & set(med_d))
        if len(keys) > 5:
            a = np.array([med_d[k] for k in keys])
            b = np.array([sam_d[k] for k in keys])
            p = wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
            print(f'[Glioma {ds}] MedSAM {a.mean():.4f} vs SAM {b.mean():.4f} '
                  f'(MedSAM > SAM: {a.mean() > b.mean()}, Wilcoxon p={p:.3g})')
    print(f'Bảng điểm: {out_dir / "oracle_box_summary.csv"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_everything', type=int, default=50, help='số ảnh mỗi bộ cho Everything Mode')
    ap.add_argument('--max_per_dataset', type=int, default=500, help='số ảnh val/test tối đa mỗi bộ (Oracle Box)')
    ap.add_argument('--skip', nargs='*', default=[], choices=['everything', 'oracle'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import json
    records = []
    for f in JSON_FILES:
        p = ROOT / f
        if p.exists():
            records.extend(json.loads(p.read_text(encoding='utf-8')))
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths = download_checkpoints(ROOT / 'checkpoints')
    sam = load_model(paths['sam'], device)
    if 'everything' not in args.skip:
        experiment_everything(records, sam, device, args, rng)
    if 'oracle' not in args.skip:
        experiment_oracle_box(records, sam, load_model(paths['medsam'], device), device, args, rng)


if __name__ == '__main__':
    main()

