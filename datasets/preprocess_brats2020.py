"""BraTS2020 -> lát cắt 2D Pseudo-RGB [FLAIR, T1ce, T2] + mặt nạ Whole Tumor.

Đầu vào : dataset/brats2020/BraTS2020_training_data/content/data/volume_X_slice_Y.h5
          (khóa 'image' 4 kênh xung, 'mask' các kênh nhãn NCR/NET, ED, ET)
Đầu ra  : dataset/brats2020_png/{images,masks}/*.png và dataset/brats2020_annotations.json
"""
import argparse
from pathlib import Path

import h5py

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from datasets.libs import *
from datasets.preprocess_2d_all import mask_to_box

# Thứ tự kênh xung trong file h5; nếu ảnh ra sai màu thì chỉnh lại danh sách này
CHANNEL_ORDER = ['FLAIR', 'T1', 'T1ce', 'T2']
RGB_ORDER = ['FLAIR', 'T1ce', 'T2']


def to_channels_last(arr):
    return np.moveaxis(arr, 0, -1) if arr.ndim == 3 and arr.shape[0] <= 4 < arr.shape[-1] else arr


def load_slice(path):
    with h5py.File(path, 'r') as f:
        image = to_channels_last(f['image'][()])
        mask = to_channels_last(f['mask'][()])
    return image, mask


def whole_tumor(mask):
    """WT = (seg > 0): hợp của NCR/NET (1), ED (2), ET (4); hỗ trợ mask one-hot hoặc nhãn đơn."""
    if mask.ndim == 3:
        return (mask > 0).any(axis=-1).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def normalize_channel(ch, low=0.5, high=99.5):
    """Chuẩn hóa một kênh về [0, 255] theo percentile của vùng não (giá trị != 0)."""
    brain = ch[ch != 0]
    if brain.size == 0:
        return np.zeros(ch.shape, np.uint8)
    lo, hi = np.percentile(brain, [low, high])
    out = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1) * 255.0
    out[ch == 0] = 0
    return out.astype(np.uint8)


def pseudo_rgb(image, order=CHANNEL_ORDER, rgb=RGB_ORDER):
    """Ghép [FLAIR, T1ce, T2] thành ảnh 3 kênh RGB uint8."""
    idx = [order.index(n) for n in rgb]
    return np.stack([normalize_channel(image[..., i].astype(np.float32)) for i in idx], axis=-1)


def scan(files, min_brain_frac):
    """Lượt 1: thống kê diện tích u và diện tích não từng lát cắt."""
    stats = []
    for n, p in enumerate(files):
        image, mask = load_slice(p)
        area = int(whole_tumor(mask).sum())
        brain_frac = float((image != 0).any(axis=-1).mean())
        stats.append((p, area, brain_frac))
        if (n + 1) % 5000 == 0:
            print(f'  đã quét {n + 1}/{len(files)}')
    return stats


def parse_name(path):
    stem = Path(path).stem  # volume_X_slice_Y
    parts = stem.split('_')
    return int(parts[1]), int(parts[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--data', default='dataset/brats2020/BraTS2020_training_data/content/data')
    ap.add_argument('--out_dir', default='dataset/brats2020_png')
    ap.add_argument('--out_json', default='dataset/brats2020_annotations.json')
    ap.add_argument('--min_tumor_px', type=int, default=50)
    ap.add_argument('--min_brain_frac', type=float, default=0.05)
    ap.add_argument('--healthy_ratio', type=float, default=1.0, help='số lát lành / số lát có u')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted((root / args.data).glob('volume_*_slice_*.h5'))
    if not files:
        raise SystemExit(f'Không tìm thấy file .h5 trong {root / args.data}')
    print(f'Quét {len(files)} lát cắt...')
    stats = scan(files, args.min_brain_frac)

    tumor = [s for s in stats if s[1] >= args.min_tumor_px and s[2] >= args.min_brain_frac]
    healthy = [s for s in stats if s[1] == 0 and s[2] >= args.min_brain_frac]
    rng = np.random.default_rng(args.seed)
    n_h = min(len(healthy), int(round(len(tumor) * args.healthy_ratio)))
    healthy = [healthy[i] for i in rng.choice(len(healthy), n_h, replace=False)]
    print(f'Giữ {len(tumor)} lát có u, {len(healthy)} lát lành '
          f'(bỏ {len(stats) - len(tumor) - len(healthy)} lát rỗng/u nhỏ)')

    # Chia 70/15/15 theo volume (bệnh nhân) để không rò rỉ giữa các tập
    volumes = sorted({parse_name(p)[0] for p, _, _ in tumor + healthy})
    perm = rng.permutation(volumes)
    n_test, n_val = int(round(len(perm) * 0.15)), int(round(len(perm) * 0.15))
    split_of = {int(v): 'test' for v in perm[:n_test]}
    split_of.update({int(v): 'val' for v in perm[n_test:n_test + n_val]})
    split_of.update({int(v): 'train' for v in perm[n_test + n_val:]})

    out = root / args.out_dir
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)

    records = []
    all_kept = tumor + healthy
    for n, (p, area, _) in enumerate(all_kept):
        if n % 2000 == 0:
            print(f'  đã ghi {n}/{len(all_kept)} lát cắt', flush=True)
        vol, sl = parse_name(p)
        image, mask = load_slice(p)
        wt = whole_tumor(mask)
        name = f'volume_{vol}_slice_{sl}.png'
        rgb = pseudo_rgb(image)
        cv2.imwrite(str(out / 'images' / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / 'masks' / name), wt * 255)
        has_tumor = int(area >= args.min_tumor_px)
        records.append({
            'dataset': 'brats2020',
            'img_path': (out / 'images' / name).relative_to(root).as_posix(),
            'mask_path': (out / 'masks' / name).relative_to(root).as_posix(),
            'has_mask': True,
            'has_tumor': has_tumor,
            'sub_class_label': -1 if has_tumor else 0,
            'gt_box': mask_to_box(wt) if has_tumor else None,
            'volume': vol,
            'slice': sl,
            'split': split_of[vol],
            'fold': None,
        })

    (root / args.out_json).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Đã ghi {len(records)} record -> {root / args.out_json}')


if __name__ == '__main__':
    main()
