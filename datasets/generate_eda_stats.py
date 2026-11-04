"""Xuất thống kê EDA của 4 bộ dữ liệu vào eval/eda_summary/.

Đầu ra: sample_counts.csv, class_distribution.csv, tumor_area_stats.csv,
        class_distribution.png, tumor_area_distribution.png
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from datasets.libs import *

DATASETS = ['navoneel', 'indk214', 'brisc2025', 'brats2020']  # DS1..DS4
CLASS_NAMES = {-1: 'Tumor (unspecified)', 0: 'No Tumor', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'}


def load_records(root, files):
    records = []
    for f in files:
        p = root / f
        if p.exists():
            records.extend(json.loads(p.read_text(encoding='utf-8')))
        else:
            print(f'Cảnh báo: không thấy {p}')
    return records


def tumor_areas(root, records):
    """Diện tích khối u (pixel và % ảnh) từ mask của các mẫu có u và có mask."""
    areas = defaultdict(list)
    for r in records:
        if not (r['has_tumor'] and r['has_mask']):
            continue
        data = np.fromfile(str(root / r['mask_path']), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
        if mask is None:
            continue
        px = int((mask > 0).sum())
        areas[r['dataset']].append((px, 100.0 * px / mask.size))
    return areas


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--json', nargs='+',
                    default=['dataset/unified_2d_annotations.json', 'dataset/brats2020_annotations.json'])
    ap.add_argument('--out', default='eval/eda_summary')
    args = ap.parse_args()

    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(root, args.json)
    if not records:
        raise SystemExit('Không có record nào để thống kê.')

    # 1. Số lượng mẫu theo bộ dữ liệu và tập chia
    counts = Counter((r['dataset'], r.get('split')) for r in records)
    rows = [(ds, sp, n, sum(1 for r in records if r['dataset'] == ds and r.get('split') == sp and r['has_mask']))
            for (ds, sp), n in sorted(counts.items(), key=lambda x: (DATASETS.index(x[0][0]), str(x[0][1])))]
    write_csv(out / 'sample_counts.csv', ['dataset', 'split', 'n_samples', 'n_with_mask'], rows)

    # 2. Phân bố lớp
    cls = Counter((r['dataset'], r['sub_class_label']) for r in records)
    rows = [(ds, CLASS_NAMES.get(c, str(c)), n, round(100.0 * n / sum(v for (d, _), v in cls.items() if d == ds), 2))
            for (ds, c), n in sorted(cls.items(), key=lambda x: (DATASETS.index(x[0][0]), x[0][1]))]
    write_csv(out / 'class_distribution.csv', ['dataset', 'class', 'count', 'percent'], rows)

    present = [d for d in DATASETS if any(r['dataset'] == d for r in records)]
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4), squeeze=False)
    for ax, ds in zip(axes[0], present):
        items = sorted((c, n) for (d, c), n in cls.items() if d == ds)
        ax.bar([CLASS_NAMES.get(c, str(c)) for c, _ in items], [n for _, n in items], color='steelblue')
        ax.set_title(ds)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel('Số mẫu')
    fig.tight_layout()
    fig.savefig(out / 'class_distribution.png', dpi=150)
    plt.close(fig)

    # 3. Phân bố diện tích khối u
    areas = tumor_areas(root, records)
    rows = []
    for ds in present:
        px = np.array([a for a, _ in areas.get(ds, [])])
        if px.size:
            rows.append((ds, px.size, int(px.min()), float(np.median(px)), float(px.mean()), int(px.max())))
        else:
            rows.append((ds, 0, '', '', '', ''))
    write_csv(out / 'tumor_area_stats.csv', ['dataset', 'n_masks', 'min_px', 'median_px', 'mean_px', 'max_px'], rows)

    with_area = [d for d in present if areas.get(d)]
    if with_area:
        fig, axes = plt.subplots(1, len(with_area), figsize=(4.5 * len(with_area), 4), squeeze=False)
        for ax, ds in zip(axes[0], with_area):
            ax.hist([p for _, p in areas[ds]], bins=40, color='indianred')
            ax.set_title(f'{ds} (n={len(areas[ds])})')
            ax.set_xlabel('Diện tích u (% ảnh)')
            ax.set_ylabel('Số mẫu')
        fig.tight_layout()
        fig.savefig(out / 'tumor_area_distribution.png', dpi=150)
        plt.close(fig)

    print(f'Đã xuất thống kê vào {out}')


if __name__ == '__main__':
    main()

