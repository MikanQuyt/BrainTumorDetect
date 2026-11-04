"""Kiểm định & chuẩn hóa gộp 3 bộ dữ liệu 2D -> dataset/unified_2d_annotations.json.

DS1 navoneel : chỉ nhãn nhị phân, dùng cho 5-Fold CV và External Test.
DS2 indk214  : cặp ảnh/mask (Segmentation) + ảnh no-tumor (classification).
DS3 brisc2025: cặp ảnh/mask (segmentation_task, train+test gộp lại rồi chia lại).

Nhãn: 0 No Tumor, 1 Glioma, 2 Meningioma, 3 Pituitary (DS1: -1 cho ảnh có u vì không rõ loại).
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from datasets.libs import *

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
FOLDER_LABEL = {'glioma': 1, 'meningioma': 2, 'pituitary': 3, 'pituitary tumor': 3,
                'notumor': 0, 'no_tumor': 0}
BRISC_CODE = {'no': 0, 'gl': 1, 'me': 2, 'pi': 3}
SEED = 42


def read_gray_or_color(path, flags):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def mask_to_box(mask):
    """Hộp bao sát khối u [xmin, ymin, xmax, ymax] (xmax/ymax là biên loại trừ)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def build_record(root, dataset, img_path, mask_path, label, dropped):
    """Kiểm tra toàn vẹn và tạo record; trả về None nếu mẫu bị loại."""
    image = read_gray_or_color(img_path, cv2.IMREAD_COLOR)
    if image is None:
        dropped.append((str(img_path), 'ảnh hỏng'))
        return None

    has_tumor = int(label > 0)
    mask_rel, box = None, None
    if mask_path is not None:
        mask = read_gray_or_color(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            dropped.append((str(mask_path), 'mask hỏng'))
            return None
        if mask.shape != image.shape[:2]:
            dropped.append((str(img_path), f'lệch kích thước ảnh {image.shape[:2]} / mask {mask.shape}'))
            return None
        box = mask_to_box(mask)
        if has_tumor and box is None:
            dropped.append((str(img_path), 'mask rỗng nhưng nhãn có u'))
            return None
        if not has_tumor:
            box = None
        mask_rel = Path(mask_path).relative_to(root).as_posix()

    return {
        'dataset': dataset,
        'img_path': Path(img_path).relative_to(root).as_posix(),
        'mask_path': mask_rel,
        'has_mask': mask_rel is not None,
        'has_tumor': has_tumor,
        'sub_class_label': label,
        'gt_box': box,
        'group': file_md5(img_path),
    }


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def collect_ds2(root, dropped):
    base = root / 'dataset' / 'indk214' / 'DATASET'
    records = []
    # Ảnh có u kèm mask: enh_N.png <-> enh_N_mask.png
    for cls_dir in sorted((base / 'Segmentation').iterdir()):
        if not cls_dir.is_dir():
            continue
        label = FOLDER_LABEL[cls_dir.name.lower()]
        for img in list_images(cls_dir):
            if img.stem.endswith('_mask'):
                continue
            mask = next((cls_dir / f'{img.stem}_mask{e}' for e in IMG_EXTS
                         if (cls_dir / f'{img.stem}_mask{e}').exists()), None)
            if mask is None:
                dropped.append((str(img), 'không có mask tương ứng'))
                continue
            rec = build_record(root, 'indk214', img, mask, label, dropped)
            if rec:
                records.append(rec)
    # Ảnh no-tumor (không có mask)
    for split in ('Training', 'Testing'):
        d = base / 'classification' / split / 'notumor'
        if d.exists():
            for img in list_images(d):
                rec = build_record(root, 'indk214', img, None, 0, dropped)
                if rec:
                    rec['mask_path'], rec['has_mask'] = None, False
                    records.append(rec)
    return records


def collect_ds3(root, dropped):
    base = root / 'dataset' / 'brisc2025' / 'brisc2025' / 'segmentation_task'
    records = []
    for split in ('train', 'test'):
        for img in list_images(base / split / 'images'):
            code = img.stem.split('_')[3]
            mask = base / split / 'masks' / f'{img.stem}.png'
            if code not in BRISC_CODE or not mask.exists():
                dropped.append((str(img), 'không có mask hoặc mã lớp không hợp lệ'))
                continue
            rec = build_record(root, 'brisc2025', img, mask, BRISC_CODE[code], dropped)
            if rec:
                records.append(rec)
    return records


def collect_ds1(root, dropped):
    base = root / 'dataset' / 'navoneel'
    records = []
    for name, has_tumor in (('no', 0), ('yes', 1)):
        for img in list_images(base / name):
            rec = build_record(root, 'navoneel', img, None, 0 if not has_tumor else -1, dropped)
            if rec:
                rec['has_tumor'] = has_tumor  # build_record suy từ label; DS1 chỉ biết nhị phân
                records.append(rec)
    return records


def split_70_15_15(records):
    """Chia 70/15/15 gộp DS2+DS3; ảnh trùng nội dung (cùng md5) luôn nằm cùng một tập."""
    keys = [f"{r['dataset']}_{r['sub_class_label']}" for r in records]
    y = np.unique(keys, return_inverse=True)[1]
    groups = np.array([r['group'] for r in records])
    # 20 fold ~5% mỗi fold: 3 fold -> test (15%), 3 fold -> val (15%), 14 fold -> train (70%)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        name = 'test' if fold < 3 else 'val' if fold < 6 else 'train'
        for i in idx:
            records[i]['split'] = name
            records[i]['fold'] = None


def assign_ds1_folds(records, n_folds=5):
    """DS1: vừa là 5-Fold CV (trường fold) vừa là External Test độc lập (split)."""
    y = [r['has_tumor'] for r in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        for i in idx:
            records[i]['fold'] = fold
            records[i]['split'] = 'external_test'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--out', default='dataset/unified_2d_annotations.json')
    args = ap.parse_args()
    root = Path(args.root)

    dropped = []
    ds1 = collect_ds1(root, dropped)
    ds23 = collect_ds2(root, dropped) + collect_ds3(root, dropped)

    split_70_15_15(ds23)
    assign_ds1_folds(ds1)

    records = ds23 + ds1
    for r in records:
        r.pop('group')

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Đã ghi {len(records)} record -> {out}')
    print(f'Loại bỏ {len(dropped)} mẫu:')
    for reason, n in Counter(r for _, r in dropped).most_common():
        print(f'  {n:5d}  {reason}')
    for ds in ('indk214', 'brisc2025', 'navoneel'):
        print(ds, dict(Counter((r['split'], r['sub_class_label']) for r in records if r['dataset'] == ds)))


if __name__ == '__main__':
    main()

