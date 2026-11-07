"""Liệt kê file của Dataset 1 (navoneel) và Dataset 2 (indk214: Segmentation & Classification)."""
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Nhãn: 0 No Tumor, 1 Glioma, 2 Meningioma, 3 Pituitary
FOLDER_LABEL = {"glioma": 1, "meningioma": 2, "pituitary": 3, "pituitary tumor": 3,
                "notumor": 0, "no_tumor": 0}
CLASS_NAME = {-1: "tumor (không rõ loại)", 0: "no_tumor", 1: "glioma", 2: "meningioma", 3: "pituitary"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def ds1_items(root):
    """DS1: chỉ có nhãn nhị phân. Bỏ thư mục brain_tumor_dataset/ vì là bản sao của yes/ và no/."""
    base = Path(root) / "dataset" / "navoneel"
    items = []
    for sub, has_tumor in (("no", 0), ("yes", 1)):
        for p in list_images(base / sub):
            items.append(dict(dataset="navoneel", source="classification", img=p, mask=None,
                              has_tumor=has_tumor, label=-1 if has_tumor else 0, orig_split=None,
                              out_name=f"{sub}/{p.stem}.png"))
    return items


def ds2_segmentation_items(root):
    """DS2 Segmentation: cặp enh_N.png <-> enh_N_mask.png trong từng thư mục lớp u."""
    base = Path(root) / "dataset" / "indk214" / "DATASET" / "Segmentation"
    items, missing = [], []
    if not base.exists():
        return items, missing
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        label = FOLDER_LABEL[cls_dir.name.lower()]
        for img in list_images(cls_dir):
            if img.stem.endswith("_mask"):
                continue
            mask = next((cls_dir / f"{img.stem}_mask{e}" for e in IMG_EXTS
                         if (cls_dir / f"{img.stem}_mask{e}").exists()), None)
            if mask is None:
                missing.append(img)
                continue
            tag = f"{cls_dir.name.lower().replace(' ', '_')}_{img.stem}"
            items.append(dict(dataset="indk214", source="segmentation", img=img, mask=mask,
                              has_tumor=1, label=label, orig_split=None, out_name=f"{tag}.png"))
    return items, missing


def ds2_classification_items(root):
    """DS2 Classification: Training/Testing x {glioma, meningioma, notumor, pituitary} (không có mask)."""
    base = Path(root) / "dataset" / "indk214" / "DATASET" / "classification"
    items = []
    for split in ("Training", "Testing"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = FOLDER_LABEL[cls_dir.name.lower()]
            for img in list_images(cls_dir):
                items.append(dict(dataset="indk214", source="classification", img=img, mask=None,
                                  has_tumor=int(label > 0), label=label, orig_split=split.lower(),
                                  out_name=f"{split.lower()}_{cls_dir.name.lower()}_{img.stem}.png"))
    return items

