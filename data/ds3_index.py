"""Liệt kê và phân tích tên file của Dataset 3 (BRISC 2025: Segmentation & Classification)."""
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Nhãn chuẩn: 0 No Tumor, 1 Glioma, 2 Meningioma, 3 Pituitary
BRISC_CODE = {"no": 0, "nt": 0, "gl": 1, "me": 2, "pi": 3}
FOLDER_CODE = {"no_tumor": "no", "notumor": "no", "glioma": "gl", "meningioma": "me", "pituitary": "pi"}
CLASS_NAME = {0: "no_tumor", 1: "glioma", 2: "meningioma", 3: "pituitary"}
PLANE_NAME = {"ax": "axial", "co": "coronal", "sa": "sagittal"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def parse_brisc_stem(stem):
    """Phân tích tên chuẩn: brisc2025_<split>_<index>_<tumor>_<plane>_<seq>."""
    parts = stem.lower().split("_")
    if len(parts) != 6 or parts[0] != "brisc2025":
        return None
    _, split, idx_str, tumor, plane, seq = parts
    if split not in ("train", "test") or not idx_str.isdigit():
        return None
    if tumor not in BRISC_CODE or plane not in PLANE_NAME:
        return None
    return dict(
        orig_split=split,
        index=int(idx_str),
        tumor_code="no" if tumor == "nt" else tumor,
        label=BRISC_CODE[tumor],
        plane_code=plane,
        sequence=seq.upper(),
    )


def brisc_root(root):
    base = Path(root) / "dataset" / "brisc2025"
    nested = base / "brisc2025"
    return nested if nested.exists() else base


def ds3_segmentation_items(root):
    """Duyệt segmentation_task/{train,test}/{images,masks}, kiểm tra đối sánh 1-1 hai chiều."""
    base = brisc_root(root) / "segmentation_task"
    items, missing_mask, orphan_mask, bad_name = [], [], [], []
    if not base.exists():
        return items, missing_mask, orphan_mask, bad_name

    for split in ("train", "test"):
        img_dir = base / split / "images"
        msk_dir = base / split / "masks"
        imgs = {p.stem: p for p in list_images(img_dir)}
        msks = {p.stem: p for p in list_images(msk_dir)}

        for stem, mpath in sorted(msks.items()):
            if stem not in imgs:
                orphan_mask.append(mpath)

        for stem, ipath in sorted(imgs.items()):
            meta = parse_brisc_stem(stem)
            if meta is None or meta["orig_split"] != split:
                bad_name.append(ipath)
                continue
            mpath = msks.get(stem)
            if mpath is None:
                missing_mask.append(ipath)
                continue
            items.append(dict(
                dataset="brisc2025",
                source="segmentation",
                img=ipath,
                mask=mpath,
                has_tumor=int(meta["label"] > 0),
                label=meta["label"],
                tumor_code=meta["tumor_code"],
                plane_code=meta["plane_code"],
                index=meta["index"],
                orig_split=split,
                stem=stem,
                out_name=f"{stem}.png",
            ))
    return items, missing_mask, orphan_mask, bad_name


def ds3_classification_items(root, only_no_tumor=False):
    """Duyệt classification_task/{train,test}/{glioma,meningioma,pituitary,no_tumor}.

    Lưu ý: 4.793 ảnh u (gl, me, pi) trong classification_task trùng 100% (cùng tên & SHA256)
    với segmentation_task/images; chỉ có 1.207 ảnh no_tumor là mẫu bổ sung không có trong segmentation_task.
    """
    base = brisc_root(root) / "classification_task"
    items, bad_name = [], []
    if not base.exists():
        return items, bad_name

    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            expected_code = FOLDER_CODE.get(cls_dir.name.lower())
            if expected_code is None:
                continue
            if only_no_tumor and expected_code != "no":
                continue
            for ipath in list_images(cls_dir):
                meta = parse_brisc_stem(ipath.stem)
                if meta is None or meta["orig_split"] != split or meta["tumor_code"] != expected_code:
                    bad_name.append(ipath)
                    continue
                items.append(dict(
                    dataset="brisc2025",
                    source="classification",
                    img=ipath,
                    mask=None,
                    has_tumor=int(meta["label"] > 0),
                    label=meta["label"],
                    tumor_code=meta["tumor_code"],
                    plane_code=meta["plane_code"],
                    index=meta["index"],
                    orig_split=split,
                    stem=ipath.stem,
                    out_name=f"{ipath.stem}.png",
                ))
    return items, bad_name

