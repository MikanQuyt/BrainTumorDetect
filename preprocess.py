from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image

DOWNLOADS = Path.home() / "Downloads"
NAVONEEL_DIR = DOWNLOADS / "navoneel"
INDK214_DIR = DOWNLOADS / "indk214"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

Box = tuple[int, int, int, int]


def grayscale(image: Image.Image) -> np.ndarray:
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, rgba)
    return np.asarray(image.convert("L"))


def head_box(gray: np.ndarray) -> tuple[Box, str]:
    """Find a padded head crop from the MRI intensities alone."""
    height, width = gray.shape
    full_frame = (0, 0, width, height)
    scale = min(1.0, 512 / max(height, width))
    if scale < 1:
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    h, w = gray.shape
    short_side = min(h, w)
    if short_side < 16:
        return full_frame, "low_confidence"

    smooth = cv2.GaussianBlur(gray, (5, 5), sigmaX=0.9)
    corner = max(3, round(short_side * 0.05))
    background = float(np.median([
        np.median(smooth[:corner, :corner]),
        np.median(smooth[:corner, -corner:]),
        np.median(smooth[-corner:, :corner]),
        np.median(smooth[-corner:, -corner:]),
    ]))
    center = smooth[h // 4:3 * h // 4, w // 4:3 * w // 4]
    if background > 170 and np.median(center) < background - 25:
        smooth = 255 - smooth
        background = 255 - background

    histogram = np.bincount(smooth.ravel(), minlength=256)
    bright = int(np.searchsorted(histogram.cumsum(), math.ceil(h * w * 0.99)))
    if bright - background < 15:
        return full_frame, "low_confidence"
    level = background + 0.12 * (bright - background)
    foreground = (smooth > level).astype(np.uint8)

    radius = max(1, round(0.006 * short_side))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    head = max(contours, key=cv2.contourArea, default=None)
    if head is None or cv2.contourArea(head) < 0.035 * h * w:
        return full_frame, "low_confidence"
    x, y, box_w, box_h = cv2.boundingRect(head)
    if box_w < 0.25 * w or box_h < 0.25 * h:
        return full_frame, "low_confidence"

    pad = max(3, round(0.025 * min(height, width)))
    box = (
        max(0, math.floor(x * width / w) - pad),
        max(0, math.floor(y * height / h) - pad),
        min(width, math.ceil((x + box_w) * width / w) + pad),
        min(height, math.ceil((y + box_h) * height / h) + pad),
    )
    return box, "full_frame" if box == full_frame else "cropped"


def mask_box(mask: np.ndarray) -> Box | None:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def contains(outer: Box, inner: Box) -> bool:
    x1, y1, x2, y2 = outer
    a1, b1, a2, b2 = inner
    return x1 <= a1 and y1 <= b1 and a2 <= x2 and b2 <= y2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
INDK_CLASSES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "notumor": "no_tumor",
    "pituitary": "pituitary",
    "pituitary tumor": "pituitary",
}
MASK_THRESHOLD = 128
EXPECTED_IMAGES = {
    "navoneel": 253,
    "indk214_classification": 7023,
    "indk214_segmentation": 2192,
}


@dataclass(frozen=True)
class Sample:
    image: PurePosixPath
    group: str
    label: str
    split: str = "unsplit"
    mask: PurePosixPath | None = None


@dataclass(frozen=True)
class Measurement:
    label: str
    split: str
    width: int
    height: int
    mode: str
    extension: str
    gray_digest: tuple[int, int, bytes]
    crop_status: str
    crop_removed_percent: float
    mask_pixels: int | None
    mask_bbox_size: tuple[int, int] | None


def image_members(root: Path) -> set[PurePosixPath]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    return {
        PurePosixPath(file.relative_to(root).as_posix())
        for file in root.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
    }


def navoneel_samples(root: Path, paths: set[PurePosixPath]) -> tuple[list[Sample], int]:
    direct = {p for p in paths if len(p.parts) == 2 and p.parts[0] in {"yes", "no"}}
    nested = {
        p for p in paths
        if len(p.parts) == 3 and p.parts[0] == "brain_tumor_dataset"
        and p.parts[1] in {"yes", "no"}
    }
    if paths != direct | nested or not paths:
        raise ValueError(f"Expected folders yes and no inside {root}.")
    if direct and nested:
        aliases = {p.relative_to("brain_tumor_dataset"): p for p in nested}
        if direct != aliases.keys():
            raise ValueError("The two navoneel copies have different filenames.")
        for path, alias in aliases.items():
            if (root / path).read_bytes() != (root / alias).read_bytes():
                raise ValueError(f"The two navoneel copies of {path} differ.")
    canonical = direct or nested
    if {p.parent.name for p in canonical} != {"yes", "no"}:
        raise ValueError("Expected both yes and no labels in navoneel.")
    samples = [Sample(p, "navoneel", p.parent.name) for p in sorted(canonical)]
    return samples, len(paths) - len(canonical)


def indk214_samples(paths: set[PurePosixPath]) -> list[Sample]:
    classification = []
    segmentation = set()
    mask_paths = set()
    splits = {"Training": "train", "Testing": "test"}
    for path in sorted(paths):
        parts = path.parts
        if len(parts) == 5 and parts[:2] == ("DATASET", "classification") and parts[2] in splits:
            classification.append(
                Sample(path, "indk214_classification", INDK_CLASSES[parts[3].lower()], splits[parts[2]])
            )
        elif len(parts) == 4 and parts[:2] == ("DATASET", "Segmentation"):
            if path.stem.endswith("_mask"):
                mask_paths.add(path)
            else:
                segmentation.add(path)
        else:
            raise ValueError(f"Unexpected image layout in the indk214 folder: {path}")

    if {sample.split for sample in classification} != set(splits.values()) or not segmentation:
        raise ValueError("Expected both classification splits and segmentation images in the indk214 folder.")
    pairs = []
    for image in sorted(segmentation):
        mask = image.with_name(f"{image.stem}_mask.png")
        if mask not in mask_paths:
            raise ValueError(f"Missing ground-truth mask for {image}")
        mask_paths.remove(mask)
        pairs.append(Sample(image, "indk214_segmentation", INDK_CLASSES[image.parent.name.lower()], mask=mask))
    if mask_paths:
        raise ValueError(f"Mask without its MRI: {min(mask_paths)}")
    return classification + pairs


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image


def export_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as file:
        image.save(file, format="PNG", compress_level=1)


def process_sample(
    root: Path, sample: Sample, output: Path,
) -> tuple[Measurement, dict | None]:
    image = read_image(root / sample.image)
    width, height = image.size
    gray = grayscale(image)
    box, status = head_box(gray)
    annotation = None
    pixels = None
    bbox_size = None
    dataset = "navoneel" if sample.group == "navoneel" else "indk214"
    image_out = PurePosixPath("cropped", dataset) / sample.image
    if image_out.suffix != ".png":
        image_out = image_out.with_name(image_out.name + ".png")

    if sample.mask is not None:
        mask = read_image(root / sample.mask)
        if mask.size != image.size:
            raise ValueError(f"Different MRI and mask sizes for {sample.image}")
        foreground = grayscale(mask) >= MASK_THRESHOLD
        tumor_box = mask_box(foreground)
        if tumor_box is not None and not contains(box, tumor_box):
            box, status = (0, 0, width, height), "mask_guard_full_frame"
        x1, y1, x2, y2 = box
        foreground_crop = foreground[y1:y2, x1:x2]
        pixels = int(np.count_nonzero(foreground))
        if np.count_nonzero(foreground_crop) != pixels:
            raise ValueError(f"Crop lost mask pixels for {sample.image}")

        tumor_crop_box = None
        if tumor_box is not None:
            a1, b1, a2, b2 = tumor_box
            bbox_size = (a2 - a1, b2 - b1)
            tumor_crop_box = (a1 - x1, b1 - y1, a2 - x1, b2 - y1)
        mask_out = PurePosixPath("cropped", dataset) / sample.mask.with_suffix(".png")
        export_png(Image.fromarray(foreground_crop.astype(np.uint8) * 255), output / mask_out)
        annotation = {
            "label": sample.label,
            "source_image": sample.image.as_posix(),
            "source_mask": sample.mask.as_posix(),
            "cropped_image": image_out.as_posix(),
            "cropped_mask": mask_out.as_posix(),
            "original_size_wh": (width, height),
            "cropped_size_wh": (x2 - x1, y2 - y1),
            "skull_bbox_xyxy": box,
            "mask_bbox_xyxy": tumor_box,
            "mask_bbox_crop_xyxy": tumor_crop_box,
            "mask_area_pixels": pixels,
            "crop_status": status,
        }

    export_png(image.crop(box), output / image_out)
    x1, y1, x2, y2 = box
    measurement = Measurement(
        label=sample.label,
        split=sample.split,
        width=width,
        height=height,
        mode=image.mode,
        extension=sample.image.suffix.lower(),
        gray_digest=(width, height, hashlib.sha256(gray.tobytes()).digest()),
        crop_status=status,
        crop_removed_percent=100 * (1 - (x2 - x1) * (y2 - y1) / (width * height)),
        mask_pixels=pixels,
        mask_bbox_size=bbox_size,
    )
    return measurement, annotation


def distribution(values: list[int | float]) -> dict:
    if not values:
        return {}
    points = np.percentile(values, [0, 25, 50, 75, 100])
    return dict(zip(("min", "p25", "median", "p75", "max"), (round(float(v), 3) for v in points)))


def summarize(rows: list[Measurement]) -> dict:
    class_counts = Counter(row.label for row in rows)
    split_counts = Counter(row.split for row in rows)
    sizes = Counter((row.width, row.height) for row in rows)
    digests = Counter(row.gray_digest for row in rows)
    labels_by_digest = defaultdict(set)
    for row in rows:
        labels_by_digest[row.gray_digest].add(row.label)

    summary = {
        "images": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "class_percent": {name: round(100 * count / len(rows), 3) for name, count in sorted(class_counts.items())},
        "split_counts": dict(sorted(split_counts.items())),
        "split_class_counts": {
            split: dict(sorted(Counter(row.label for row in rows if row.split == split).items()))
            for split in sorted(split_counts)
        },
        "width_px": distribution([row.width for row in rows]),
        "height_px": distribution([row.height for row in rows]),
        "most_common_sizes_wh": [{"size": size, "images": count} for size, count in sizes.most_common(5)],
        "modes": dict(sorted(Counter(row.mode for row in rows).items())),
        "extensions": dict(sorted(Counter(row.extension for row in rows).items())),
        "crop_status_counts": dict(sorted(Counter(row.crop_status for row in rows).items())),
        "image_area_removed_percent": distribution([row.crop_removed_percent for row in rows]),
        "exact_grayscale_duplicates": {
            "distinct_images": len(digests),
            "duplicate_groups": sum(count > 1 for count in digests.values()),
            "repeated_images_beyond_first": len(rows) - len(digests),
            "conflicting_label_groups": sum(len(labels) > 1 for labels in labels_by_digest.values()),
        },
    }
    if "train" in split_counts and "test" in split_counts:
        train = [row.gray_digest for row in rows if row.split == "train"]
        test = [row.gray_digest for row in rows if row.split == "test"]
        shared = set(train) & set(test)
        summary["exact_train_test_overlap"] = {
            "distinct_shared_images": len(shared),
            "train_images_in_shared_groups": sum(digest in shared for digest in train),
            "test_images_in_shared_groups": sum(digest in shared for digest in test),
        }
    if rows[0].mask_pixels is not None:
        summary["masks"] = len(rows)
        summary["empty_masks"] = sum(row.mask_pixels == 0 for row in rows)
        summary["mask_by_class"] = {}
        for label in sorted(class_counts):
            subset = [row for row in rows if row.label == label]
            positive = [row for row in subset if row.mask_pixels]
            summary["mask_by_class"][label] = {
                "masks": len(subset),
                "nonempty_masks": len(positive),
                "foreground_pixels": distribution([row.mask_pixels for row in positive]),
                "foreground_percent": distribution([
                    100 * row.mask_pixels / (row.width * row.height) for row in positive
                ]),
                "bbox_width_px": distribution([row.mask_bbox_size[0] for row in positive]),
                "bbox_height_px": distribution([row.mask_bbox_size[1] for row in positive]),
            }
    return summary


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(navoneel: Path, indk214: Path, output: Path) -> None:
    cv2.setNumThreads(1)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Use --output output_lan2.")
    if indk214.name.lower() == "dataset" and (indk214 / "Segmentation").is_dir():
        indk214 = indk214.parent

    nav_paths = image_members(navoneel)
    indk_paths = image_members(indk214)
    nav, aliases = navoneel_samples(navoneel, nav_paths)
    indk = indk214_samples(indk_paths)
    actual_counts = Counter(sample.group for sample in nav + indk)
    if actual_counts != EXPECTED_IMAGES:
        raise ValueError(f"Check extraction. Found {dict(actual_counts)}, expected {EXPECTED_IMAGES}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        measurements = defaultdict(list)
        annotations = []
        for root, samples in ((navoneel, nav), (indk214, indk)):
            for index, sample in enumerate(samples, 1):
                row, annotation = process_sample(root, sample, staged)
                measurements[sample.group].append(row)
                if annotation is not None:
                    annotations.append(annotation)
                if index % 500 == 0 or index == len(samples):
                    print(f"{root.name}: {index}/{len(samples)} MRI images exported", flush=True)

        write_json(staged / "ds_annotations.json", {
            "schema_version": 1,
            "dataset": "indk214",
            "coordinate_format": "xyxy, zero-based, x_max/y_max exclusive",
            "source_paths_relative_to": "extracted indk214 folder containing DATASET",
            "cropped_paths_relative_to": "pipeline output directory",
            "mask_foreground_min_intensity": MASK_THRESHOLD,
            "annotation_count": len(annotations),
            "annotations": annotations,
        })
        write_json(staged / "dataset_statistics.json", {
            "navoneel_image_files_including_aliases": len(nav_paths),
            "navoneel_verified_archive_aliases_excluded": aliases,
            "indk214_image_files_including_masks": len(indk_paths),
            "total_mri_images_processed": len(nav) + len(indk),
            "total_segmentation_pairs": len(annotations),
            "groups": {group: summarize(rows) for group, rows in measurements.items()},
        })
        staged.rename(output)

    print(f"Done: {len(nav) + len(indk)} MRI images, {len(annotations)} masks")
    print(f"Output: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop the extracted navoneel and indk214 MRI datasets.")
    parser.add_argument("--navoneel", type=Path, default=NAVONEEL_DIR)
    parser.add_argument("--indk214", type=Path, default=INDK214_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run(args.navoneel, args.indk214, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()