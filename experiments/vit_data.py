"""Input images and group-aware splits for the DS1/DS2 ViT baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from data.ds12_index import FOLDER_LABEL, list_images
from data.skull_crop import apply_clahe, crop_with_mask, read_image


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    original_split: str


@dataclass(frozen=True)
class Split:
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def collect_samples(dataset: str, root: Path) -> tuple[list[Sample], tuple[str, ...]]:
    samples = []
    if dataset == "ds1":
        if not (root / "yes").is_dir():
            root = root / "brain_tumor_dataset"
        classes = ("no_tumor", "tumor")
        for folder, label in (("no", 0), ("yes", 1)):
            paths = sorted(list_images(root / folder), key=lambda path: path.name)
            samples.extend(Sample(path, label, "") for path in paths)
        expected = 253
    else:
        base = root / "DATASET" / "classification"
        if root.name.lower() == "dataset":
            base = root / "classification"
        classes = ("no_tumor", "glioma", "meningioma", "pituitary")
        for original_split in ("Training", "Testing"):
            split_dir = base / original_split
            if not split_dir.is_dir():
                raise ValueError(f"Missing classification folder: {split_dir}")
            for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                label = FOLDER_LABEL[folder.name.lower()]
                paths = sorted(list_images(folder), key=lambda path: path.name)
                samples.extend(Sample(path, label, original_split) for path in paths)
        expected = 7023
    if len(samples) != expected:
        raise ValueError(f"{dataset}: found {len(samples)} images under {root}, expected {expected}.")
    return samples, classes


def pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            image = Image.alpha_composite(black, rgba)
        gray = image.convert("L")
        digest = hashlib.sha256(f"{gray.width}x{gray.height}:".encode("ascii"))
        digest.update(gray.tobytes())
        return digest.hexdigest()


def prepare_images(samples: list[Sample], image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2.setNumThreads(1)
    images = np.empty((len(samples), image_size, image_size, 3), dtype=np.uint8)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    groups = []
    group_labels = {}
    for index, sample in enumerate(samples):
        group = pixel_digest(sample.path)
        if group_labels.setdefault(group, sample.label) != sample.label:
            raise ValueError(f"Identical pixels have conflicting labels: {sample.path}")
        groups.append(group)
        image = read_image(sample.path)
        if image is None:
            raise ValueError(f"Could not decode MRI: {sample.path}")
        cropped, _, _, _ = crop_with_mask(image)
        enhanced = apply_clahe(cropped)
        resized = cv2.resize(enhanced, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        images[index] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if (index + 1) % 1000 == 0 or index + 1 == len(samples):
            print(f"Prepared {index + 1}/{len(samples)} MRI images", flush=True)
    return images, labels, np.asarray(groups)


def holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique, first = np.unique(groups[indices], return_index=True)
    _, held_groups = train_test_split(
        unique, test_size=fraction, stratify=labels[indices[first]], random_state=seed,
    )
    held = np.isin(groups[indices], held_groups)
    return indices[~held], indices[held]


def make_splits(dataset: str, labels: np.ndarray, groups: np.ndarray, seed: int) -> list[Split]:
    all_indices = np.arange(len(labels))
    splits = []
    if dataset == "ds1":
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (pool, test) in enumerate(outer.split(all_indices, labels, groups), start=1):
            train, val = holdout(pool, labels, groups, 0.20, seed + fold)
            splits.append(Split(fold, train, val, test))
        if not np.array_equal(np.sort(np.concatenate([split.test for split in splits])), all_indices):
            raise ValueError("Each DS1 image must be evaluated exactly once.")
    else:
        train, held = holdout(all_indices, labels, groups, 0.30, seed)
        val, test = holdout(held, labels, groups, 0.50, seed + 1)
        splits.append(Split(1, train, val, test))

    classes = set(labels)
    for split in splits:
        partitions = (split.train, split.val, split.test)
        group_sets = [set(groups[part]) for part in partitions]
        if not np.array_equal(np.sort(np.concatenate(partitions)), all_indices):
            raise ValueError(f"Images are missing or repeated in fold {split.fold}.")
        if sum(map(len, group_sets)) != len(set.union(*group_sets)):
            raise ValueError(f"Duplicate group overlaps between partitions in fold {split.fold}.")
        if any(set(labels[part]) != classes for part in partitions):
            raise ValueError(f"A class is missing from a partition in fold {split.fold}.")
    return splits
