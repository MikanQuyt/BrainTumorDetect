"""Prepare Huy's DS1/DS2 data without replacing the team's annotations.

Run: python -m data.prepare_huy_week3
Raw data: dataset/navoneel and dataset/indk214 (the extracted Kaggle folders).
Outputs: dataset/huy_week3/{ds1,ds2}_annotations.json, splits.json, audit.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from data.ds12_index import ds2_segmentation_items
from data.skull_crop import (apply_clahe, binarize_mask, crop_with_mask,
                             mask_to_box, read_image, write_png)
from experiments.vit_data import collect_samples, holdout, pixel_digest


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_digest(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    return hashlib.sha256(f"{w}x{h}:".encode() + gray.tobytes()).hexdigest()


def prepare_record(root, output, dataset, image_path, label, source,
                   baseline_image=None, mask_path=None):
    """Append .png to the WHOLE filename: Y92.jpg.png and Y92.png.png stay distinct."""
    image_path = Path(image_path)
    rel = image_path.relative_to(root / "dataset")
    destination = output / "images" / dataset / rel.parent / (rel.name + ".png")
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Cannot read MRI: {image_path}")
    mask = None
    if mask_path is not None:
        raw_mask = read_image(mask_path, 0)
        if raw_mask is None or raw_mask.shape != image.shape[:2]:
            raise ValueError(f"Missing/mismatched mask: {mask_path}")
        mask = binarize_mask(raw_mask)
        if not mask.any():
            raise ValueError(f"Tumor-positive sample has an empty mask: {mask_path}")
    crop, cropped_mask, crop_box, info = crop_with_mask(image, mask)
    enhanced = apply_clahe(crop)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_png(destination, enhanced)
    mask_out = None
    if mask is not None:
        if int(mask.sum()) != int(cropped_mask.sum()):
            raise ValueError(f"Crop lost tumor pixels: {image_path}")
        mask_out = output / "masks" / dataset / rel.parent / (rel.name + ".png")
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        write_png(mask_out, cropped_mask * 255)
    return {
        "dataset": "navoneel" if dataset == "ds1" else "indk214",
        "source": source,
        "sample_id": image_path.relative_to(root).as_posix(),
        "source_image": image_path.relative_to(root).as_posix(),
        "baseline_image": baseline_image,
        "img_path": destination.relative_to(root).as_posix(),
        "mask_path": mask_out.relative_to(root).as_posix() if mask_out else None,
        "has_mask": mask_out is not None,
        "has_tumor": int(label != 0),
        "sub_class_label": int(label),
        "gt_box": mask_to_box(cropped_mask) if mask is not None else None,
        "gt_box_orig": mask_to_box(mask) if mask is not None else None,
        "crop_box": list(crop_box),
        "mask_area_pixels": int(mask.sum()) if mask is not None else None,
        "raw_pixel_digest": pixel_digest(image_path),
        "crop_pixel_digest": image_digest(enhanced),
        "crop_info": info,
        "split": None,
        "fold": None,
    }


def assign_duplicate_groups(records):
    """Union exact raw OR preprocessed grayscale duplicates, including across DS2 sources."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for field in ("raw_pixel_digest", "crop_pixel_digest"):
            key = (field, record[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i in range(len(records)):
        members[find(i)].append(i)
    for indices in members.values():
        labels = {records[i]["sub_class_label"] for i in indices}
        if len(labels) != 1:
            raise ValueError(f"Exact duplicates have conflicting labels: {indices}")
        group = min(records[i]["raw_pixel_digest"] for i in indices)
        for i in indices:
            records[i]["duplicate_group"] = group


def validate_split(records, split):
    parts = [list(split[k]) for k in ("train", "val", "test")]
    flat = [i for part in parts for i in part]
    if sorted(flat) != list(range(len(records))):
        raise ValueError("Split is not a complete, disjoint partition of the dataset")
    groups = [{records[i]["duplicate_group"] for i in part} for part in parts]
    if sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError("Exact pixel duplicates cross train/val/test; do not train")
    if any(not part for part in parts):
        raise ValueError("Empty train/val/test partition")


def repair_group_split(records, split):
    """Keep merged duplicate groups intact, preserving every original held-out test image.

    Test takes precedence over validation, which takes precedence over training.
    This rule is fixed before model training and never uses model predictions.
    """
    priority = {"train": 0, "val": 1, "test": 2}
    flat = [i for part in priority for i in split[part]]
    original = {i: part for part in priority for i in split[part]}
    if sorted(flat) != list(range(len(records))):
        raise ValueError("Cannot repair an incomplete classification split")
    destinations = defaultdict(set)
    for i, record in enumerate(records):
        destinations[record["duplicate_group"]].add(original[i])
    target = {group: max(parts, key=priority.get) for group, parts in destinations.items()}
    repaired = {"fold": split["fold"], "train": [], "val": [], "test": []}
    moves = []
    for i, record in enumerate(records):
        before, after = original[i], target[record["duplicate_group"]]
        record["week2_split"] = before
        repaired[after].append(i)
        if before != after:
            moves.append({"sample_id": record["sample_id"], "duplicate_group": record["duplicate_group"],
                          "sub_class_label": record["sub_class_label"],
                          "week2_split": before, "week3_split": after})
    validate_split(records, repaired)
    if not set(split["test"]).issubset(repaired["test"]):
        raise ValueError("Original test images must stay in the test partition")
    if any(priority[original[i]] > priority[part] for part in priority for i in repaired[part]):
        raise ValueError("Held-out images must never move toward training")
    summary = {"rule": "test > val > train; retain all images; no held-out image moves to training",
               "n_conflicting_groups": sum(len(parts) > 1 for parts in destinations.values()),
               "n_images_moved": len(moves),
               "counts_week2": {part: len(split[part]) for part in priority},
               "counts_week3": {part: len(repaired[part]) for part in priority},
               "all_original_test_images_preserved": True,
               "moves": moves}
    return repaired, summary


def baseline_splits(records, path, *, repair_merged_groups=False, repair_log=None):
    """Verify Week 2 raw hashes/splits, then validate or repair the expanded groups."""
    names = {r["baseline_image"]: i for i, r in enumerate(records)}
    if len(names) != len(records):
        raise ValueError("Repeated source filename")
    plans = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["image"] not in names:
                raise ValueError(f"Week 2 source image is missing: {row['image']}")
            i = names[row["image"]]
            if records[i]["raw_pixel_digest"] != row["duplicate_group"]:
                raise ValueError(f"Data differs from Week 2: {row['image']}")
            fold = int(row["fold"])
            plan = plans.setdefault(fold, {"fold": fold, "train": [], "val": [], "test": []})
            plan[row["partition"]].append(i)
    result = [plans[f] for f in sorted(plans)]
    checked = []
    raw_records = [dict(record, duplicate_group=record["raw_pixel_digest"]) for record in records]
    for split in result:
        # The historical protocol grouped raw pixels; verify it independently first.
        validate_split(raw_records, split)
        if repair_merged_groups:
            split, changes = repair_group_split(records, split)
            if repair_log is not None:
                repair_log.append(changes)
        else:
            validate_split(records, split)
        checked.append(split)
    return checked


def extend_ds2_split(records, classification_split, n_classification, seed=42):
    """Anchor segmentation to the verified/repaired classification groups."""
    assignment = {}
    for part in ("train", "val", "test"):
        for i in classification_split[part]:
            group = records[i]["duplicate_group"]
            if group in assignment and assignment[group] != part:
                raise ValueError("A cross-source duplicate joins incompatible Week 2 splits")
            assignment[group] = part
    remaining = np.array([i for i in range(n_classification, len(records))
                          if records[i]["duplicate_group"] not in assignment], dtype=int)
    if len(remaining):
        labels = np.array([r["sub_class_label"] for r in records])
        groups = np.array([r["duplicate_group"] for r in records])
        train, held = holdout(remaining, labels, groups, 0.30, seed)
        val, test = holdout(held, labels, groups, 0.50, seed + 1)
        for part, indices in (("train", train), ("val", val), ("test", test)):
            for i in indices:
                assignment[records[i]["duplicate_group"]] = part
    plan = {"fold": 1, "train": [], "val": [], "test": []}
    for i, record in enumerate(records):
        part = assignment[record["duplicate_group"]]
        record["split"] = part
        plan[part].append(i)
    validate_split(records, plan)
    return plan


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root)  # UnifiedBrainMRIDataset resolves paths relative to the repo.
    plans, audit = {}, {
        "split_policy": "DS1: Week 2 folds; DS2: Week 2 classification split with merged-group repair (test > val > train), segmentation anchored to repaired groups",
        "duplicate_scope": "exact grayscale pixels + dimensions, before and after crop/CLAHE; not patient-level or near-duplicate verification",
        "week2_input_size": 112,
        "week3_input_size": 224,
    }
    for dataset in ("ds1", "ds2"):
        name = "navoneel" if dataset == "ds1" else "indk214"
        raw_root = root / "dataset" / name
        samples, _ = collect_samples(dataset, raw_root)
        records = []
        for j, sample in enumerate(samples):
            label = (-1 if sample.label else 0) if dataset == "ds1" else sample.label
            baseline_image = (f"{sample.path.parent.name}/{sample.path.name}" if dataset == "ds1"
                              else sample.path.relative_to(raw_root).as_posix())
            records.append(prepare_record(root, output, dataset, sample.path, label,
                                          "classification", baseline_image))
            if (j + 1) % 500 == 0:
                print(f"{dataset} classification {j + 1}/{len(samples)}", flush=True)
        n_cls = len(records)
        if dataset == "ds2":
            items, missing = ds2_segmentation_items(root)
            if missing or len(items) != 2192:
                raise ValueError(f"Expected 2192 mask pairs; found {len(items)}, missing {len(missing)}")
            for j, item in enumerate(items):
                records.append(prepare_record(root, output, dataset, item["img"], item["label"],
                                              "segmentation", mask_path=item["mask"]))
                if (j + 1) % 500 == 0:
                    print(f"ds2 segmentation {j + 1}/{len(items)}", flush=True)
        assign_duplicate_groups(records)
        repairs = []
        cls_plans = baseline_splits(
            records[:n_cls], root / f"reports/tuan2/vit_baseline/{dataset}/splits.csv",
            repair_merged_groups=(dataset == "ds2"), repair_log=repairs)
        if dataset == "ds1":
            if len(cls_plans) != 5:
                raise ValueError("DS1 must have five outer test folds")
            test_indices = [i for split in cls_plans for i in split["test"]]
            if sorted(test_indices) != list(range(len(records))):
                raise ValueError("Every DS1 image must be tested exactly once")
            for split in cls_plans:
                for i in split["test"]:
                    records[i]["fold"] = split["fold"]
            plans[dataset] = cls_plans
        else:
            plans[dataset] = [extend_ds2_split(records, cls_plans[0], n_cls)]
        if len({r["img_path"] for r in records}) != len(records):
            raise ValueError("Output paths collide")
        save_json(output / f"{dataset}_annotations.json", records)
        group_sources = defaultdict(set)
        for record in records:
            group_sources[record['duplicate_group']].add(record['source'])
        audit[dataset] = {
            "n": len(records), "n_unique_paths": len({r['img_path'] for r in records}),
            "n_exact_groups": len({r['duplicate_group'] for r in records}),
            "n_masks": sum(r["has_mask"] for r in records),
            "cross_source_duplicate_groups": sum(len(sources) > 1 for sources in group_sources.values()),
            "splits": [{"fold": s["fold"], **{
                part: dict(Counter(records[i]["source"] for i in s[part]))
                for part in ("train", "val", "test")}} for s in plans[dataset]],
        }
        if repairs:
            changes = repairs[0]
            audit[dataset]["classification_split_repair"] = {
                k: value for k, value in changes.items() if k != "moves"}
            changes_path = output / "ds2_split_changes.csv"
            with changes_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["sample_id", "duplicate_group",
                    "sub_class_label", "week2_split", "week3_split"])
                writer.writeheader()
                writer.writerows(changes["moves"])
            audit[dataset]["split_changes_file"] = changes_path.relative_to(root).as_posix()
    save_json(output / "splits.json", plans)
    save_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dataset/huy_week3"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    prepare(args.root, output)


if __name__ == "__main__":
    main()
