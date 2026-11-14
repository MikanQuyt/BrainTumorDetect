"""Tiền xử lý Dataset 3 (BRISC 2025): Kiểm định 1-1, CLAHE, trích xuất Bounding Box & chia tập 70/15/15.

Chạy từ thư mục gốc dự án:
    python -m data.build_ds3
Đầu ra:
    - Ảnh đã lọc CLAHE: dataset/ds3_processed/images/<stem>.png
    - Mask nhị phân chuẩn {0, 255}: dataset/ds3_processed/masks/<stem>.png
    - Tệp chú thích chuẩn hóa: dataset/ds3_annotations.json
    - Nhật ký xử lý & chia tập: reports/build_ds3/process_log.csv, reports/build_ds3/split_summary.csv
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tqdm import tqdm

from data.ds3_index import (CLASS_NAME, PLANE_NAME, ds3_classification_items,
                            ds3_segmentation_items)
from data.skull_crop import (apply_clahe, binarize_mask, mask_to_box, read_image,
                             skull_crop_box, write_png)

SEED = 42


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thumb_vec(image, size=32):
    """Vector chuẩn hóa zero-mean unit-norm kích thước 32x32 phục vụ gom nhóm lát cắt cùng bệnh nhân."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm > 1e-6 else small


def assign_patient_groups(records, vecs, adj_corr_thresh=0.90, global_corr_thresh=0.96, max_seq_len=15):
    """Suy luận nhóm bệnh nhân (patient_id) cho BRISC 2025 bằng đồ thị liên thông (Union-Find).

    Vì tên file BRISC 2025 (brisc2025_<split>_<index>_<tumor>_<plane>_t1) không chứa mã bệnh nhân tường minh,
    ta liên kết 2 lát cắt (i, j) vào cùng một bệnh nhân khi thỏa ít nhất 1 trong 3 điều kiện:
      1. Trùng khớp tuyệt đối mã băm nội dung ảnh (sha256[i] == sha256[j]) — bắt toàn bộ 46 cụm ảnh trùng.
      2. Hai lát cắt kề nhau trong cùng chuỗi quét (cùng orig_split, lớp, mặt phẳng cắt, kích thước ảnh,
         |index_i - index_j| == 1) có tương quan cấu trúc 32x32 >= adj_corr_thresh (0.90), giới hạn độ dài
         chuỗi liên tiếp tối đa max_seq_len để tránh bắc cầu qua nhiều bệnh nhân khác nhau.
      3. Hai lát cắt bất kỳ trong cùng (lớp, mặt phẳng cắt, kích thước ảnh) có tương quan cấu trúc
         32x32 >= global_corr_thresh (0.96) — bắt các lát cắt gần trùng bị đặt khác chỉ số hoặc khác tập gốc.
    """
    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # 1. Liên kết tuyệt đối theo SHA256
    by_sha = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha256"]].append(i)
    for idxs in by_sha.values():
        for k in idxs[1:]:
            union(idxs[0], k)

    # 2 & 3. Liên kết trong từng cụm (sub_class_label, plane_code, img_h, img_w)
    buckets = defaultdict(list)
    for i, r in enumerate(records):
        buckets[(r["sub_class_label"], r["plane_code"], r["img_h"], r["img_w"])].append(i)

    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: (records[i]["orig_split"], records[i]["index"]))
        mat = np.stack([vecs[i] for i in idxs], axis=0)
        sim = mat @ mat.T

        # Cặp có tương quan toàn cục cực cao (>= 0.96)
        hi_a, hi_b = np.where(np.triu(sim >= global_corr_thresh, k=1))
        for a, b in zip(hi_a.tolist(), hi_b.tolist()):
            union(idxs[a], idxs[b])

        # Cặp lát cắt kề nhau trong cùng chuỗi quét (|index_i - index_j| == 1)
        run_len = 1
        for pos in range(len(idxs) - 1):
            ia, ib = idxs[pos], idxs[pos + 1]
            ra, rb = records[ia], records[ib]
            if (ra["orig_split"] == rb["orig_split"]
                    and abs(ra["index"] - rb["index"]) == 1
                    and sim[pos, pos + 1] >= adj_corr_thresh
                    and run_len < max_seq_len):
                union(ia, ib)
                run_len += 1
            else:
                run_len = 1

    comp_map = {}
    for i, r in enumerate(records):
        root_idx = find(i)
        if root_idx not in comp_map:
            comp_map[root_idx] = f"brisc_pat_{len(comp_map) + 1:04d}"
        r["patient_id"] = comp_map[root_idx]
    return len(comp_map)


def split_stratified_group_70_15_15(records, seed=SEED):
    """Chia Stratified Group 70% Train / 15% Val / 15% Test theo từng lớp & mặt phẳng cắt,
    đảm bảo 100% các lát cắt cùng patient_id (hoặc cùng sha256) nằm trong cùng 1 tập.
    """
    rng = np.random.default_rng(seed)

    # Gom toàn bộ record theo patient_id (mỗi patient_id thuộc duy nhất 1 sub_class_label)
    pat_groups = defaultdict(list)
    for i, r in enumerate(records):
        pat_groups[r["patient_id"]].append(i)

    # Phân tầng theo lớp u (sub_class_label) để tỷ lệ 70/15/15 chính xác trên từng lớp
    by_class = defaultdict(list)
    for pid, idxs in pat_groups.items():
        cls = records[idxs[0]]["sub_class_label"]
        plane = records[idxs[0]]["plane_code"]
        by_class[cls].append((pid, plane, idxs))

    for cls, groups in sorted(by_class.items()):
        n_cls = sum(len(idxs) for _, _, idxs in groups)
        target_test = int(round(0.15 * n_cls))
        target_val = int(round(0.15 * n_cls))
        target_train = n_cls - target_test - target_val
        targets = {"train": target_train, "val": target_val, "test": target_test}
        counts = {"train": 0, "val": 0, "test": 0}
        plane_counts = {sp: Counter() for sp in ("train", "val", "test")}

        # Xáo trộn ngẫu nhiên các nhóm bệnh nhân rồi xếp các nhóm nhiều lát cắt trước
        perm = rng.permutation(len(groups))
        shuffled = [groups[i] for i in perm]
        shuffled.sort(key=lambda g: len(g[2]), reverse=True)

        for pid, plane, idxs in shuffled:
            k = len(idxs)
            # Ưu tiên tập còn thiếu mẫu so với mục tiêu (kết hợp cân bằng mặt phẳng cắt)
            candidates = [sp for sp in ("test", "val", "train") if counts[sp] + k <= targets[sp]]
            if not candidates:
                candidates = ["train", "val", "test"]
            best_sp = min(
                candidates,
                key=lambda sp: (
                    (counts[sp] + k - targets[sp]) / max(1, targets[sp]),
                    plane_counts[sp][plane] / max(1, targets[sp]),
                ),
            )
            counts[best_sp] += k
            plane_counts[best_sp][plane] += k
            for i in idxs:
                records[i]["split"] = best_sp
                records[i]["fold"] = None


def process_item(item, root, out_img_dir, out_mask_dir, overwrite=False):
    """Kiểm định 1-1, áp dụng CLAHE, lưu mask nhị phân {0, 255} và trích xuất gt_box."""
    log = dict(
        dataset="brisc2025",
        source=item["source"],
        orig_split=item["orig_split"],
        stem=item["stem"],
        img=str(item["img"].relative_to(root)),
        mask=str(item["mask"].relative_to(root)) if item["mask"] else "",
        status="ok",
        img_h="",
        img_w="",
        gt_box="",
    )
    image = read_image(item["img"], cv2.IMREAD_COLOR)
    if image is None:
        log["status"] = "drop: ảnh hỏng không đọc được"
        return None, None, log

    h, w = image.shape[:2]
    log["img_h"], log["img_w"] = h, w

    mask_bin = None
    gt_box = None
    if item["mask"] is not None:
        raw_mask = read_image(item["mask"], cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            log["status"] = "drop: mask hỏng không đọc được"
            return None, None, log
        if raw_mask.shape[:2] != (h, w):
            log["status"] = f"drop: lệch kích thước 1-1 ảnh {(h, w)} != mask {raw_mask.shape[:2]}"
            return None, None, log
        mask_bin = binarize_mask(raw_mask, thresh=127)
        gt_box = mask_to_box(mask_bin)
        if item["has_tumor"] and gt_box is None:
            log["status"] = "drop: nhãn rỗng sai quy cách (mask = 0 pixel)"
            return None, None, log

    out_img = out_img_dir / item["out_name"]
    if overwrite or not out_img.exists():
        img_clahe = apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
        out_img.parent.mkdir(parents=True, exist_ok=True)
        write_png(out_img, img_clahe)

    mask_rel = None
    if mask_bin is not None:
        out_mask = out_mask_dir / item["out_name"]
        if overwrite or not out_mask.exists():
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            write_png(out_mask, mask_bin * 255)
        mask_rel = out_mask.relative_to(root).as_posix()

    cbox, fallback = skull_crop_box(image)
    log["gt_box"] = json.dumps(gt_box) if gt_box else ""

    rec = dict(
        dataset="brisc2025",
        source=item["source"],
        img_path=out_img.relative_to(root).as_posix(),
        mask_path=mask_rel,
        has_mask=mask_rel is not None,
        has_tumor=int(item["has_tumor"]),
        sub_class_label=int(item["label"]),
        plane_code=item["plane_code"],
        gt_box=gt_box,
        skull_box=list(cbox),
        skull_fallback=bool(fallback),
        img_h=h,
        img_w=w,
        orig_split=item["orig_split"],
        index=int(item["index"]),
        sha256=file_sha256(item["img"]),
        patient_id=None,
        split=None,
        fold=None,
    )
    return rec, thumb_vec(image), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N mẫu đầu (chạy thử)")
    ap.add_argument("--overwrite", action="store_true", help="ghi đè lại toàn bộ ảnh PNG đã xuất")
    args = ap.parse_args()
    root = Path(args.root)

    seg_items, missing_mask, orphan_mask, seg_bad = ds3_segmentation_items(root)
    cls_no_tumor, cls_bad = ds3_classification_items(root, only_no_tumor=True)

    logs = []
    for p in missing_mask:
        logs.append(dict(dataset="brisc2025", source="segmentation", orig_split="", stem=p.stem,
                         img=str(p.relative_to(root)), mask="", status="drop: thiếu mask tương ứng",
                         img_h="", img_w="", gt_box=""))
    for p in orphan_mask:
        logs.append(dict(dataset="brisc2025", source="segmentation", orig_split="", stem=p.stem,
                         img="", mask=str(p.relative_to(root)), status="drop: mask mồ côi thiếu ảnh",
                         img_h="", img_w="", gt_box=""))
    for p in seg_bad + cls_bad:
        logs.append(dict(dataset="brisc2025", source="unknown", orig_split="", stem=p.stem,
                         img=str(p.relative_to(root)), mask="", status="drop: tên file sai quy cách",
                         img_h="", img_w="", gt_box=""))

    out_img_dir = root / "dataset/ds3_processed/images"
    out_mask_dir = root / "dataset/ds3_processed/masks"

    groups = [
        ("DS3 segmentation (có mask)", seg_items),
        ("DS3 classification (no_tumor)", cls_no_tumor),
    ]
    records, vecs = [], []
    for name, items in groups:
        items = items[: args.limit] if args.limit else items
        print(f"{name}: {len(items)} mẫu")
        for item in tqdm(items, desc=name):
            rec, vec, log = process_item(item, root, out_img_dir, out_mask_dir, overwrite=args.overwrite)
            logs.append(log)
            if rec is not None:
                records.append(rec)
                vecs.append(vec)

    n_patients = assign_patient_groups(records, vecs)
    split_stratified_group_70_15_15(records, seed=SEED)

    # Xuất dataset/ds3_annotations.json
    (root / "dataset").mkdir(exist_ok=True)
    out_json = root / "dataset/ds3_annotations.json"
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # Xuất nhật ký & bảng tổng kết chia tập
    rep_dir = root / "reports/tuan1/build_ds3"
    rep_dir.mkdir(parents=True, exist_ok=True)
    if logs:
        fields = list(dict.fromkeys(k for row in logs for k in row))
        with open(rep_dir / "process_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(logs)

    split_rows = []
    for sp in ("train", "val", "test"):
        sub = [r for r in records if r["split"] == sp]
        pats = {r["patient_id"] for r in sub}
        cls_cnt = Counter(r["sub_class_label"] for r in sub)
        split_rows.append(dict(
            split=sp,
            n_samples=len(sub),
            pct_samples=round(100.0 * len(sub) / max(1, len(records)), 2),
            n_patients=len(pats),
            n_with_mask=sum(1 for r in sub if r["has_mask"]),
            no_tumor=cls_cnt[0],
            glioma=cls_cnt[1],
            meningioma=cls_cnt[2],
            pituitary=cls_cnt[3],
        ))
    with open(rep_dir / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
        w.writeheader()
        w.writerows(split_rows)

    print(f"\nĐã xuất {len(records)} record ({n_patients} nhóm bệnh nhân) -> {out_json}")
    for row in split_rows:
        print(f"  {row['split']:5s}: {row['n_samples']:4d} mẫu ({row['pct_samples']:5.2f}%), "
              f"{row['n_patients']:4d} bệnh nhân | có mask={row['n_with_mask']:4d} | "
              f"no_tumor={row['no_tumor']}, glioma={row['glioma']}, "
              f"meningioma={row['meningioma']}, pituitary={row['pituitary']}")
    drops = Counter(l["status"] for l in logs if l["status"] != "ok")
    print("Mẫu bị loại bỏ:", dict(drops) if drops else "0 mẫu (tất cả hợp lệ)")


if __name__ == "__main__":
    main()
