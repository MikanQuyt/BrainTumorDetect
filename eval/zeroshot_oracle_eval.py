"""
zeroshot_oracle_eval.py - Tuan 2: Zero-shot SAM goc va zero-shot MedSAM voi hop chuan (oracle GT box).

Doc file JSON annotation cua nhom (dataset/brats2020_annotations.json, dataset/ds2_annotations.json, ...),
chon tap (mac dinh split == "val") va cac mau co gt_box, chay:
    - SAM ViT-B goc        (SamPredictor, multimask_output=False)
    - MedSAM ViT-B         (quy trinh suy luan chinh thuc: resize 1024, min-max [0,1], sigmoid > 0.5)
    - "box-fill"           (to kin ca hop chuan lam mask) - moc tham chieu re tien
roi ghi CSV dang rong: moi dong la mot anh, kem Dice/IoU cua tung phuong phap va cac mo ta bien u.

Chay thu (khong can GPU/checkpoint) de kiem tra pipeline:
    python eval/zeroshot_oracle_eval.py --ann dataset/brats2020_annotations.json --root . --dry_run --limit 20 --out_dir reports/week2/_dry

Xem cau truc JSON truoc khi chay:
    python eval/zeroshot_oracle_eval.py --ann dataset/ds2_annotations.json --root . --inspect

Chay that:
    python eval/zeroshot_oracle_eval.py --ann dataset/brats2020_annotations.json --root . --split val ^
        --sam_ckpt checkpoints/sam_vit_b_01ec64.pth --medsam_ckpt checkpoints/medsam_vit_b.pth ^
        --tag ds4 --out_dir reports/week2/ds4

Co the ngat va chay tiep bang --resume.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------------ doc anh
def imread(path):
    buf = np.fromfile(str(path), dtype=np.uint8)      # doc duoc duong dan co dau cach/ky tu dac biet tren Windows
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return img


def load_rgb(path):
    img = imread(path)
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = img[..., :3][..., ::-1]                  # BGR(A) -> RGB
    return np.ascontiguousarray(img)


def load_mask(path):
    m = imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def resolve(p, root):
    p = Path(p)
    return p if p.is_absolute() else Path(root) / p


# ------------------------------------------------------------------ hop chuan
def tight_box(mask):
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)   # nua mo


def box_convention(gt, mask):
    """'excl': [x0,y0,x1+1,y1+1] | 'incl': [x0,y0,x1,y1] | 'other': hop khong sat mask."""
    t = tight_box(mask)
    if np.allclose(gt, t):
        return "excl"
    if np.allclose(gt, t - np.array([0, 0, 1, 1])):
        return "incl"
    return "other"


def box_fill(box, shape):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


# ------------------------------------------------------------------ so do
def dice_iou(pred, gt, eps=1e-8):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = float((p & g).sum())
    return 2.0 * inter / (p.sum() + g.sum() + eps), inter / ((p | g).sum() + eps)


def boundary_descriptors(img_rgb, mask):
    """Do ro net cua bien u tren kenh dau (FLAIR voi BraTS): gradient Sobel trung binh trong dai quanh bien
    (da chuan hoa theo khoang cuong do) va do tuong phan trong/ngoai. Gia tri nho = bien mo."""
    g = img_rgb[..., 0].astype(np.float32)
    nz = g[g > 0]
    rng = float(np.percentile(nz, 99) - np.percentile(nz, 1)) if nz.size > 50 else float(g.max() - g.min())
    rng = max(rng, 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2) / rng
    m8 = mask.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    band = cv2.dilate(m8, k, iterations=2).astype(bool) & ~cv2.erode(m8, k, iterations=2).astype(bool)
    sharp = float(grad[band].mean()) if band.any() else np.nan
    ring = cv2.dilate(m8, k, iterations=6).astype(bool) & ~cv2.dilate(m8, k, iterations=2).astype(bool)
    inside = cv2.erode(m8, k, iterations=2).astype(bool)
    if not inside.any():
        inside = mask
    contrast = float(abs(g[inside].mean() - g[ring].mean()) / rng) if ring.any() else np.nan
    return sharp, contrast


# ------------------------------------------------------------------ mo hinh
class SamRunner:
    name = "sam"

    def __init__(self, ckpt, device):
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=str(ckpt)).to(device).eval()
        self.pred = SamPredictor(sam)

    def __call__(self, img_rgb, box):
        self.pred.set_image(img_rgb)                    # RGB uint8 HxWx3
        masks, _, _ = self.pred.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
        return masks[0].astype(bool)


class MedSamRunner:
    name = "medsam"

    def __init__(self, ckpt, device):
        import torch
        from segment_anything import sam_model_registry
        self.torch, self.device = torch, device
        model = sam_model_registry["vit_b"](checkpoint=None)
        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "model" in state and not any(k.startswith("image_encoder") for k in state):
            state = state["model"]
        model.load_state_dict(state)
        self.model = model.to(device).eval()

    def __call__(self, img_rgb, box):
        torch = self.torch
        H, W = img_rgb.shape[:2]
        with torch.no_grad():
            x = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
            emb = self.model.image_encoder(x)
            b = np.asarray([box], dtype=np.float32) / np.array([W, H, W, H], dtype=np.float32) * 1024.0
            b = torch.from_numpy(b).float().to(self.device)[:, None, :]
            sparse, dense = self.model.prompt_encoder(points=None, boxes=b, masks=None)
            low, _ = self.model.mask_decoder(image_embeddings=emb,
                                             image_pe=self.model.prompt_encoder.get_dense_pe(),
                                             sparse_prompt_embeddings=sparse,
                                             dense_prompt_embeddings=dense, multimask_output=False)
            prob = torch.nn.functional.interpolate(torch.sigmoid(low), size=(H, W),
                                                   mode="bilinear", align_corners=False)
        return prob.squeeze().cpu().numpy() > 0.5


class DryRunner:
    """Chi de kiem tra pipeline (KHONG phai ket qua that): to kin hop, co the co xoi mon."""

    def __init__(self, name, erode):
        self.name, self.erode = name, erode

    def __call__(self, img_rgb, box):
        m = box_fill(box, img_rgb.shape[:2])
        if self.erode:
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=self.erode).astype(bool)
        return m


# ------------------------------------------------------------------ tien ich
def file_md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pkg_version(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


def case_key(r):
    for k in ("case_id", "patient_id"):
        if r.get(k):
            return str(r[k])
    if r.get("volume") is not None:
        return f"volume_{r['volume']}"
    return Path(str(r.get("img_path", ""))).stem


def inspect_json(records, root):
    print(f"Tong so ban ghi: {len(records)}")
    print("Cac truong cua ban ghi dau tien:", list(records[0].keys()))
    print("Ban ghi dau tien:", records[0])
    with_box = [r for r in records if r.get("gt_box") is not None]
    print(f"Ban ghi co gt_box: {len(with_box)} / {len(records)}")
    print("Dem (split, fold) trong cac ban ghi co gt_box:")
    for k, v in sorted(Counter((r.get("split"), r.get("fold")) for r in with_box).items(), key=str):
        print(f"   split={k[0]!r:16} fold={k[1]!r:6} -> {v}")
    r0 = with_box[0] if with_box else records[0]
    for key in ("img_path", "mask_path"):
        if r0.get(key):
            p = resolve(r0[key], root)
            print(f"{key}: {r0[key]}  ->  {'TON TAI' if p.exists() else 'KHONG TON TAI'} ({p})")
    if with_box and r0.get("mask_path") and resolve(r0["mask_path"], root).exists():
        m = load_mask(resolve(r0["mask_path"], root))
        if m.any():
            print("Quy uoc hop cua ban ghi dau:", box_convention(np.array(r0["gt_box"], dtype=np.float32), m))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Zero-shot SAM / MedSAM voi oracle GT box")
    ap.add_argument("--ann", required=True, help="File JSON annotation (list cac ban ghi)")
    ap.add_argument("--root", default=".", help="Thu muc goc de giai duong dan tuong doi trong JSON")
    ap.add_argument("--split", default="val", help="Tap can danh gia (mac dinh val); dat '' de bo loc")
    ap.add_argument("--fold", type=int, default=None, help="Loc them theo fold (neu JSON dung fold)")
    ap.add_argument("--tag", default=None, help="Ten dataset trong CSV (mac dinh: ten file JSON)")
    ap.add_argument("--sam_ckpt", default=None)
    ap.add_argument("--medsam_ckpt", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (mac dinh: tu chon)")
    ap.add_argument("--out_dir", default="reports/week2/run")
    ap.add_argument("--limit", type=int, default=None, help="Chi chay N anh (lay ngau nhien, co seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_template", default=None,
                    help="Mau duong dan anh, vi du C:/.../images/BraTS20_Training_{volume:03d}_z{slice:03d}.png "
                         "(dung cac truong cua ban ghi JSON); thay cho img_path trong JSON")
    ap.add_argument("--mask_template", default=None, help="Giong --img_template nhung cho mask")
    ap.add_argument("--transpose", action="store_true",
                    help="Hoan doi hai truc cua anh va mask (de khop he toa do hop chuan cua JSON)")
    ap.add_argument("--swap_box_xy", action="store_true",
                    help="Hoan doi x va y cua gt_box")
    ap.add_argument("--box_end", choices=["auto", "as_is"], default="auto",
                    help="auto: neu hop theo quy uoc bao gom diem cuoi thi cong them 1 pixel")
    ap.add_argument("--no_save_masks", action="store_true", help="Khong luu mask du doan")
    ap.add_argument("--resume", action="store_true", help="Bo qua cac anh da co trong CSV")
    ap.add_argument("--dry_run", action="store_true", help="Kiem tra pipeline voi mo hinh gia (khong phai ket qua that)")
    ap.add_argument("--inspect", action="store_true", help="In cau truc JSON roi thoat")
    args = ap.parse_args()

    records = json.load(open(args.ann, encoding="utf-8"))
    if args.inspect:
        inspect_json(records, args.root)
        return

    tag = args.tag or Path(args.ann).stem.replace("_annotations", "")
    recs = []
    for i, r in enumerate(records):
        if r.get("gt_box") is None:
            continue
        if args.split and r.get("split") != args.split:
            continue
        if args.fold is not None and r.get("fold") != args.fold:
            continue
        r = dict(r)
        r["_idx"] = i
        recs.append(r)
    if not recs:
        sys.exit("Khong co ban ghi nao thoa dieu kien (split/fold/gt_box). Chay lai voi --inspect de xem JSON.")
    if args.limit and args.limit < len(recs):
        pick = np.sort(np.random.RandomState(args.seed).permutation(len(recs))[: args.limit])
        recs = [recs[i] for i in pick]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "oracle_results.csv"
    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["idx"].tolist())
        print(f"Resume: bo qua {len(done)} anh da xong.")
    elif csv_path.exists():
        csv_path.unlink()

    # ---- mo hinh
    if args.dry_run:
        runners = [DryRunner("sam", 0), DryRunner("medsam", 2)]
        device = "dry_run"
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not args.sam_ckpt or not args.medsam_ckpt:
            sys.exit("Thieu --sam_ckpt hoac --medsam_ckpt.")
        print(f"Thiet bi: {device}")
        runners = [SamRunner(args.sam_ckpt, device), MedSamRunner(args.medsam_ckpt, device)]

    info = dict(tag=tag, ann=str(args.ann), split=args.split, fold=args.fold, n_images=len(recs),
                device=device, dry_run=bool(args.dry_run), seed=args.seed, box_end=args.box_end,
                transpose=bool(args.transpose), img_template=args.img_template, mask_template=args.mask_template,
                multimask_output=False, medsam_threshold=0.5, model_type="vit_b",
                torch=pkg_version("torch"), segment_anything=pkg_version("segment-anything"),
                sam_ckpt=None, medsam_ckpt=None, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not args.dry_run:
        for k in ("sam_ckpt", "medsam_ckpt"):
            p = getattr(args, k)
            info[k] = dict(file=os.path.basename(p), bytes=os.path.getsize(p), md5=file_md5(p))
    mask_dirs = {}
    if not args.no_save_masks:
        for rn in runners:
            mask_dirs[rn.name] = out_dir / "pred" / rn.name
            mask_dirs[rn.name].mkdir(parents=True, exist_ok=True)

    # ---- chay
    rows, conv_counter, errors = [], Counter(), []
    header_written = csv_path.exists()

    def flush():
        nonlocal rows, header_written
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not header_written, index=False)
            header_written, rows = True, []

    for r in tqdm(recs, desc=f"Oracle {tag}"):
        idx = r["_idx"]
        if idx in done:
            continue
        try:
            img_p = Path(args.img_template.format(**r)) if args.img_template else resolve(r["img_path"], args.root)
            msk_p = Path(args.mask_template.format(**r)) if args.mask_template else resolve(r["mask_path"], args.root)
            img, mask = load_rgb(img_p), load_mask(msk_p)
            if args.transpose:
                img = np.ascontiguousarray(img.transpose(1, 0, 2))
                mask = np.ascontiguousarray(mask.T)
            if img.shape[:2] != mask.shape:
                raise ValueError(f"kich thuoc anh {img.shape[:2]} != mask {mask.shape}")
            if not mask.any():
                raise ValueError("mask rong")
            gt_box = np.asarray(r["gt_box"], dtype=np.float32)
            if args.swap_box_xy:
                gt_box = gt_box[[1, 0, 3, 2]]
            conv = box_convention(gt_box, mask)
            conv_counter[conv] += 1
            box = gt_box.copy()
            if args.box_end == "auto" and conv == "incl":
                box[2:] += 1.0
            H, W = mask.shape
            sharp, contrast = boundary_descriptors(img, mask)
            area, barea = int(mask.sum()), float(max(box[2] - box[0], 1) * max(box[3] - box[1], 1))
            row = dict(idx=idx, dataset=tag, group=case_key(r), case_id=r.get("case_id"), slice=r.get("slice"),
                       split=r.get("split"), fold=r.get("fold"), img_abs=str(img_p), mask_abs=str(msk_p),
                       H=H, W=W, tumor_area=area, box_area=barea, tightness=area / barea,
                       box_conv=conv, edge_sharpness=sharp, contrast=contrast)
            fill = box_fill(box, mask.shape)
            row["dice_boxfill"], row["iou_boxfill"] = dice_iou(fill, mask)
            for rn in runners:
                pred = rn(img, box)
                row[f"dice_{rn.name}"], row[f"iou_{rn.name}"] = dice_iou(pred, mask)
                row[f"empty_{rn.name}"] = int(pred.sum() == 0)
                if rn.name in mask_dirs:
                    cv2.imencode(".png", (pred.astype(np.uint8) * 255))[1].tofile(
                        str(mask_dirs[rn.name] / f"{idx:06d}.png"))
            rows.append(row)
        except Exception as e:
            errors.append((idx, repr(e)))
        if len(rows) >= 100:
            flush()
    flush()

    prev_path = out_dir / "run_info.json"
    if args.resume and prev_path.exists():
        prev = json.load(open(prev_path, encoding="utf-8"))
        for k, v in prev.get("box_conventions", {}).items():
            conv_counter[k] += v
        info["started"] = prev.get("started", info["started"])
    info.update(box_conventions=dict(conv_counter), n_errors=len(errors),
                errors=[f"{i}: {e}" for i, e in errors[:20]], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(info, open(out_dir / "run_info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nDa luu {len(df)} dong vao {csv_path}")
        for rn in runners:
            print(f"  {rn.name:7s} Dice TB = {df[f'dice_{rn.name}'].mean():.4f} | IoU TB = {df[f'iou_{rn.name}'].mean():.4f}")
        print(f"  boxfill Dice TB = {df['dice_boxfill'].mean():.4f} | IoU TB = {df['iou_boxfill'].mean():.4f}")
    print(f"Quy uoc hop gt_box: {dict(conv_counter)}")
    if errors:
        print(f"[Canh bao] {len(errors)} anh loi (vi du: {errors[:3]}); xem run_info.json")
    if args.dry_run:
        print("\n[LUU Y] Day la DRY RUN voi mo hinh gia - khong dung cac so nay lam ket qua.")


if __name__ == "__main__":
    main()
