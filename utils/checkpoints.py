"""Tải và kiểm tra checksum trọng số SAM nguyên bản và MedSAM (ViT-B).

    python -m utils.checkpoints              # tải cả hai vào checkpoints/ nếu chưa có / sai checksum
    python -m utils.checkpoints --only sam   # chỉ SAM
    python -m utils.checkpoints --verify     # chỉ kiểm tra checksum, không tải

Trọng số nặng ~375 MB mỗi file nên KHÔNG đưa lên Git (checkpoints/*.pth đã nằm trong .gitignore);
kho code lưu URL nguồn + MD5 chuẩn để mọi thành viên tải về cùng một bản.
MD5 chuẩn trùng với giá trị đã ghi trong reports/week2/*/run_info.json của thí nghiệm Zero-shot Tuần 2.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINTS = {
    "sam": {
        "file": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "md5": "01ec64d29a2fca3f0661936605ae66f8",
        "bytes": 375042383,
    },
    "medsam": {
        "file": "medsam_vit_b.pth",
        "url": "https://zenodo.org/records/10689643/files/medsam_vit_b.pth",
        "md5": "3bb6db55bd0c9ca30b61248bca72f8d6",
        "bytes": 375049145,
    },
}


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_path(name, ckpt_dir=None):
    return Path(ckpt_dir or CKPT_DIR) / CHECKPOINTS[name]["file"]


def verify_checkpoint(name, ckpt_dir=None):
    """True nếu file tồn tại và MD5 khớp giá trị chuẩn."""
    path = checkpoint_path(name, ckpt_dir)
    return path.exists() and md5sum(path) == CHECKPOINTS[name]["md5"]


_last_pct = {"v": -1}


def _progress(block_num, block_size, total):
    """In tiến độ mỗi 10% để log gọn."""
    if total <= 0:
        return
    pct = int(min(block_num * block_size, total) * 100 / total) // 10 * 10
    if pct != _last_pct["v"]:
        _last_pct["v"] = pct
        print(f"  {pct}% ({total / 1e6:.0f} MB)", flush=True)


def download_checkpoint(name, ckpt_dir=None, max_retries=8):
    info = CHECKPOINTS[name]
    path = checkpoint_path(name, ckpt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if verify_checkpoint(name, ckpt_dir):
        print(f"[OK] {path.name} đã có và đúng MD5.")
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    expected = info["bytes"]
    print(f"Tải {info['url']}")
    _last_pct["v"] = -1

    for attempt in range(1, max_retries + 1):
        downloaded = tmp.stat().st_size if tmp.exists() else 0
        if downloaded >= expected:
            break
        headers = {"User-Agent": "BrainTumorDetect/1.0"}
        mode = "wb"
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
            print(f"  [Resume lần {attempt}] Tiếp tục từ {downloaded / 1e6:.1f} MB / {expected / 1e6:.0f} MB", flush=True)
        req = urllib.request.Request(info["url"], headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if downloaded > 0 and status == 200:
                    # Server không hỗ trợ Range, ghi đè lại từ đầu
                    downloaded = 0
                    mode = "wb"
                with open(tmp, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        _progress(downloaded, 1, expected)
            if downloaded >= expected:
                break
        except Exception as exc:
            print(f"  [Cảnh báo lần {attempt}/{max_retries}] Gián đoạn mạng: {exc}", flush=True)
            time.sleep(min(2 * attempt, 10))

    got = md5sum(tmp) if tmp.exists() else "<missing>"
    if got != info["md5"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 sai cho {path.name}: nhận {got}, cần {info['md5']}")
    tmp.replace(path)
    print(f"[OK] {path} (MD5 {got})")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(CHECKPOINTS), default=None)
    ap.add_argument("--dir", default=str(CKPT_DIR))
    ap.add_argument("--verify", action="store_true", help="chỉ kiểm tra checksum")
    args = ap.parse_args()
    names = [args.only] if args.only else list(CHECKPOINTS)
    ok = True
    for n in names:
        if args.verify:
            good = verify_checkpoint(n, args.dir)
            print(f"{'[OK] ' if good else '[THIẾU/SAI]'} {checkpoint_path(n, args.dir)}")
            ok &= good
        else:
            download_checkpoint(n, args.dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
