"""Tạo Notebook Google Colab Tuần 4 dành riêng cho Nguyễn Thanh Hải:
- Huấn luyện MedSAM-LoRA (r=8, alpha=16) trên BRISC 2025, đánh giá theo từng nhóm bệnh (Glioma, Meningioma, Pituitary)
- Ablation Study chiến lược prompt: chỉ so Box và Hybrid (Box + Peak Positive Point)

Chạy:
    python scripts/create_week4_hai_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1] + "\n"]


def make_notebook() -> dict:
    cells = []

    def add_md(source: str):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})

    def add_code(source: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(source),
        })

    # ── Cell 0: Title ───────────────────────────────────────────────────────────────────────────
    add_md("""# Tuần 4 — Huấn luyện `MedSAM-LoRA (r = 8)` trên BRISC 2025 & Ablation Prompt (`Box` vs `Hybrid`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MikanQuyt/BrainTumorDetect/blob/Nguyen_Thanh_Hai_tuan4/notebooks/Week4_MedSAM_LoRA_BRISC2025_Hai_Colab.ipynb)

**Thành viên phụ trách:** Nguyễn Thanh Hải (`Nguyen_Thanh_Hai_tuan4`)

### Hai nhiệm vụ thực nghiệm trọng tâm Tuần 4 trên BRISC 2025:
1. **Huấn luyện `MedSAM-LoRA (r = 8, α = 16)` trên `BRISC 2025`** (`3,355` Train / `719` Val / `719` Test lát cắt có mask) với hàm mất mát $L_{\\text{Seg}} = 0.5\\,L_{\\text{Dice}} + 0.5\\,L_{\\text{Focal}}$ và **Box Jittering `±10px`** (hệ `1024×1024`). Đánh giá **Dice Score, IoU, HD95** chi tiết theo **3 nhóm bệnh**:
   - **Glioma** (`n = 210` mẫu Test)
   - **Meningioma** (`n = 245` mẫu Test)
   - **Pituitary** (`n = 264` mẫu Test)
2. **Ablation Study Chiến lược Prompt (`Box` vs `Hybrid`):** So sánh công bằng hai cấu hình được huấn luyện và đánh giá trên cùng tập dữ liệu, cùng siêu tham số và seed:
   - **Chỉ dùng Box (`Box-only`)**
   - **Hybrid Prompt (`Box + Peak Positive Point`)** — điểm cực đại nội u trích xuất tự động hoàn toàn từ ảnh MRI bên trong vùng lõi hộp (**không dùng `gt_mask`**, `0%` rò rỉ dữ liệu).

> 💡 **An toàn khi mất kết nối / hết quota Colab:** Mọi checkpoint (`latest_medsam_lora.pt`, `best_medsam_lora.pt`) và lịch sử (`history.csv`) đều lưu tự động sau mỗi epoch vào Google Drive (`/content/drive/MyDrive/BrainTumorDetect_Week4_Hai`). Nếu ngắt giữa chừng, chỉ cần mở lại Colab (kể cả bằng tài khoản Google khác có Drive đó) và bấm chạy tiếp — `--resume auto` sẽ phục hồi chính xác từ epoch vừa dừng.""")

    # ── Cell 1-2: Mount Drive & Config ──────────────────────────────────────────────────────────
    add_md("""---
## Cell 1: Mount Google Drive & Cấu hình Siêu tham số Tuần 4""")

    add_code("""from pathlib import Path
import os
import sys
import torch

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# ================= CẤU HÌNH TRUNG TÂM =================
REPO_URL = "https://github.com/MikanQuyt/BrainTumorDetect.git"
REPO_BRANCH = "Nguyen_Thanh_Hai_tuan4"
TARGET_DATASETS = ["brisc2025"]

# Kiến trúc MedSAM-LoRA & hàm mất mát Tuần 4
LORA_R = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LOSS_MODE = "focal"          # 0.5 Dice + 0.5 Focal
DICE_WEIGHT = 0.5
PIXEL_WEIGHT = 0.5
BOX_JITTER = 10.0            # Biên độ Box Jittering ±10px trên hệ 1024x1024 khi train

# Lịch huấn luyện
EPOCHS = 15
PATIENCE = 5
LR_LORA = 1e-4
LR_DECODER = 1e-4
WEIGHT_DECAY = 0.01
SEED = 42
NUM_WORKERS = 2 if IN_COLAB else 0

# Tự động tối ưu Batch Size & Grad Accum theo VRAM GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb >= 22:
        BATCH_SIZE, GRAD_ACCUM = 4, 4   # L4 / A100 (>=22GB)
    else:
        BATCH_SIZE, GRAD_ACCUM = 2, 8   # Tesla T4 (16GB)
    print(f"🚀 GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | CUDA {torch.version.cuda}")
else:
    BATCH_SIZE, GRAD_ACCUM = 2, 2
    print("⚠️ Cảnh báo: Chưa bật GPU! Hãy chọn Runtime -> Change runtime type -> T4 GPU trước khi huấn luyện.")

# Tùy chọn chạy thêm
RUN_SMOKE_TEST = False        # Bật True nếu muốn kiểm thử nhanh 16 mẫu / 1 epoch trước khi train full
EVAL_PERTURBED_JITTER = 15.0  # Kịch bản 2 trong Ablation: Box nhiễu ±15px mô phỏng hộp đầu ra Stage 1
EVAL_MAX_SAMPLES = 0          # 0 = đánh giá toàn bộ 719 mẫu tập Test BRISC 2025

# Thư mục dự án & lưu trữ bền vững trên Google Drive
PROJECT_ROOT = Path("/content/BrainTumorDetect") if IN_COLAB else Path.cwd().resolve()
if not IN_COLAB and PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DRIVE_WORKDIR = (
    Path("/content/drive/MyDrive/BrainTumorDetect_Week4_Hai")
    if IN_COLAB
    else PROJECT_ROOT / "runs" / "week4_hai_local"
)
DRIVE_CKPT_DIR = DRIVE_WORKDIR / "pretrained_checkpoints"
DRIVE_DATA_CACHE = DRIVE_WORKDIR / "data_cache"
RUN_BOX_DIR = DRIVE_WORKDIR / "medsam_lora_brisc2025_r8_box"
RUN_HYBRID_DIR = DRIVE_WORKDIR / "medsam_lora_brisc2025_r8_hybrid"
REPORT_OUT_DIR = DRIVE_WORKDIR / "reports_tuan4_hai"

for d in (DRIVE_WORKDIR, DRIVE_CKPT_DIR, DRIVE_DATA_CACHE, RUN_BOX_DIR, RUN_HYBRID_DIR, REPORT_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

print(f"BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM} (effective batch={BATCH_SIZE * GRAD_ACCUM})")
print(f"PROJECT_ROOT   = {PROJECT_ROOT}")
print(f"RUN_BOX_DIR    = {RUN_BOX_DIR}")
print(f"RUN_HYBRID_DIR = {RUN_HYBRID_DIR}")
print(f"REPORT_OUT_DIR = {REPORT_OUT_DIR}")""")

    # ── Cell 3-4: Clone & Unit Tests ────────────────────────────────────────────────────────────
    add_md("""---
## Cell 2: Clone Nhánh `Nguyen_Thanh_Hai_tuan4`, Cài đặt Thư viện & Chạy Unit Tests""")

    add_code("""import subprocess

if IN_COLAB:
    if not (PROJECT_ROOT / ".git").exists():
        subprocess.check_call(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(PROJECT_ROOT)])
    else:
        subprocess.check_call(["git", "fetch", "origin", REPO_BRANCH], cwd=PROJECT_ROOT)
        subprocess.check_call(["git", "checkout", REPO_BRANCH], cwd=PROJECT_ROOT)
        subprocess.check_call(["git", "pull", "--ff-only", "origin", REPO_BRANCH], cwd=PROJECT_ROOT)

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if IN_COLAB:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    pkgs = [
        "git+https://github.com/facebookresearch/segment-anything.git",
        "albumentations>=2.0",
        "opencv-python-headless",
        "scipy",
        "scikit-learn",
        "kagglehub",
        "kaggle",
        "matplotlib",
        "pandas",
        "python-docx",
        "tqdm",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Chạy toàn bộ Unit Tests Tuần 4 (MedSAM-LoRA, Hybrid Peak Point, Per-Class Eval, Losses, Box Jitter)
subprocess.check_call([sys.executable, "-m", "unittest", "tests.test_medsam_lora", "-v"])
print("✅ Môi trường và mã nguồn Tuần 4 (Box + Hybrid Prompt) đã sẵn sàng!")""")

    # ── Cell 5-6: MedSAM Pretrained Checkpoint ──────────────────────────────────────────────────
    add_md("""---
## Cell 3: Tải Trọng số Gốc `medsam_vit_b.pth` & Xác thực MD5""")

    add_code("""import shutil
from utils.checkpoints import CHECKPOINTS, checkpoint_path, download_checkpoint, md5sum, verify_checkpoint

local_ckpt = checkpoint_path("medsam")
drive_ckpt = DRIVE_CKPT_DIR / CHECKPOINTS["medsam"]["file"]
local_ckpt.parent.mkdir(parents=True, exist_ok=True)

if verify_checkpoint("medsam"):
    print(f"[OK] Đã có sẵn {local_ckpt} đúng MD5.")
elif drive_ckpt.is_file() and md5sum(drive_ckpt) == CHECKPOINTS["medsam"]["md5"]:
    print(f"Đang chép medsam_vit_b.pth từ cache Drive ({drive_ckpt}) về Colab...")
    shutil.copy2(drive_ckpt, local_ckpt)
else:
    print("Đang tải medsam_vit_b.pth từ Zenodo (375 MB, có hỗ trợ resume & kiểm tra MD5)...")
    download_checkpoint("medsam", ckpt_dir=local_ckpt.parent)
    if IN_COLAB:
        shutil.copy2(local_ckpt, drive_ckpt)
        print(f"Đã lưu cache lên Drive: {drive_ckpt}")

assert verify_checkpoint("medsam"), "MD5 của checkpoints/medsam_vit_b.pth không khớp!"
size_mb = local_ckpt.stat().st_size / (1024 ** 2)
print(f"✅ Sẵn sàng trọng số gốc: {local_ckpt} ({size_mb:.1f} MB | MD5={md5sum(local_ckpt)})")""")

    # ── Cell 7-8: Prepare BRISC 2025 Dataset ────────────────────────────────────────────────────
    add_md("""---
## Cell 4: Chuẩn bị Dữ liệu `BRISC 2025` (Tương thích Cache Zip cả Tuần 3 & Tuần 4)""")

    add_code("""import json
import zipfile
import pandas as pd

ann_path = PROJECT_ROOT / "dataset" / "ds3_annotations.json"
ds3_all = json.loads(ann_path.read_text(encoding="utf-8-sig"))
recs = [r for r in ds3_all if bool(r.get("has_mask")) and bool(r.get("mask_path")) and int(r.get("has_tumor", 0)) == 1]
assert recs, f"Không tìm thấy mẫu có mask trong {ann_path}"

sample_img = PROJECT_ROOT / recs[0]["img_path"]
sample_msk = PROJECT_ROOT / recs[0]["mask_path"]

def _extract_ds3_zip_safely(zpath: Path):
    with zipfile.ZipFile(zpath, "r") as zf:
        names = zf.namelist()
    # Nếu zip được đóng gói từ gốc repo -> chứa prefix 'dataset/ds3_processed/...'
    if any(n.startswith("dataset/") for n in names[:20]):
        target_dir = PROJECT_ROOT
    else:
        target_dir = PROJECT_ROOT / "dataset"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"⚡ Đang giải nén cache {zpath} -> {target_dir} ...")
    subprocess.check_call(["unzip", "-q", "-o", str(zpath), "-d", str(target_dir)])

# 1. Tìm cache zip từ Tuần 4 hoặc Tuần 3 trên Google Drive
if not (sample_img.exists() and sample_msk.exists()) and IN_COLAB:
    zip_candidates = [
        DRIVE_DATA_CACHE / "ds3_processed.zip",
        Path("/content/drive/MyDrive/ds3_processed.zip"),
    ]
    for zc in zip_candidates:
        if zc.is_file():
            _extract_ds3_zip_safely(zc)
            if sample_img.exists() and sample_msk.exists():
                break

# 2. Nếu vẫn chưa có -> tải từ Kaggle (kagglehub hoặc kaggle CLI) & chạy data.build_ds3
if not (sample_img.exists() and sample_msk.exists()):
    kaggle_cfg = Path("/root/.kaggle/kaggle.json")
    if not kaggle_cfg.exists() and Path("/content/kaggle.json").exists():
        kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2("/content/kaggle.json", kaggle_cfg)
        kaggle_cfg.chmod(0o600)
    import kagglehub
    ann_backup = ann_path.read_text(encoding="utf-8-sig")
    try:
        from google.colab import userdata
        k_user, k_key = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if k_user and k_key:
            os.environ["KAGGLE_USERNAME"] = k_user
            os.environ["KAGGLE_KEY"] = k_key
    except Exception:
        pass
    print("📥 Đang tải BRISC 2025 từ Kaggle (briscdataset/brisc2025)...")
    dl_path = Path(kagglehub.dataset_download("briscdataset/brisc2025"))
    src_raw = dl_path / "brisc2025" if (dl_path / "brisc2025" / "classification_task").exists() else dl_path
    dst_raw = PROJECT_ROOT / "dataset" / "brisc2025" / "brisc2025"
    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    if not dst_raw.exists():
        try:
            os.symlink(src_raw, dst_raw, target_is_directory=True)
        except OSError:
            shutil.copytree(src_raw, dst_raw, dirs_exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "data.build_ds3", "--root", str(PROJECT_ROOT)])
    ann_path.write_text(ann_backup, encoding="utf-8")

    if IN_COLAB and (PROJECT_ROOT / "dataset" / "ds3_processed").is_dir():
        cache_dst = DRIVE_DATA_CACHE / "ds3_processed.zip"
        print(f"📦 Lưu cache lên Drive: {cache_dst} ...")
        subprocess.call(["zip", "-q", "-r", str(cache_dst), "dataset/ds3_processed"], cwd=PROJECT_ROOT)

missing_img = [r["img_path"] for r in recs if not (PROJECT_ROOT / r["img_path"]).is_file()]
missing_msk = [r["mask_path"] for r in recs if not (PROJECT_ROOT / r["mask_path"]).is_file()]
assert not missing_img and not missing_msk, (
    f"Thiếu {len(missing_img)} ảnh / {len(missing_msk)} mask! Ví dụ: {missing_img[:2]} {missing_msk[:2]}"
)

CLASS_NAMES = {0: "no_tumor", 1: "glioma", 2: "meningioma", 3: "pituitary"}
df_ds = pd.DataFrame([
    {"split": r["split"], "class": CLASS_NAMES.get(int(r.get("sub_class_label", 1)), "tumor")}
    for r in recs
])
print(f"✅ Dữ liệu phân vùng BRISC 2025 sẵn sàng ({len(df_ds):,} lát cắt có u & mask):")
display(pd.crosstab(df_ds["split"], df_ds["class"], margins=True))""")

    # ── Cell 9-10: Architecture Check & Prompt Visualization ───────────────────────────────────
    add_md("""---
## Cell 5: Kiểm tra Tham số `MedSAM-LoRA` & Trực quan hóa `Box` vs `Hybrid (Box + Peak Point)`""")

    add_code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.unified_dataset import DEFAULT_ANNOTATION_FILES, UnifiedBrainMRIDataset
from models.medsam_lora import MedSamLoRA, extract_peak_points, jitter_boxes
from utils.losses import SegmentationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
model_check = MedSamLoRA.from_checkpoint(
    checkpoint_path("medsam"),
    lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    device="cpu", verify=True,
)
counts = model_check.param_counts()
df_params = pd.DataFrame([
    {"Thành phần": "Image Encoder (ViT-B đóng băng)", "Số tham số": counts["image_encoder_frozen_params"], "Trạng thái": "Frozen"},
    {"Thành phần": "Prompt Encoder (Box + Point)", "Số tham số": counts["total"] - counts["image_encoder_frozen_params"] - counts["trainable"], "Trạng thái": "Frozen"},
    {"Thành phần": f"LoRA (r={LORA_R}, α={LORA_ALPHA}, 12 khối Q/V)", "Số tham số": counts["lora_params"], "Trạng thái": "Trainable"},
    {"Thành phần": "Mask Decoder (full fine-tuning)", "Số tham số": counts["mask_decoder_params"], "Trạng thái": "Trainable"},
    {"Thành phần": "TỔNG KHẢ HUẤN", "Số tham số": counts["trainable"], "Trạng thái": f"{counts['trainable_pct']:.2f}% / {counts['total']:,}"},
])
display(df_params)
del model_check

# Minh họa 1 mẫu đại diện cho từng nhóm bệnh: Glioma (1), Meningioma (2), Pituitary (3)
vis_ds = UnifiedBrainMRIDataset(list(DEFAULT_ANNOTATION_FILES), train=False, datasets=["brisc2025"], split="val")
vis_ds.records = [r for r in vis_ds.records if bool(r.get("has_mask")) and int(r.get("has_tumor", 0)) == 1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, target_lbl in zip(axes, [1, 2, 3]):
    idx = next(i for i, r in enumerate(vis_ds.records) if int(r["sub_class_label"]) == target_lbl)
    sample = vis_ds[idx]
    img_t = sample["image_1024"].unsqueeze(0)
    box_t = sample["gt_box"].unsqueeze(0)
    jit_b = jitter_boxes(box_t, max_px=BOX_JITTER, img_size=1024)[0].numpy()
    pts_coords, _ = extract_peak_points(img_t, box_t)

    img = np.clip(sample["image_1024"].permute(1, 2, 0).numpy(), 0, 1)
    msk = sample["gt_mask"].numpy()
    gt_b = box_t[0].numpy()
    px, py = pts_coords[0, 0].tolist()

    ax.imshow(img)
    ax.imshow(np.ma.masked_where(msk < 0.5, msk), cmap="autumn", alpha=0.40)
    ax.add_patch(patches.Rectangle((gt_b[0], gt_b[1]), gt_b[2] - gt_b[0], gt_b[3] - gt_b[1],
                                   lw=2.2, ec="lime", fc="none", label="GT Box"))
    ax.add_patch(patches.Rectangle((jit_b[0], jit_b[1]), jit_b[2] - jit_b[0], jit_b[3] - jit_b[1],
                                   lw=1.8, ec="red", ls="--", fc="none", label=f"Jitter ±{int(BOX_JITTER)}px"))
    ax.scatter([px], [py], s=180, c="yellow", marker="*", edgecolors="black", linewidths=1.2, zorder=5, label="Peak Point (+)")
    ax.set_title(f"{CLASS_NAMES[target_lbl].capitalize()} | Hybrid Prompt", fontsize=12, fontweight="bold")
    ax.axis("off")
    if target_lbl == 1:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

plt.suptitle("Minh họa Hybrid Prompt (Box + Peak Positive Point) & Box Jittering ±10px trên BRISC 2025", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_OUT_DIR / "sample_hybrid_prompt_by_class.png", dpi=160, bbox_inches="tight")
plt.show()""")

    # ── Cell 11-12: Smoke Test (Optional) ───────────────────────────────────────────────────────
    add_md("""---
## Cell 6: Kiểm thử Nhanh Pipeline (`RUN_SMOKE_TEST = False` — Bật `True` nếu muốn chạy thử 16 mẫu)""")

    add_code("""def run_stream(cmd: list[str]):
    print("COMMAND:", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1, env=env) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)

def build_train_cmd(prompt_mode: str, out_dir: Path, limit: int = 0, epochs: int | None = None, resume: str = "auto") -> list[str]:
    cmd = [
        sys.executable, "-m", "engine.train_medsam_lora",
        "--datasets", *TARGET_DATASETS,
        "--lora-r", str(LORA_R),
        "--lora-alpha", str(LORA_ALPHA),
        "--lora-dropout", str(LORA_DROPOUT),
        "--loss-mode", LOSS_MODE,
        "--dice-weight", str(DICE_WEIGHT),
        "--pixel-weight", str(PIXEL_WEIGHT),
        "--box-jitter", str(BOX_JITTER),
        "--prompt-mode", prompt_mode,
        "--epochs", str(epochs if epochs is not None else EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum", str(GRAD_ACCUM if limit == 0 else 2),
        "--lr-lora", str(LR_LORA),
        "--lr-decoder", str(LR_DECODER),
        "--weight-decay", str(WEIGHT_DECAY),
        "--patience", str(PATIENCE),
        "--workers", str(NUM_WORKERS),
        "--seed", str(SEED),
        "--resume", resume,
        "--no-val-hd95",   # Bỏ EDT HD95 lúc val để tiết kiệm ~3.5 phút/epoch; Test vẫn tính full HD95 1024x1024
        "--out", str(out_dir),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if torch.cuda.is_available():
        cmd.append("--amp")
    return cmd

if RUN_SMOKE_TEST:
    for mode in ("box", "hybrid"):
        print(f"\\n=== Smoke test prompt={mode} (16 mẫu, 1 epoch) ===")
        run_stream(build_train_cmd(mode, DRIVE_WORKDIR / f"smoke_{mode}", limit=16, epochs=1, resume="never"))
else:
    print("RUN_SMOKE_TEST = False -> Bỏ qua chạy thử, chuyển sang Cell 7 & Cell 8 để huấn luyện full BRISC 2025.")""")

    # ── Cell 13-14: Train Model A (Box-only) ────────────────────────────────────────────────────
    add_md("""---
## Cell 7: Huấn luyện Cấu hình 1 — `MedSAM-LoRA (r = 8)` với `Box-only Prompt`

> Tự động lưu `latest_medsam_lora.pt` và `best_medsam_lora.pt` vào Drive (`RUN_BOX_DIR`) sau mỗi epoch.""")

    add_code("""print("=== [1/2] Huấn luyện MedSAM-LoRA (r=8) — Chiến lược BOX-ONLY ===")
run_stream(build_train_cmd("box", RUN_BOX_DIR))
print(f"✅ Hoàn thành huấn luyện Box-only! Kết quả lưu tại: {RUN_BOX_DIR}")""")

    # ── Cell 15-16: Train Model B (Hybrid: Box + Peak Point) ────────────────────────────────────
    add_md("""---
## Cell 8: Huấn luyện Cấu hình 2 — `MedSAM-LoRA (r = 8)` với `Hybrid Prompt (Box + Peak Point)`

> Cùng tập BRISC 2025, cùng kiến trúc `r = 8`, cùng `0.5 Dice + 0.5 Focal`, cùng `Box Jittering ±10px` và `seed = 42`. Chỉ thay đổi chiến lược prompt sang `Hybrid (Box + Peak Positive Point)`.""")

    add_code("""print("=== [2/2] Huấn luyện MedSAM-LoRA (r=8) — Chiến lược HYBRID (Box + Peak Point) ===")
run_stream(build_train_cmd("hybrid", RUN_HYBRID_DIR))
print(f"✅ Hoàn thành huấn luyện Hybrid! Kết quả lưu tại: {RUN_HYBRID_DIR}")""")

    # ── Cell 17-18: Comprehensive Evaluation & Ablation ─────────────────────────────────────────
    add_md("""---
## Cell 9: Đánh giá Tập Test BRISC 2025 (`n = 719`) — Theo Nhóm Bệnh & Ablation `Box` vs `Hybrid`

Thực hiện đánh giá ghép cặp công bằng trên toàn bộ **719 lát cắt Test** (`210 Glioma`, `245 Meningioma`, `264 Pituitary`):
1. **Kịch bản chuẩn (Oracle GT Box):** Đối sánh `Zero-shot MedSAM`, `MedSAM-LoRA (Box-only)` và `MedSAM-LoRA (Hybrid: Box + Peak Point)`.
2. **Kịch bản nhiễu hộp (`Perturbed Box ±15px`, cố định `seed = 42`):** Mô phỏng hộp dự đoán bị lệch từ Stage 1 để kiểm định tác dụng neo tâm khối u của `Peak Positive Point`.
3. **Kiểm định thống kê ghép cặp:** `Paired Wilcoxon Signed-Rank Test` & `Paired t-test` giữa `Box-only` và `Hybrid`.""")

    add_code("""from collections import defaultdict
from scipy import stats
from torch.utils.data import DataLoader
from utils.metrics import SegmentationMeter

test_ds = UnifiedBrainMRIDataset(list(DEFAULT_ANNOTATION_FILES), train=False, datasets=["brisc2025"], split="test")
test_ds.records = [
    r for r in test_ds.records
    if bool(r.get("has_mask")) and bool(r.get("mask_path")) and int(r.get("has_tumor", 0)) == 1
]
if EVAL_MAX_SAMPLES and len(test_ds.records) > EVAL_MAX_SAMPLES:
    rng = np.random.RandomState(SEED)
    keep = sorted(rng.choice(len(test_ds.records), EVAL_MAX_SAMPLES, replace=False))
    test_ds.records = [test_ds.records[i] for i in keep]

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=torch.cuda.is_available())
use_amp = torch.cuda.is_available()

@torch.no_grad()
def evaluate_prompt_strategy(
    model: MedSamLoRA,
    loader: DataLoader,
    prompt_mode: str,
    tag: str,
    box_jitter_px: float = 0.0,
    seed: int = 42,
) -> dict:
    model.eval()
    overall = SegmentationMeter()
    per_class = defaultdict(SegmentationMeter)
    per_sample_rows = []
    vis_samples = {}  # Lưu 1 mẫu đầu tiên của mỗi nhóm bệnh (glioma, meningioma, pituitary)
    gen = torch.Generator(device=device).manual_seed(seed) if box_jitter_px > 0.0 else None

    sample_idx = 0
    for batch in loader:
        images = batch["image_1024"].to(device, non_blocking=True)
        boxes = batch["gt_box"].to(device, non_blocking=True)
        if box_jitter_px > 0.0:
            boxes = jitter_boxes(boxes, max_px=box_jitter_px, img_size=images.shape[-1], generator=gen)
        points = extract_peak_points(images, boxes) if prompt_mode == "hybrid" else None

        masks = batch["gt_mask"].numpy() > 0.5
        labels = batch["sub_class_label"].tolist()
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            logits = model(images, boxes, points=points, upsample=True)
        preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1).cpu().numpy()

        for idx, (pm, gm, lbl) in enumerate(zip(preds, masks, labels)):
            row = overall.update(pm, gm)
            cname = CLASS_NAMES.get(int(lbl), f"class_{int(lbl)}")
            per_class[cname].rows.append(row)
            per_sample_rows.append({
                "idx": sample_idx,
                "model_tag": tag,
                "prompt_mode": prompt_mode,
                "box_jitter_px": float(box_jitter_px),
                "class": cname,
                "dice": row["dice"],
                "iou": row["iou"],
                "hd95": row["hd95"],
            })
            if cname not in vis_samples:
                vis_samples[cname] = {
                    "idx": sample_idx,
                    "image": batch["image_1024"][idx].permute(1, 2, 0).numpy(),
                    "box": boxes[idx].detach().cpu().numpy(),
                    "peak_pt": points[0][idx, 0].detach().cpu().numpy() if points is not None else None,
                    "gt": gm,
                    "pred": pm,
                    "dice": row["dice"],
                    "iou": row["iou"],
                    "hd95": row["hd95"],
                    "class": cname,
                }
            sample_idx += 1

    ov = overall.summary()
    print(f"[{tag}] n={ov['n']} | Dice={ov['dice']['mean']*100:.2f}% | IoU={ov['iou']['mean']*100:.2f}% | HD95={ov['hd95']['mean']:.2f} px")
    return {
        "overall": ov,
        "per_class": {k: v.summary() for k, v in sorted(per_class.items())},
        "per_sample": per_sample_rows,
        "vis_samples": vis_samples,
    }

# Nạp mô hình Zero-shot MedSAM gốc (làm baseline đối chiếu Tuần 2)
m_zs = MedSamLoRA.from_checkpoint(checkpoint_path("medsam"), lora_r=LORA_R, lora_alpha=LORA_ALPHA, device=device, verify=True)
res_zs_box = evaluate_prompt_strategy(m_zs, test_loader, prompt_mode="box", tag="Zero-shot MedSAM (Box)")

# Nạp MedSAM-LoRA (Box-only)
ckpt_box = RUN_BOX_DIR / "best_medsam_lora.pt"
assert ckpt_box.is_file(), f"Thiếu {ckpt_box}! Hãy chạy Cell 7 trước."
m_zs.load_checkpoint(ckpt_box, map_location=device)
res_lora_box = evaluate_prompt_strategy(m_zs, test_loader, prompt_mode="box", tag="MedSAM-LoRA r=8 (Box)")
res_lora_box_jit = evaluate_prompt_strategy(
    m_zs, test_loader, prompt_mode="box", tag=f"MedSAM-LoRA Box (Jitter ±{int(EVAL_PERTURBED_JITTER)}px)",
    box_jitter_px=EVAL_PERTURBED_JITTER, seed=SEED,
)

# Nạp MedSAM-LoRA (Hybrid: Box + Peak Point)
ckpt_hyb = RUN_HYBRID_DIR / "best_medsam_lora.pt"
assert ckpt_hyb.is_file(), f"Thiếu {ckpt_hyb}! Hãy chạy Cell 8 trước."
m_zs.load_checkpoint(ckpt_hyb, map_location=device)
res_lora_hyb = evaluate_prompt_strategy(m_zs, test_loader, prompt_mode="hybrid", tag="MedSAM-LoRA r=8 (Hybrid)")
res_lora_hyb_jit = evaluate_prompt_strategy(
    m_zs, test_loader, prompt_mode="hybrid", tag=f"MedSAM-LoRA Hybrid (Jitter ±{int(EVAL_PERTURBED_JITTER)}px)",
    box_jitter_px=EVAL_PERTURBED_JITTER, seed=SEED,
)

# 1. Bảng Tổng hợp Ablation Chiến lược Prompt (Box vs Hybrid)
def _row_summary(name: str, prompt: str, jitter: float, ov: dict) -> dict:
    return {
        "Mô hình": name,
        "Chiến lược Prompt": prompt,
        "Box Jitter Eval": f"±{int(jitter)}px" if jitter > 0 else "0px (Oracle Box)",
        "Số mẫu": ov["n"],
        "Dice Mean (%)": round(ov["dice"]["mean"] * 100, 2),
        "Dice SD (%)": round(ov["dice"]["sd"] * 100, 2),
        "Dice Median (%)": round(ov["dice"]["median"] * 100, 2),
        "IoU Mean (%)": round(ov["iou"]["mean"] * 100, 2),
        "HD95 Mean (px)": round(ov["hd95"]["mean"], 2),
        "HD95 Median (px)": round(ov["hd95"]["median"], 2),
    }

df_overall = pd.DataFrame([
    _row_summary("Zero-shot MedSAM (Baseline)", "Box-only", 0.0, res_zs_box["overall"]),
    _row_summary("MedSAM-LoRA (r=8)", "Box-only", 0.0, res_lora_box["overall"]),
    _row_summary("MedSAM-LoRA (r=8)", "Hybrid (Box + Peak Point)", 0.0, res_lora_hyb["overall"]),
])
df_perturbed = pd.DataFrame([
    _row_summary("MedSAM-LoRA (r=8)", "Box-only", EVAL_PERTURBED_JITTER, res_lora_box_jit["overall"]),
    _row_summary("MedSAM-LoRA (r=8)", "Hybrid (Box + Peak Point)", EVAL_PERTURBED_JITTER, res_lora_hyb_jit["overall"]),
])

print("\\n=== BẢNG 1: ĐỐI SÁNH CHIẾN LƯỢC PROMPT (ORACLE BOX — BRISC 2025 TEST) ===")
display(df_overall)
print(f"\\n=== BẢNG 2: KIỂM ĐỊNH ĐỘ BỀN VỮNG KHI HỘP NHIỄU (±{int(EVAL_PERTURBED_JITTER)}px) ===")
display(df_perturbed)

# 2. Bảng Đánh giá theo từng nhóm bệnh (Glioma, Meningioma, Pituitary)
cls_rows = []
for cname in ("glioma", "meningioma", "pituitary"):
    zs_c = res_zs_box["per_class"][cname]
    bx_c = res_lora_box["per_class"][cname]
    hy_c = res_lora_hyb["per_class"][cname]
    cls_rows.append({
        "Nhóm bệnh": cname.capitalize(),
        "Số mẫu": bx_c["n"],
        "Zero-shot Dice (%)": round(zs_c["dice"]["mean"] * 100, 2),
        "LoRA Box Dice (%)": round(bx_c["dice"]["mean"] * 100, 2),
        "LoRA Hybrid Dice (%)": round(hy_c["dice"]["mean"] * 100, 2),
        "ΔDice (Hybrid - Box)": round((hy_c["dice"]["mean"] - bx_c["dice"]["mean"]) * 100, 2),
        "LoRA Box IoU (%)": round(bx_c["iou"]["mean"] * 100, 2),
        "LoRA Hybrid IoU (%)": round(hy_c["iou"]["mean"] * 100, 2),
        "Zero-shot HD95 (px)": round(zs_c["hd95"]["mean"], 2),
        "LoRA Box HD95 (px)": round(bx_c["hd95"]["mean"], 2),
        "LoRA Hybrid HD95 (px)": round(hy_c["hd95"]["mean"], 2),
        "ΔHD95 (Hybrid - Box)": round(hy_c["hd95"]["mean"] - bx_c["hd95"]["mean"], 2),
    })
df_per_class = pd.DataFrame(cls_rows)
print("\\n=== BẢNG 3: KẾT QUẢ PHÂN ĐOẠN THEO TỪNG NHÓM BỆNH (GLIOMA, MENINGIOMA, PITUITARY) ===")
display(df_per_class)

# 3. Kiểm định thống kê ghép cặp (Paired Wilcoxon Signed-Rank Test)
def _paired_stat(res_a: dict, res_b: dict, metric: str = "dice") -> dict:
    va = np.array([r[metric] for r in res_a["per_sample"]], dtype=float)
    vb = np.array([r[metric] for r in res_b["per_sample"]], dtype=float)
    valid = ~np.isnan(va) & ~np.isnan(vb)
    diff = vb[valid] - va[valid]
    if np.allclose(diff, 0.0):
        w_p, t_p = 1.0, 1.0
    else:
        _, w_p = stats.wilcoxon(vb[valid], va[valid])
        _, t_p = stats.ttest_rel(vb[valid], va[valid])
    return {
        "n": int(valid.sum()),
        "mean_box": float(va[valid].mean()),
        "mean_hybrid": float(vb[valid].mean()),
        "mean_diff": float(diff.mean()),
        "wilcoxon_p": float(w_p),
        "paired_t_p": float(t_p),
    }

stat_tests = {
    "oracle_dice_box_vs_hybrid": _paired_stat(res_lora_box, res_lora_hyb, "dice"),
    "oracle_hd95_box_vs_hybrid": _paired_stat(res_lora_box, res_lora_hyb, "hd95"),
    "perturbed_dice_box_vs_hybrid": _paired_stat(res_lora_box_jit, res_lora_hyb_jit, "dice"),
    "perturbed_hd95_box_vs_hybrid": _paired_stat(res_lora_box_jit, res_lora_hyb_jit, "hd95"),
}
print("\\n=== KIỂM ĐỊNH THỐNG KÊ GHÉP CẶP (BOX vs HYBRID) ===")
display(pd.DataFrame(stat_tests).T)

# Lưu toàn bộ CSV & JSON vào Google Drive lẫn thư mục reports/tuan4/medsam_lora_hai/
LOCAL_REP_DIR = PROJECT_ROOT / "reports" / "tuan4" / "medsam_lora_hai"
LOCAL_REP_DIR.mkdir(parents=True, exist_ok=True)
df_all_samples = pd.DataFrame(
    res_zs_box["per_sample"]
    + res_lora_box["per_sample"]
    + res_lora_hyb["per_sample"]
    + res_lora_box_jit["per_sample"]
    + res_lora_hyb_jit["per_sample"]
)

for target_dir in (REPORT_OUT_DIR, LOCAL_REP_DIR):
    df_overall.to_csv(target_dir / "overall_prompt_ablation.csv", index=False)
    df_perturbed.to_csv(target_dir / "perturbed_box_ablation.csv", index=False)
    df_per_class.to_csv(target_dir / "per_class_box_vs_hybrid.csv", index=False)
    df_all_samples.to_csv(target_dir / "per_sample_predictions.csv", index=False)
    (target_dir / "week4_hai_summary.json").write_text(
        json.dumps(
            {
                "overall_oracle": df_overall.to_dict(orient="records"),
                "overall_perturbed": df_perturbed.to_dict(orient="records"),
                "per_class": df_per_class.to_dict(orient="records"),
                "statistical_tests": stat_tests,
                "raw": {
                    "zeroshot_box": res_zs_box["overall"],
                    "lora_box": {"overall": res_lora_box["overall"], "per_class": res_lora_box["per_class"]},
                    "lora_hybrid": {"overall": res_lora_hyb["overall"], "per_class": res_lora_hyb["per_class"]},
                    "lora_box_jitter": res_lora_box_jit["overall"],
                    "lora_hybrid_jitter": res_lora_hyb_jit["overall"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
print(f"✅ Đã lưu toàn bộ số liệu định lượng tại {REPORT_OUT_DIR} và {LOCAL_REP_DIR}")""")

    # ── Cell 19-20: Charts & Qualitative Grid ───────────────────────────────────────────────────
    add_md("""---
## Cell 10: Vẽ 4 Biểu đồ Khoa học & Lưới Ảnh So sánh Định tính (`image/tuan4/hai/`)""")

    add_code("""LOCAL_FIG_DIR = PROJECT_ROOT / "image" / "tuan4" / "hai"
LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_fig_both(fig, filename: str):
    for d in (REPORT_OUT_DIR, LOCAL_FIG_DIR):
        fig.savefig(d / filename, dpi=180, bbox_inches="tight")

# Figure 1: Đường cong hội tụ Loss & Val Dice/IoU giữa Box-only vs Hybrid
hist_box = pd.read_csv(RUN_BOX_DIR / "history.csv")
hist_hyb = pd.read_csv(RUN_HYBRID_DIR / "history.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].plot(hist_box["epoch"], hist_box["train_loss"], "--o", color="#1f77b4", alpha=0.7, label="Box-only Train Loss")
axes[0].plot(hist_box["epoch"], hist_box["val_loss"], "-o", color="#1f77b4", label="Box-only Val Loss")
axes[0].plot(hist_hyb["epoch"], hist_hyb["train_loss"], "--s", color="#d62728", alpha=0.7, label="Hybrid Train Loss")
axes[0].plot(hist_hyb["epoch"], hist_hyb["val_loss"], "-s", color="#d62728", label="Hybrid Val Loss")
axes[0].set_title("Hội tụ Hàm mất mát (0.5 Dice + 0.5 Focal)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

axes[1].plot(hist_box["epoch"], hist_box["val_dice"] * 100, "-o", color="#1f77b4", label="Box-only Val Dice (%)")
axes[1].plot(hist_hyb["epoch"], hist_hyb["val_dice"] * 100, "-s", color="#d62728", label="Hybrid Val Dice (%)")
axes[1].set_title("Validation Dice (%) theo Epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice (%)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

axes[2].plot(hist_box["epoch"], hist_box["val_iou"] * 100, "-o", color="#1f77b4", label="Box-only Val IoU (%)")
axes[2].plot(hist_hyb["epoch"], hist_hyb["val_iou"] * 100, "-s", color="#d62728", label="Hybrid Val IoU (%)")
axes[2].set_title("Validation IoU (%) theo Epoch")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("IoU (%)"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=9)

plt.suptitle("Quá trình Huấn luyện MedSAM-LoRA (r=8) trên BRISC 2025: Box-only vs Hybrid", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig_both(fig, "fig1_medsam_lora_training_curves.png")
plt.show()

# Figure 2: Phân tích theo từng nhóm bệnh (Glioma, Meningioma, Pituitary)
classes_disp = df_per_class["Nhóm bệnh"].tolist()
x = np.arange(len(classes_disp))
w = 0.26

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
b1 = axes[0].bar(x - w, df_per_class["Zero-shot Dice (%)"], w, label="Zero-shot MedSAM", color="#95a5a6")
b2 = axes[0].bar(x, df_per_class["LoRA Box Dice (%)"], w, label="MedSAM-LoRA (Box-only)", color="#2980b9")
b3 = axes[0].bar(x + w, df_per_class["LoRA Hybrid Dice (%)"], w, label="MedSAM-LoRA (Hybrid)", color="#e74c3c")
axes[0].set_xticks(x); axes[0].set_xticklabels(classes_disp, fontsize=11)
axes[0].set_ylabel("Dice Score (%)"); axes[0].set_ylim(max(50, df_per_class["Zero-shot Dice (%)"].min() - 8), 100)
axes[0].set_title("Dice Score (%) theo Nhóm Bệnh"); axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=9)
for bars in (b1, b2, b3):
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}",
                     ha="center", va="bottom", fontsize=8)

h1 = axes[1].bar(x - w, df_per_class["Zero-shot HD95 (px)"], w, label="Zero-shot MedSAM", color="#95a5a6")
h2 = axes[1].bar(x, df_per_class["LoRA Box HD95 (px)"], w, label="MedSAM-LoRA (Box-only)", color="#2980b9")
h3 = axes[1].bar(x + w, df_per_class["LoRA Hybrid HD95 (px)"], w, label="MedSAM-LoRA (Hybrid)", color="#e74c3c")
axes[1].set_xticks(x); axes[1].set_xticklabels(classes_disp, fontsize=11)
axes[1].set_ylabel("HD95 (pixels — thấp hơn là tốt hơn)")
axes[1].set_title("Sai số Biên HD95 (px) theo Nhóm Bệnh"); axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=9)
for bars in (h1, h2, h3):
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}",
                     ha="center", va="bottom", fontsize=8)

plt.suptitle("Đánh giá Phân đoạn MedSAM-LoRA theo 3 Nhóm Bệnh trên BRISC 2025 Test (n=719)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig_both(fig, "fig2_per_class_tumor_comparison.png")
plt.show()

# Figure 3: Ablation Chiến lược Prompt (Oracle Box vs Perturbed Box ±15px)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
scenarios = ["Oracle Box (0px)", f"Perturbed Box (±{int(EVAL_PERTURBED_JITTER)}px)"]
box_dices = [res_lora_box["overall"]["dice"]["mean"] * 100, res_lora_box_jit["overall"]["dice"]["mean"] * 100]
hyb_dices = [res_lora_hyb["overall"]["dice"]["mean"] * 100, res_lora_hyb_jit["overall"]["dice"]["mean"] * 100]
box_hd95 = [res_lora_box["overall"]["hd95"]["mean"], res_lora_box_jit["overall"]["hd95"]["mean"]]
hyb_hd95 = [res_lora_hyb["overall"]["hd95"]["mean"], res_lora_hyb_jit["overall"]["hd95"]["mean"]]

sx = np.arange(len(scenarios))
sw = 0.32
r1 = axes[0].bar(sx - sw / 2, box_dices, sw, label="Box-only", color="#2980b9")
r2 = axes[0].bar(sx + sw / 2, hyb_dices, sw, label="Hybrid (Box + Peak Point)", color="#e74c3c")
axes[0].set_xticks(sx); axes[0].set_xticklabels(scenarios, fontsize=10)
axes[0].set_ylabel("Dice Score (%)"); axes[0].set_ylim(max(60, min(box_dices + hyb_dices) - 6), 100)
axes[0].set_title("Ablation Prompt theo Độ chính xác Box (Dice %)"); axes[0].grid(axis="y", alpha=0.3); axes[0].legend()
for bars in (r1, r2):
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%",
                     ha="center", va="bottom", fontsize=9)

r3 = axes[1].bar(sx - sw / 2, box_hd95, sw, label="Box-only", color="#2980b9")
r4 = axes[1].bar(sx + sw / 2, hyb_hd95, sw, label="Hybrid (Box + Peak Point)", color="#e74c3c")
axes[1].set_xticks(sx); axes[1].set_xticklabels(scenarios, fontsize=10)
axes[1].set_ylabel("HD95 Mean (px)"); axes[1].set_title("Ablation Prompt theo Độ chính xác Box (HD95 px)")
axes[1].grid(axis="y", alpha=0.3); axes[1].legend()
for bars in (r3, r4):
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}",
                     ha="center", va="bottom", fontsize=9)

plt.suptitle("Ablation Chiến lược Prompt: Chỉ dùng Box vs Hybrid (Box + Peak Point)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig_both(fig, "fig3_prompt_ablation_box_vs_hybrid.png")
plt.show()

# Figure 4: Lưới ảnh so sánh định tính cho 3 loại u (Glioma, Meningioma, Pituitary)
order_cls = ["glioma", "meningioma", "pituitary"]
fig, axes = plt.subplots(3, 5, figsize=(17, 10.5))
for r_i, cname in enumerate(order_cls):
    s_zs = res_zs_box["vis_samples"][cname]
    s_bx = res_lora_box["vis_samples"][cname]
    s_hy = res_lora_hyb["vis_samples"][cname]
    img = np.clip(s_hy["image"], 0, 1)
    b = s_hy["box"]
    pt = s_hy["peak_pt"]

    axes[r_i, 0].imshow(img)
    axes[r_i, 0].add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], lw=2, ec="lime", fc="none"))
    if pt is not None:
        axes[r_i, 0].scatter([pt[0]], [pt[1]], s=140, c="yellow", marker="*", edgecolors="black", zorder=5)
    axes[r_i, 0].set_title(f"{cname.capitalize()} | Box + Peak Point", fontsize=10, fontweight="bold")

    axes[r_i, 1].imshow(img)
    axes[r_i, 1].imshow(np.ma.masked_where(~s_hy["gt"], s_hy["gt"]), cmap="Greens", alpha=0.55)
    axes[r_i, 1].set_title("Ground Truth", fontsize=10)

    axes[r_i, 2].imshow(img)
    axes[r_i, 2].imshow(np.ma.masked_where(~s_zs["pred"], s_zs["pred"]), cmap="Oranges", alpha=0.55)
    axes[r_i, 2].set_title(f"Zero-shot\\nDice={s_zs['dice']*100:.1f}% | HD95={s_zs['hd95']:.1f}", fontsize=9.5)

    axes[r_i, 3].imshow(img)
    axes[r_i, 3].imshow(np.ma.masked_where(~s_bx["pred"], s_bx["pred"]), cmap="Blues", alpha=0.55)
    axes[r_i, 3].set_title(f"MedSAM-LoRA (Box)\\nDice={s_bx['dice']*100:.1f}% | HD95={s_bx['hd95']:.1f}", fontsize=9.5)

    axes[r_i, 4].imshow(img)
    axes[r_i, 4].imshow(np.ma.masked_where(~s_hy["pred"], s_hy["pred"]), cmap="cool", alpha=0.55)
    axes[r_i, 4].set_title(f"MedSAM-LoRA (Hybrid)\\nDice={s_hy['dice']*100:.1f}% | HD95={s_hy['hd95']:.1f}", fontsize=9.5)

    for c_i in range(5):
        axes[r_i, c_i].axis("off")

plt.suptitle("So sánh Định tính trên 3 Nhóm Bệnh BRISC 2025: Zero-shot vs Box-only vs Hybrid", fontsize=13.5, fontweight="bold", y=1.01)
plt.tight_layout()
save_fig_both(fig, "fig4_qualitative_box_vs_hybrid_grid.png")
plt.show()
print(f"✅ Đã lưu 4 biểu đồ khoa học vào {LOCAL_FIG_DIR} và {REPORT_OUT_DIR}")""")

    # ── Cell 21-22: Auto-Generate Week 4 Markdown Report & Push Instructions ────────────────────
    add_md("""---
## Cell 11: Tự động Sinh Báo cáo Tuần 4 (`reports/tuan4/BAO_CAO_TUAN4_HAI.md`) & Đóng gói Nộp Bài""")

    add_code("""def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\\n".join(lines)

ov_box = res_lora_box["overall"]
ov_hyb = res_lora_hyb["overall"]
ov_zs = res_zs_box["overall"]

report_md = f\"\"\"# BÁO CÁO THỰC NGHIỆM TUẦN 4 — Nguyễn Thanh Hải (`Nguyen_Thanh_Hai_tuan4`)

**Đề tài:** Huấn luyện `MedSAM-LoRA (r = 8, α = 16)` trên BRISC 2025, Đánh giá theo Nhóm Bệnh (`Glioma`, `Meningioma`, `Pituitary`) & Ablation Chiến lược Prompt (`Box` vs `Hybrid: Box + Peak Point`).

---

## 1. Cấu hình Huấn luyện & Kiểm định Dữ liệu BRISC 2025
- **Dữ liệu phân vùng (`has_mask=True, has_tumor=1`):** `3,355` Train / `719` Validation / `719` Test (`210` Glioma, `245` Meningioma, `264` Pituitary).
- **Kiến trúc:** Đóng băng Image Encoder (`ViT-B`) và Prompt Encoder, chèn LoRA (`r = {LORA_R}`, `α = {LORA_ALPHA}`, `dropout = {LORA_DROPOUT}`) vào nhánh `Query` và `Value` của 12 khối Attention, mở khóa toàn bộ `Mask Decoder`.
- **Hàm mất mát:** $L_{{\\\\text{{Seg}}}} = 0.5\\\\,L_{{\\\\text{{Dice}}}} + 0.5\\\\,L_{{\\\\text{{Focal}}}}$ + `Box Jittering ±{int(BOX_JITTER)}px` (hệ `1024×1024`).
- **Peak Positive Point (chiến lược Hybrid):** Trích xuất hoàn toàn tự động từ tương phản cường độ cục bộ và độ bất đối xứng bán cầu bên trong vùng lõi hộp (`extract_peak_points`), hoàn toàn không rò rỉ nhãn `gt_mask`.

---

## 2. Kết quả Ablation Chiến lược Prompt trên BRISC 2025 Test (`n = {ov_box['n']}`)

### 2.1. Đối sánh trên Oracle Box (`0px` Jitter)
{_df_to_md(df_overall)}

- **Cải thiện `MedSAM-LoRA (Box)` so với Zero-shot Tuần 2:** `ΔDice = {(ov_box['dice']['mean'] - ov_zs['dice']['mean'])*100:+.2f}%`, `ΔHD95 = {ov_box['hd95']['mean'] - ov_zs['hd95']['mean']:+.2f} px`.
- **So sánh `Hybrid (Box + Peak Point)` vs `Box-only` (Oracle Box):** `ΔDice = {(ov_hyb['dice']['mean'] - ov_box['dice']['mean'])*100:+.2f}%` (Wilcoxon $p = {stat_tests['oracle_dice_box_vs_hybrid']['wilcoxon_p']:.4e}$), `ΔHD95 = {ov_hyb['hd95']['mean'] - ov_box['hd95']['mean']:+.2f} px` ($p = {stat_tests['oracle_hd95_box_vs_hybrid']['wilcoxon_p']:.4e}$).

### 2.2. Kiểm định Độ bền vững khi Bounding Box bị Nhiễu (`Perturbed Box ±{int(EVAL_PERTURBED_JITTER)}px`)
{_df_to_md(df_perturbed)}

- **Chênh lệch khi Box nhiễu `±{int(EVAL_PERTURBED_JITTER)}px`:** `ΔDice = {(res_lora_hyb_jit['overall']['dice']['mean'] - res_lora_box_jit['overall']['dice']['mean'])*100:+.2f}%` (Wilcoxon $p = {stat_tests['perturbed_dice_box_vs_hybrid']['wilcoxon_p']:.4e}$), `ΔHD95 = {res_lora_hyb_jit['overall']['hd95']['mean'] - res_lora_box_jit['overall']['hd95']['mean']:+.2f} px`.

---

## 3. Đánh giá Chi tiết theo Từng Nhóm Bệnh (`Glioma`, `Meningioma`, `Pituitary`)
{_df_to_md(df_per_class)}

---

## 4. Biểu đồ & Minh họa Trực quan
- `image/tuan4/hai/fig1_medsam_lora_training_curves.png`
- `image/tuan4/hai/fig2_per_class_tumor_comparison.png`
- `image/tuan4/hai/fig3_prompt_ablation_box_vs_hybrid.png`
- `image/tuan4/hai/fig4_qualitative_box_vs_hybrid_grid.png`
\"\"\"

report_file = PROJECT_ROOT / "reports" / "tuan4" / "BAO_CAO_TUAN4_HAI.md"
report_file.parent.mkdir(parents=True, exist_ok=True)
report_file.write_text(report_md, encoding="utf-8")
(REPORT_OUT_DIR / "BAO_CAO_TUAN4_HAI.md").write_text(report_md, encoding="utf-8")

# Đóng gói toàn bộ báo cáo, bảng số liệu & biểu đồ (không chứa .pth nặng) thành file zip để tải về / push GitHub
bundle_zip = DRIVE_WORKDIR / "week4_hai_reports_and_figures.zip"
subprocess.check_call(["zip", "-q", "-r", str(bundle_zip), "reports/tuan4", "image/tuan4"], cwd=PROJECT_ROOT)

print(f"✅ Đã xuất báo cáo: {report_file}")
print(f"✅ Đã đóng gói toàn bộ báo cáo + biểu đồ tại: {bundle_zip}")
print("\\n📌 Bước cuối cùng: Bạn có thể chạy git add reports/tuan4/ image/tuan4/ notebooks/ -> commit -> push lên nhánh Nguyen_Thanh_Hai_tuan4!")""")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    return nb


def main():
    out = Path("notebooks/Week4_MedSAM_LoRA_BRISC2025_Hai_Colab.ipynb")
    nb = make_notebook()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã tạo {out} ({len(nb['cells'])} cells)")

    # Kiểm tra cú pháp Python của toàn bộ code cells bằng ast.parse (không cho phép SyntaxWarning)
    import ast
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                ast.parse(src)
    print("✅ Đã kiểm tra ast.parse toàn bộ code cells: 0 lỗi cú pháp & 0 cảnh báo!")


if __name__ == "__main__":
    main()
