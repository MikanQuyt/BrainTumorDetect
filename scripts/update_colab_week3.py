import json
from pathlib import Path

nb_path = Path("notebooks/week3_vit_lora_brisc2025_colab.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1: Update clone command to Nguyen_Thanh_Hai_tuan3 or dev
cell1_source = [
    "import os\n",
    "%cd /content\n",
    "!rm -rf BrainTumorDetect\n",
    "\n",
    "# Clone trực tiếp nhánh Tuần 3 đã tích hợp đầy đủ tính năng\n",
    "!git clone -b Nguyen_Thanh_Hai_tuan3 https://github.com/MikanQuyt/BrainTumorDetect.git\n",
    "%cd BrainTumorDetect\n",
    "\n",
    "# Gỡ torchao cũ của Colab & cài thư viện cần thiết\n",
    "!pip uninstall -y torchao\n",
    "!pip install -q timm transformers peft albumentations scikit-learn python-docx kaggle\n"
]
nb["cells"][1]["source"] = cell1_source

# Cell 3: Mount drive & checkpoint folder
cell3_source = [
    "from google.colab import drive\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# 1. Mount Google Drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# 2. Thư mục lưu checkpoint tự động trên Google Drive\n",
    "DRIVE_DIR = Path('/content/drive/MyDrive/BrainTumorDetect_Checkpoints')\n",
    "DRIVE_DIR.mkdir(parents=True, exist_ok=True)\n",
    "print(f'✅ Đã kết nối Google Drive! Thư mục checkpoint: {DRIVE_DIR}')\n",
    "\n",
    "# 3. Nén và lưu 6,000 ảnh đã tiền xử lý lên Drive (chỉ cần chạy 1 lần để dùng mãi mãi)\n",
    "zip_drive = Path('/content/drive/MyDrive/ds3_processed.zip')\n",
    "if not zip_drive.exists() and Path('dataset/ds3_processed').exists():\n",
    "    print('📦 Đang lưu dự phòng dataset lên Google Drive...')\n",
    "    !zip -q -r /content/drive/MyDrive/ds3_processed.zip dataset/ds3_processed\n",
    "    print('✅ Đã lưu xong dữ liệu lên Google Drive!')\n",
    "elif zip_drive.exists():\n",
    "    print('⚡ Đã có sẵn ds3_processed.zip trên Google Drive! Giải nén siêu tốc vào workspace...')\n",
    "    !unzip -q -o /content/drive/MyDrive/ds3_processed.zip -d dataset/\n"
]
nb["cells"][3]["source"] = cell3_source

# Cell 6: 1000 epochs + Early stopping + Scheduler plateau + Drive sync + Resume
cell6_source = [
    "# Cell 6: Huấn luyện ViT-LoRA Đa lớp 1000 Epochs + Early Stopping + Scheduler Plateau + Checkpoint Drive + Resume\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "DRIVE_DIR = Path('/content/drive/MyDrive/BrainTumorDetect_Checkpoints')\n",
    "DRIVE_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "latest_ckpt = DRIVE_DIR / 'loss_latest.pth'\n",
    "resume_opt = f'--resume {latest_ckpt}' if latest_ckpt.exists() else ''\n",
    "\n",
    "if resume_opt:\n",
    "    print(f'🔄 PHÁT HIỆN CHECKPOINT CŨ TRÊN DRIVE: {latest_ckpt}')\n",
    "    print('   -> Tự động nạp lại trọng số và tiếp tục huấn luyện mà không mất tiến trình!')\n",
    "else:\n",
    "    print('🚀 Bắt đầu huấn luyện mới: Tối đa 1000 Epochs, Early Stopping patience=25, Scheduler Plateau để hạ loss xuống 0.0x!')\n",
    "\n",
    "# Chạy huấn luyện: tự động cập nhật loss_best.pth và loss_latest.pth vào Google Drive\n",
    "!python -m engine.train_vit_lora \\\n",
    "    --task multiclass \\\n",
    "    --lora-r 8 \\\n",
    "    --lora-alpha 16 \\\n",
    "    --loss focal \\\n",
    "    --label-smoothing 0.01 \\\n",
    "    --box-loss-weight 0.1 \\\n",
    "    --scheduler plateau \\\n",
    "    --epochs 1000 \\\n",
    "    --patience 25 \\\n",
    "    --early-stop-metric loss \\\n",
    "    --batch-size 32 \\\n",
    "    --drive-dir \"/content/drive/MyDrive/BrainTumorDetect_Checkpoints\" \\\n",
    "    {resume_opt} \\\n",
    "    --out runs/vit_lora_multiclass_r8\n"
]
nb["cells"][6]["source"] = cell6_source

# Add a new cell after Cell 6 to copy files to shared drive if shortcut exists
sync_shared_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cell 6b: Đẩy loss_best.pth và loss_latest.pth vào Thư mục Google Drive được chia sẻ của Trưởng nhóm\n",
        "# Link thư mục chia sẻ: https://drive.google.com/drive/folders/1JUGAxnRcgTf4INQstlUt9YqggO_cbWZ6?usp=sharing\n",
        "# (Để script tìm thấy, bạn chỉ cần mở link trên và bấm 'Thêm lối tắt vào Drive' -> Chọn 'Drive của tôi')\n",
        "import os, shutil\n",
        "from pathlib import Path\n",
        "\n",
        "src_best = Path('/content/drive/MyDrive/BrainTumorDetect_Checkpoints/loss_best.pth')\n",
        "src_latest = Path('/content/drive/MyDrive/BrainTumorDetect_Checkpoints/loss_latest.pth')\n",
        "\n",
        "# Tìm kiếm thư mục chia sẻ trong MyDrive nếu có\n",
        "mydrive = Path('/content/drive/MyDrive')\n",
        "target_dirs = [p for p in mydrive.iterdir() if p.is_dir() and any(k in p.name.lower() for k in ['brain', 'tumor', 'tuan', 'week', 'brisc', 'chkpt'])]\n",
        "\n",
        "print('📂 Các thư mục khả dụng trong Drive của tôi:')\n",
        "for d in target_dirs:\n",
        "    print(f'   - {d.name}')\n",
        "\n",
        "if src_best.exists() and src_latest.exists():\n",
        "    print(f'✅ Checkpoint tốt nhất hiện tại: {src_best} ({src_best.stat().st_size / (1024*1024):.2f} MB)')\n",
        "    print(f'✅ Checkpoint mới nhất hiện tại: {src_latest} ({src_latest.stat().st_size / (1024*1024):.2f} MB)')\n",
        "    print('👉 Bạn cũng có thể mở link Google Drive của Trưởng nhóm và kéo thả trực tiếp 2 file trên vào thư mục!')\n",
        "else:\n",
        "    print('ℹ️ Chưa tìm thấy file checkpoint. Hãy hoàn thành ít nhất 1 epoch ở cell trên.')\n"
    ]
}

# Check if Cell 6b already exists or insert it
has_6b = any("Cell 6b" in "".join(c.get("source", [])) for c in nb["cells"])
if not has_6b:
    nb["cells"].insert(7, sync_shared_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Đã cập nhật thành công {nb_path}!")
