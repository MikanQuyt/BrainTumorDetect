"""
Global configuration for BRISC 2025 experiments.
Covers:
  1. ViT Baseline Classification  (Linear Probing & Full Fine-tuning)
  2. Zero-shot MedSAM segmentation (Oracle Box prompt)
  3. SAM Everything Mode on 50 images (25 BRISC 2025 + 25 Dataset 1 Navoneel)
"""

import os
from pathlib import Path

# ─────────────────────────── Paths ───────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT   = PROJECT_ROOT / "dataset" / "brisc2025" / "brisc2025"
DATASET1_ROOT  = PROJECT_ROOT / "dataset" / "navoneel"

# Classification task
CLS_TRAIN_DIR  = ARCHIVE_ROOT / "classification_task" / "train"
CLS_TEST_DIR   = ARCHIVE_ROOT / "classification_task" / "test"

# Segmentation task
SEG_TRAIN_IMG  = ARCHIVE_ROOT / "segmentation_task" / "train" / "images"
SEG_TRAIN_MASK = ARCHIVE_ROOT / "segmentation_task" / "train" / "masks"
SEG_TEST_IMG   = ARCHIVE_ROOT / "segmentation_task" / "test"  / "images"
SEG_TEST_MASK  = ARCHIVE_ROOT / "segmentation_task" / "test"  / "masks"

# Output dirs
RESULTS_ROOT   = PROJECT_ROOT / "reports" / "tuan2" / "brisc2025"
CLS_RESULTS    = RESULTS_ROOT / "classification"
SEG_RESULTS    = RESULTS_ROOT / "segmentation"
SAM_RESULTS    = RESULTS_ROOT / "sam_everything"
CKPT_DIR       = PROJECT_ROOT / "checkpoints"

for d in [CLS_RESULTS, SEG_RESULTS, SAM_RESULTS, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Classes ─────────────────────────────────────────
CLASS_NAMES    = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES    = 4

# ─────────────────────────── ViT Classification ──────────────────────────────
VIT_MODEL_NAME = "vit_tiny_patch16_224"   # 12-layer Vision Transformer (CPU efficient)
IMG_SIZE       = 224
BATCH_SIZE     = 32
NUM_WORKERS    = 0          # Windows-safe (no fork)
EPOCHS_LP      = 10         # Linear Probing epochs
EPOCHS_FT      = 5          # Full Fine-tuning epochs
LR_LP          = 1e-3       # LR for linear probing head
LR_FT          = 5e-5       # LR for full fine-tuning
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.1        # fraction of train used as validation

# Data augmentation flags
USE_AUGMENT    = True

# ─────────────────────────── MedSAM ──────────────────────────────────────────
MEDSAM_CKPT    = CKPT_DIR / "medsam_vit_b.pth"
MEDSAM_HF_ID   = "wanglab/medsam-vit-base"   # HuggingFace Hub id (fallback)
BOX_PADDING    = 0           # pixels to expand GT bbox (0 = tight oracle box)
SAM_IMG_SIZE   = 1024        # MedSAM input resolution

# Subset to evaluate segmentation on (for speed); None = full test set
SEG_EVAL_LIMIT = None        # set to e.g. 200 for a quick run

# ─────────────────────────── SAM Everything Mode ─────────────────────────────
SAM_VIT_B_CKPT = CKPT_DIR / "sam_vit_b_01ec64.pth"
SAM_HF_B_ID    = "facebook/sam-vit-base"
N_EVERYTHING   = 50          # number of images to process in everything mode

# SAM automatic mask generator params
AMG_POINTS_PER_SIDE     = 32
AMG_PRED_IOU_THRESH     = 0.86
AMG_STABILITY_THRESH    = 0.92
AMG_CROP_N_LAYERS       = 1
AMG_CROP_N_POINTS_DOWN  = 2
AMG_MIN_MASK_REGION_AREA= 100

# ─────────────────────────── Misc ────────────────────────────────────────────
SEED            = 42
DEVICE_PREF     = "cpu"      # "cuda" if GPU available, else "cpu"
