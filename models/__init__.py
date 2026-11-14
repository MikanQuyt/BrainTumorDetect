"""Models package: ViT-Base/16 baseline, ViT-LoRA + ViTDetBoxHead, AutoPromptGenerator và wrappers SAM / MedSAM."""
from models.auto_prompt import (
    AutoPromptGenerator,
    compute_attention_rollout,
    compute_layercam,
    extract_brain_parenchyma_mask,
    fuse_rollout_and_layercam,
    heatmap_to_box_and_points,
)
from models.sam_wrappers import MedSamBoxSegmenter, SamBoxSegmenter, load_segmenter
from models.vit_baseline import MODEL_NAME, MODES, ViTBaseline, build_vit_baseline
from models.vit_lora import ViTDetBoxHead, ViTLoRA, build_vit_lora, pad_normalized_boxes

__all__ = [
    "MODEL_NAME",
    "MODES",
    "ViTBaseline",
    "build_vit_baseline",
    "ViTDetBoxHead",
    "ViTLoRA",
    "build_vit_lora",
    "pad_normalized_boxes",
    "AutoPromptGenerator",
    "compute_attention_rollout",
    "compute_layercam",
    "extract_brain_parenchyma_mask",
    "fuse_rollout_and_layercam",
    "heatmap_to_box_and_points",
    "SamBoxSegmenter",
    "MedSamBoxSegmenter",
    "load_segmenter",
]
