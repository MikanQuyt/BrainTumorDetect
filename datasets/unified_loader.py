import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import json
import numpy as np

class UnifiedBrainMRIDataset(Dataset):
    def __init__(self, json_annotation_files, transform=None):
        # Đọc trực tiếp các file JSON chú thích của cả 4 bộ dữ liệu
        self.data_records = []
        for json_file in json_annotation_files:
            with open(json_file, 'r') as f:
                self.data_records.extend(json.load(f))
        
        # Cấu hình pipeline Albumentations bảo toàn tọa độ Box và Mask
        self.transform = transform or A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        
        # (Giả định hàm đọc ảnh và mặt nạ đã được viết ở đây)
        image = cv2.imread(record['img_path']) 
        mask = cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE) if record['has_mask'] else np.zeros(image.shape[:2], dtype=np.uint8)
        
        gt_box = record['gt_box'] # [xmin, ymin, xmax, ymax]
        sub_class_label = record['sub_class_label']

        # Transform đồng bộ khi xoay hoặc lật ảnh
        if self.transform:
            transformed = self.transform(
                image=image, 
                mask=mask, 
                bboxes=[gt_box], 
                class_labels=[sub_class_label]
            )
            image = transformed['image']
            mask = transformed['mask']
            gt_box = transformed['bboxes'][0] if transformed['bboxes'] else gt_box

        # Trả về từ điển đồng nhất cho cả 2 nhánh ViT và MedSAM
        return {
            'image_224': cv2.resize(image, (224, 224)),
            'image_1024': cv2.resize(image, (1024, 1024)),
            'binary_label': torch.tensor(record['has_tumor'], dtype=torch.long),
            'sub_class_label': torch.tensor(sub_class_label, dtype=torch.long),
            'gt_box': torch.tensor(gt_box, dtype=torch.float32),
            'gt_mask': torch.tensor(mask, dtype=torch.long),
            'has_mask': torch.tensor(record['has_mask'], dtype=torch.bool)
        }