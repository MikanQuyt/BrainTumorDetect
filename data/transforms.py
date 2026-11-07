"""Pipeline tăng cường dữ liệu Albumentations: ảnh, mask và bounding box biến đổi đồng bộ."""
import albumentations as A
import cv2

BBOX_PARAMS = dict(format="pascal_voc", label_fields=["class_labels"])


def build_train_transform(rotate_limit=15, min_visibility=0.3):
    """HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast.

    Mask được xử lý nội suy nearest; box bị cắt mất quá nhiều (min_visibility) sẽ bị loại,
    box còn lại được clip vào trong ảnh.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ],
        bbox_params=A.BboxParams(min_visibility=min_visibility, clip=True, **BBOX_PARAMS),
    )


def build_eval_transform():
    """Không tăng cường; vẫn đi qua Compose để định dạng đầu ra đồng nhất với lúc train."""
    return A.Compose([A.NoOp()], bbox_params=A.BboxParams(**BBOX_PARAMS))

