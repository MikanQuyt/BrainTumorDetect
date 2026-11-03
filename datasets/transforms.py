from datasets.libs import *


def extreme_point_skull_crop(image, mask=None, thresh=45, iterations=2):
    """Cắt sát hộp sọ bằng 4 điểm cực trị của đường bao lớn nhất.

    Trả về (image_crop, mask_crop, (xmin, ymin, xmax, ymax)); mask_crop là None nếu không có mask.
    Nếu không tìm thấy đường bao thì giữ nguyên ảnh.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.erode(binary, None, iterations=iterations)
    binary = cv2.dilate(binary, None, iterations=iterations)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    if not contours:
        return image, mask, (0, 0, w, h)

    c = max(contours, key=cv2.contourArea)
    xmin = int(c[:, :, 0].min())  # cực Tây
    xmax = int(c[:, :, 0].max())  # cực Đông
    ymin = int(c[:, :, 1].min())  # cực Bắc
    ymax = int(c[:, :, 1].max())  # cực Nam

    # +1 để bao gồm cả điểm cực biên
    box = (xmin, ymin, xmax + 1, ymax + 1)
    image = image[ymin:ymax + 1, xmin:xmax + 1]
    if mask is not None:
        mask = mask[ymin:ymax + 1, xmin:xmax + 1]
    return image, mask, box


def apply_clahe_zscore(image, clip_limit=2.0, tile_grid_size=(8, 8), eps=1e-8):
    """CLAHE trên từng kênh (uint8) rồi chuẩn hóa Z-score: (I - mean) / (std + eps)."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        out = clahe.apply(image)
    else:
        out = np.stack([clahe.apply(image[..., i]) for i in range(image.shape[2])], axis=-1)

    out = out.astype(np.float32)
    return (out - out.mean()) / (out.std() + eps)


def preprocess_mri(image, mask=None):
    """Cắt hộp sọ (ảnh + mask) -> CLAHE -> Z-score. Trả về (image float32, mask, box)."""
    image, mask, box = extreme_point_skull_crop(image, mask)
    return apply_clahe_zscore(image), mask, box

