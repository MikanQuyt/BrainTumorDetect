"""Cắt sát hộp sọ (Extreme-Point Skull Cropping, OpenCV), CLAHE và trích hộp chuẩn từ mask."""
import cv2
import numpy as np


def read_image(path, flag=cv2.IMREAD_COLOR):
    """Đọc ảnh bằng imdecode để hỗ trợ đường dẫn Unicode trên Windows; trả None nếu hỏng."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flag) if buf.size else None


def write_png(path, image):
    ok, enc = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Không mã hóa được PNG: {path}")
    enc.tofile(str(path))


def binarize_mask(raw, thresh=127):
    """Nhị phân hóa mask xám về {0, 1}.

    Mask của DS2/DS3 không sạch (nền ~0-7, vùng u ~248-255 do nén ảnh), nên KHÔNG được dùng `> 0`:
    cách đó biến cả ảnh thành "u" và hộp bao thành toàn khung hình.
    """
    return (np.asarray(raw) > thresh).astype(np.uint8)


def mask_to_box(mask):
    """Hộp [x_min, y_min, x_max, y_max] bao sát vùng mask > 0.

    x_max/y_max là biên LOẠI TRỪ (= chỉ số điểm cuối + 1), khớp định dạng pascal_voc của Albumentations.
    Trả None nếu mask rỗng.
    """
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def skull_crop_box(image, thresh=45, iterations=2, min_area_frac=0.2):
    """Tìm khung cắt từ 4 điểm cực trị (Tây, Đông, Bắc, Nam) của đường bao lớn nhất.

    Gaussian blur (5x5) -> ngưỡng nhị phân -> erode/dilate (khử nhiễu rời rạc) -> findContours.
    Trả về ((x0, y0, x1, y1), fallback). Khung có biên x1/y1 loại trừ. fallback=True nghĩa là không
    tìm được đường bao hợp lý (hoặc khung quá nhỏ, < min_area_frac diện tích ảnh) nên dùng toàn ảnh.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.erode(binary, None, iterations=iterations)
    binary = cv2.dilate(binary, None, iterations=iterations)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h), True

    c = max(contours, key=cv2.contourArea)
    xmin, xmax = int(c[:, :, 0].min()), int(c[:, :, 0].max())   # cực Tây, cực Đông
    ymin, ymax = int(c[:, :, 1].min()), int(c[:, :, 1].max())   # cực Bắc, cực Nam
    box = (xmin, ymin, xmax + 1, ymax + 1)
    if (box[2] - box[0]) * (box[3] - box[1]) < min_area_frac * w * h:
        return (0, 0, w, h), True
    return box, False


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE áp lên từng kênh của ảnh uint8 (ảnh xám hoặc 3 kênh)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    return np.stack([clahe.apply(image[..., i]) for i in range(image.shape[2])], axis=-1)


def crop_with_mask(image, mask=None, **kwargs):
    """Cắt sát hộp sọ cho cả ảnh và mask (nếu có) theo cùng một khung [y0:y1, x0:x1].

    Nếu khung cắt làm mất một phần mask (ngưỡng sai trên ảnh tối), khung được mở rộng để bao trọn mask.
    Trả về (image_crop, mask_crop, crop_box, info) với info = {"fallback_full": bool, "expanded_for_mask": bool}.
    """
    (x0, y0, x1, y1), fallback = skull_crop_box(image, **kwargs)
    expanded = False
    if mask is not None:
        mb = mask_to_box(mask)
        if mb is not None and (mb[0] < x0 or mb[1] < y0 or mb[2] > x1 or mb[3] > y1):
            x0, y0 = min(x0, mb[0]), min(y0, mb[1])
            x1, y1 = max(x1, mb[2]), max(y1, mb[3])
            expanded = True
    img_c = image[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1] if mask is not None else None
    return img_c, mask_c, (x0, y0, x1, y1), {"fallback_full": fallback, "expanded_for_mask": expanded}
