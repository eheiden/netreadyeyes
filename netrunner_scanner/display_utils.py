
import cv2
import numpy as np

_last_view = {
    "scale": 1.0,
    "x": 0,
    "y": 0,
    "source_w": 1,
    "source_h": 1,
    "target_w": 1,
    "target_h": 1,
}


def fit_image_to_window(window_name, image):
    try:
        _x, _y, target_w, target_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return image

    if target_w <= 0 or target_h <= 0:
        return image

    src_h, src_w = image.shape[:2]

    if src_w <= 0 or src_h <= 0:
        return image

    scale = min(target_w / src_w, target_h / src_h)

    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    _last_view.update({
        "scale": scale,
        "x": x,
        "y": y,
        "source_w": src_w,
        "source_h": src_h,
        "target_w": target_w,
        "target_h": target_h,
    })

    return canvas


def display_to_source(x, y):
    scale = _last_view["scale"]

    if scale <= 0:
        return int(x), int(y)

    source_x = int((x - _last_view["x"]) / scale)
    source_y = int((y - _last_view["y"]) / scale)

    source_x = max(0, min(source_x, _last_view["source_w"] - 1))
    source_y = max(0, min(source_y, _last_view["source_h"] - 1))

    return source_x, source_y
