import cv2
import numpy as np

def find_card_candidates(frame, roi):
    x, y, w, h = roi
    roi_img = frame[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    roi_area = w * h

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < roi_area * 0.003:
            continue

        if area > roi_area * 0.20:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect

        if rw <= 0 or rh <= 0:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect = long_side / short_side

        if not (1.15 <= aspect <= 1.85):
            continue

        box = cv2.boxPoints(rect)
        box = np.intp(box)

        box[:, 0] += x
        box[:, 1] += y

        margin = 8
        box_x, box_y, box_w, box_h = cv2.boundingRect(box)

        if (
            box_x <= x + margin or
            box_y <= y + margin or
            box_x + box_w >= x + w - margin or
            box_y + box_h >= y + h - margin
        ):
            continue

        candidates.append({
            "box": box,
            "area": area,
            "aspect": aspect,
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)

    return candidates
