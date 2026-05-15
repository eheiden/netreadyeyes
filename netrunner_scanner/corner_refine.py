import cv2
import numpy as np
import collector_vision as cvg

from .config import (
    CORNER_REFINER_PADDING_RATIO,
    CORNER_REFINER_MIN_SHARPNESS,
    CORNER_REFINER_MIN_IOU_WITH_PROPOSAL,
    CORNER_REFINER_MIN_AREA_RATIO,
    CORNER_REFINER_MAX_AREA_RATIO,
    CORNER_REFINER_MIN_EDGE_LENGTH,
    CORNER_REFINER_MIN_ASPECT,
    CORNER_REFINER_MAX_ASPECT,
    CORNER_REFINER_FALLBACK_TO_OPENCV,
    MANUAL_SCAN_BYPASS_GEOMETRY_REJECTION,
)

_detector = None


def get_corner_detector():
    global _detector
    if _detector is None:
        print("Loading CollectorVision NeuralCornerDetector...")
        _detector = cvg.NeuralCornerDetector()
    return _detector


def clamp(value, low, high):
    return max(low, min(value, high))


def expanded_candidate_crop(frame, candidate, padding_ratio=CORNER_REFINER_PADDING_RATIO):
    box = candidate["box"]
    x, y, w, h = cv2.boundingRect(box)

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    frame_h, frame_w = frame.shape[:2]

    x1 = clamp(x - pad_x, 0, frame_w - 1)
    y1 = clamp(y - pad_y, 0, frame_h - 1)
    x2 = clamp(x + w + pad_x, 0, frame_w)
    y2 = clamp(y + h + pad_y, 0, frame_h)

    return frame[y1:y2, x1:x2], (x1, y1)


def detection_corners_to_frame_box(detection, crop_origin, crop_shape):
    if detection.corners is None:
        return None

    ox, oy = crop_origin
    ch, cw = crop_shape[:2]

    corners = detection.corners * np.array([cw, ch], dtype=np.float32)
    corners[:, 0] += ox
    corners[:, 1] += oy

    return corners.astype(np.intp)


def polygon_area(points):
    if points is None:
        return 0.0
    return abs(cv2.contourArea(points.astype(np.float32)))


def bounding_rect_iou(a_box, b_box):
    ax, ay, aw, ah = cv2.boundingRect(a_box)
    bx, by, bw, bh = cv2.boundingRect(b_box)

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0 else 0.0


def min_edge_length(box):
    if box is None or len(box) != 4:
        return 0.0

    pts = box.astype(np.float32)
    return min(
        float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
        for i in range(4)
    )


def refined_aspect_ratio(box):
    x, y, w, h = cv2.boundingRect(box)
    if w <= 0 or h <= 0:
        return 0.0
    return max(w, h) / min(w, h)


def fallback_result(reason, sharpness=None, confidence=None):
    if not CORNER_REFINER_FALLBACK_TO_OPENCV:
        return None
    return {
        "fallback_to_opencv": True,
        "reason": reason,
        "sharpness": sharpness,
        "confidence": confidence,
        "refined_box": None,
    }


def validate_refined_box(candidate, refined_box):
    if refined_box is None:
        return False, "no_refined_box"

    if len(refined_box) != 4:
        return False, "not_four_corners"

    if not cv2.isContourConvex(refined_box):
        return False, "not_convex"

    proposal_box = candidate["box"]

    proposal_area = polygon_area(proposal_box)
    refined_area = polygon_area(refined_box)

    if proposal_area <= 0 or refined_area <= 0:
        return False, "bad_area"

    area_ratio = refined_area / proposal_area

    if area_ratio < CORNER_REFINER_MIN_AREA_RATIO:
        return False, f"area_ratio_low:{area_ratio:.2f}"

    if area_ratio > CORNER_REFINER_MAX_AREA_RATIO:
        return False, f"area_ratio_high:{area_ratio:.2f}"

    overlap = bounding_rect_iou(proposal_box, refined_box)

    if overlap < CORNER_REFINER_MIN_IOU_WITH_PROPOSAL:
        return False, f"iou_low:{overlap:.2f}"

    edge_len = min_edge_length(refined_box)

    if edge_len < CORNER_REFINER_MIN_EDGE_LENGTH:
        return False, f"edge_short:{edge_len:.1f}"

    aspect = refined_aspect_ratio(refined_box)

    if not (CORNER_REFINER_MIN_ASPECT <= aspect <= CORNER_REFINER_MAX_ASPECT):
        return False, f"aspect_bad:{aspect:.2f}"

    return True, "ok"


def dewarp_candidate_with_collectorvision(frame, candidate):
    crop, origin = expanded_candidate_crop(frame, candidate)

    if crop is None or crop.size == 0:
        return fallback_result("empty_crop")

    detector = get_corner_detector()
    detection = detector.detect(crop)

    sharpness = float(detection.sharpness or 0.0)
    confidence = float(detection.confidence)

    if not detection.card_present:
        return fallback_result("not_card_present", sharpness, confidence)

    if sharpness < CORNER_REFINER_MIN_SHARPNESS:
        return fallback_result(f"sharpness_low:{sharpness:.3f}", sharpness, confidence)

    refined_box = detection_corners_to_frame_box(
        detection=detection,
        crop_origin=origin,
        crop_shape=crop.shape,
    )

    valid, reason = validate_refined_box(candidate, refined_box)

    if not valid:
        manual_source = str(candidate.get("source", "")).startswith("manual")
        if not (MANUAL_SCAN_BYPASS_GEOMETRY_REJECTION and manual_source and refined_box is not None):
            return fallback_result(reason, sharpness, confidence)

    try:
        dewarped = detection.dewarp(crop)
    except Exception as e:
        return fallback_result(f"dewarp_failed:{e}", sharpness, confidence)

    return {
        "image": dewarped,
        "refined_box": refined_box,
        "sharpness": sharpness,
        "confidence": confidence,
        "fallback_to_opencv": False,
        "reason": "refined",
    }
