import cv2
import numpy as np
from PIL import Image

from .config import (
    CONFIDENCE_THRESHOLD,
    AUTO_SEND_TO_OBS,
    DEBUG_SAVE_CROPS,
    DEBUG_CROPS_DIR,
    TRACK_IOU_THRESHOLD,
    STATIONARY_CENTER_THRESHOLD_PX,
    STATIONARY_REFRESH_SECONDS,
    TRACK_EXPIRE_SECONDS,
    OBS_QUEUE_ENABLED,
    USE_COLLECTORVISION_CORNER_REFINER,
    AMBIGUOUS_MATCH_SCORE_THRESHOLD,
    AMBIGUOUS_MATCH_MIN_MARGIN,
    CARD_BACK_EDGE_RATIO_THRESHOLD,
    CARD_BACK_SAT_STD_THRESHOLD,
    CARD_BACK_VAL_STD_THRESHOLD,
    CARD_BACK_CENTER_EDGE_RATIO_THRESHOLD,
    CARD_BACK_CENTER_SAT_STD_THRESHOLD,
    CARD_BACK_CENTER_VAL_STD_THRESHOLD,
    REFINED_BOX_SMOOTHING_ALPHA,
)
from .crop import crop_candidate
from .detection import find_card_candidates
from .obs_bridge import send_match_to_obs
from .tracking import CardTracker
from .obs_queue import ObsFifoQueue
from .corner_refine import dewarp_candidate_with_collectorvision

latest_matches = {"left": [], "right": []}

tracker = CardTracker(
    iou_threshold=TRACK_IOU_THRESHOLD,
    stationary_center_threshold_px=STATIONARY_CENTER_THRESHOLD_PX,
    stationary_refresh_seconds=STATIONARY_REFRESH_SECONDS,
    expire_seconds=TRACK_EXPIRE_SECONDS,
    refined_box_smoothing_alpha=REFINED_BOX_SMOOTHING_ALPHA,
)

obs_queue = ObsFifoQueue()


def legacy_opencv_crop_to_pil(frame, candidate):
    crop = crop_candidate(frame, candidate)

    if crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def analyze_crop_detail(pil_image):
    img = np.array(pil_image.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = np.count_nonzero(edges) / edges.size

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, w = gray.shape[:2]
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    center_gray = gray[y1:y2, x1:x2]
    center_hsv = hsv[y1:y2, x1:x2]
    center_edges = cv2.Canny(center_gray, 50, 150)

    center_edge_ratio = np.count_nonzero(center_edges) / center_edges.size

    return {
        "edge_ratio": float(edge_ratio),
        "sat_std": float(hsv[:, :, 1].std()),
        "val_std": float(hsv[:, :, 2].std()),
        "center_edge_ratio": float(center_edge_ratio),
        "center_sat_std": float(center_hsv[:, :, 1].std()),
        "center_val_std": float(center_hsv[:, :, 2].std()),
    }


def looks_like_card_back(detail):
    # Sleeve color changes often. Facedown cards are usually smoother in the center than real card faces.
    center_low_detail = (
        detail["center_edge_ratio"] < CARD_BACK_CENTER_EDGE_RATIO_THRESHOLD
        and (
            detail["center_sat_std"] < CARD_BACK_CENTER_SAT_STD_THRESHOLD
            or detail["center_val_std"] < CARD_BACK_CENTER_VAL_STD_THRESHOLD
        )
    )

    overall_low_detail = (
        detail["edge_ratio"] < CARD_BACK_EDGE_RATIO_THRESHOLD
        and (
            detail["sat_std"] < CARD_BACK_SAT_STD_THRESHOLD
            or detail["val_std"] < CARD_BACK_VAL_STD_THRESHOLD
        )
    )

    return center_low_detail or overall_low_detail



def get_candidate_pil_image(frame, candidate):
    if USE_COLLECTORVISION_CORNER_REFINER:
        refined = dewarp_candidate_with_collectorvision(frame, candidate)

        if refined is not None and not refined.get("fallback_to_opencv"):
            refined["proposal_box"] = candidate["box"]
            return refined

        fallback_image = legacy_opencv_crop_to_pil(frame, candidate)

        if fallback_image is None:
            return None

        reason = refined.get("reason", "refiner_none") if refined else "refiner_none"

        return {
            "image": fallback_image,
            "refined_box": None,
            "proposal_box": candidate["box"],
            "sharpness": refined.get("sharpness") if refined else None,
            "confidence": refined.get("confidence") if refined else None,
            "fallback_to_opencv": True,
            "reason": reason,
        }

    fallback_image = legacy_opencv_crop_to_pil(frame, candidate)

    if fallback_image is None:
        return None

    return {
        "image": fallback_image,
        "refined_box": None,
        "proposal_box": candidate["box"],
        "sharpness": None,
        "confidence": None,
        "fallback_to_opencv": True,
        "reason": "refiner_disabled",
    }


def card_back_result(refined, detail):
    return {
        "id": "card_back",
        "score": 1.0,
        "margin": None,
        "rotation": "?",
        "refined_box": refined.get("refined_box"),
        "proposal_box": refined.get("proposal_box"),
        "sharpness": refined.get("sharpness"),
        "confidence": refined.get("confidence"),
        "used_fallback": refined.get("fallback_to_opencv", False),
        "refine_reason": "low_detail_card_back",
        "detail_metrics": detail,
    }


def recognize_candidate_crop(frame, candidate, side, candidate_index, catalog):
    refined = get_candidate_pil_image(frame, candidate)

    if refined is None:
        return None

    base_image = refined["image"]

    detail = analyze_crop_detail(base_image)

    if candidate.get("force_card_back"):
        return card_back_result(refined, detail)

    if looks_like_card_back(detail):
        return card_back_result(refined, detail)

    if DEBUG_SAVE_CROPS:
        suffix = "opencv" if refined.get("fallback_to_opencv") else "cvrefined"
        debug_path = DEBUG_CROPS_DIR / f"{side}_candidate_{candidate_index}_{suffix}.jpg"
        base_image.save(debug_path)

    rotation_tests = [
        ("0", base_image),
        ("90", base_image.rotate(90, expand=True)),
        ("180", base_image.rotate(180, expand=True)),
        ("270", base_image.rotate(270, expand=True)),
    ]

    all_results = []

    for rotation_name, pil_image in rotation_tests:
        for result in catalog.search_image(pil_image):
            result["rotation"] = rotation_name
            result["refined_box"] = refined.get("refined_box")
            result["proposal_box"] = refined.get("proposal_box")
            result["sharpness"] = refined.get("sharpness")
            result["confidence"] = refined.get("confidence")
            result["used_fallback"] = refined.get("fallback_to_opencv", False)
            result["refine_reason"] = refined.get("reason")
            result["detail_metrics"] = detail
            all_results.append(result)

    all_results.sort(key=lambda r: r["score"], reverse=True)

    if not all_results:
        return None

    best = all_results[0]

    if len(all_results) > 1:
        second = all_results[1]
        margin = best["score"] - second["score"]
        best["margin"] = margin

        if best["score"] < AMBIGUOUS_MATCH_SCORE_THRESHOLD and margin < AMBIGUOUS_MATCH_MIN_MARGIN:
            best["id"] = "unknown"
            best["refine_reason"] = f"ambiguous_margin:{margin:.3f}"
    else:
        best["margin"] = None

    return best


def scan_side_for_matches(frame, roi, side, catalog):
    candidates = find_card_candidates(frame, roi)

    for i, candidate in enumerate(candidates, start=1):
        track, needs_processing = tracker.update_or_create_track(
            side=side,
            box=candidate["box"],
        )

        if needs_processing:
            best = recognize_candidate_crop(
                frame=frame,
                candidate=candidate,
                side=side,
                candidate_index=i,
                catalog=catalog,
            )

            previous_label = track.get("label")
            tracker.apply_recognition_result(track, best)

            current_label = track.get("label")
            current_score = float(track.get("score", 0.0))

            if (
                current_label
                and current_label not in ("unknown", "card_back")
                and current_score >= CONFIDENCE_THRESHOLD
                and (
                    current_label != previous_label
                    or not track.get("displayed", False)
                )
                and not track.get("queued", False)
            ):
                if OBS_QUEUE_ENABLED:
                    obs_queue.enqueue_match(
                        side=side,
                        card_id=current_label,
                        score=current_score,
                        track=track,
                    )
                elif AUTO_SEND_TO_OBS:
                    send_match_to_obs(side, current_label)

    latest_matches[side] = tracker.get_visible_matches(side)

    if OBS_QUEUE_ENABLED:
        obs_queue.tick()


def tick_obs_queue():
    if OBS_QUEUE_ENABLED:
        obs_queue.tick()
