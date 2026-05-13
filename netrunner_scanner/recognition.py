import cv2
from PIL import Image

from .config import (
    CONFIDENCE_THRESHOLD,
    AUTO_SEND_TO_OBS,
    DEBUG_SAVE_CROPS,
    DEBUG_CROPS_DIR,
)
from .crop import crop_candidate
from .detection import find_card_candidates
from .obs_bridge import send_match_to_obs

latest_matches = {
    "left": [],
    "right": [],
}

def recognize_candidate_crop(frame, candidate, side, candidate_index, catalog):
    crop = crop_candidate(frame, candidate)

    if crop is None or crop.size == 0:
        return None

    if DEBUG_SAVE_CROPS:
        debug_path = DEBUG_CROPS_DIR / f"{side}_candidate_{candidate_index}.jpg"
        cv2.imwrite(str(debug_path), crop)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    base_image = Image.fromarray(crop_rgb)

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
            all_results.append(result)

    all_results.sort(key=lambda r: r["score"], reverse=True)

    if not all_results:
        return None

    return all_results[0]

def scan_side_for_matches(frame, roi, side, catalog):
    candidates = find_card_candidates(frame, roi)

    matches = []

    for i, candidate in enumerate(candidates, start=1):
        best = recognize_candidate_crop(
            frame=frame,
            candidate=candidate,
            side=side,
            candidate_index=i,
            catalog=catalog,
        )

        if best is None:
            label = "unknown"
            score = 0.0
            rotation = "?"
        else:
            label = best["id"]
            score = best["score"]
            rotation = best["rotation"]

        matches.append({
            "box": candidate["box"],
            "label": label,
            "score": score,
            "rotation": rotation,
        })

    latest_matches[side] = matches

    confident_matches = [
        m for m in matches
        if m["score"] >= CONFIDENCE_THRESHOLD
    ]

    if AUTO_SEND_TO_OBS and confident_matches:
        best_match = max(confident_matches, key=lambda m: m["score"])
        send_match_to_obs(side, best_match["label"])
