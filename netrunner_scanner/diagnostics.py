
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CARD_DIAGNOSTICS_DIR,
    CARD_DIAGNOSTICS_ENABLED,
    CARD_DIAGNOSTICS_TARGET,
    CARD_DIAGNOSTICS_SAVE_ALL_MANUAL_SCANS,
    CARD_DIAGNOSTICS_TOP_N,
)


def normalize_id(value):
    return str(value or "").lower().replace(" ", "_")


def should_diagnose(card_id=None, source=None, forced=False):
    if forced:
        return True

    if not CARD_DIAGNOSTICS_ENABLED:
        return False

    target = normalize_id(CARD_DIAGNOSTICS_TARGET)

    if not target:
        return bool(CARD_DIAGNOSTICS_SAVE_ALL_MANUAL_SCANS and str(source or "").startswith("manual"))

    return target in normalize_id(card_id)


def diagnostic_folder(label):
    safe = normalize_id(label) or "unknown"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(CARD_DIAGNOSTICS_DIR) / safe / stamp
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_candidate_diagnostics(
    *,
    frame,
    candidate,
    side,
    candidate_index,
    refined,
    detail,
    results,
    best,
    label=None,
    forced=False,
):
    source = candidate.get("source")
    result_id = best.get("id") if best else None
    target_label = label or result_id or CARD_DIAGNOSTICS_TARGET or source or "unknown"

    if not should_diagnose(result_id, source=source, forced=forced):
        return None

    folder = diagnostic_folder(target_label)

    raw_path = folder / "01_frame_candidate.jpg"
    crop_path = folder / "02_raw_crop.jpg"
    refined_path = folder / "03_refined_or_fallback.jpg"
    overlay_path = folder / "04_overlay.jpg"
    report_path = folder / "report.json"

    overlay = frame.copy()

    box = candidate.get("box")
    if box is not None:
        cv2.polylines(overlay, [box.astype(np.intp)], True, (0, 255, 255), 2)

    refined_box = None
    if refined:
        refined_box = refined.get("refined_box") or refined.get("proposal_box")
        if refined_box is not None:
            cv2.polylines(overlay, [refined_box.astype(np.intp)], True, (255, 255, 255), 2)

    cv2.imwrite(str(overlay_path), overlay)

    if box is not None:
        x, y, w, h = cv2.boundingRect(box)
        x = max(0, x)
        y = max(0, y)
        crop = frame[y:y + h, x:x + w]
        if crop.size:
            cv2.imwrite(str(crop_path), crop)

    cv2.imwrite(str(raw_path), frame)

    if refined and refined.get("image") is not None:
        refined["image"].save(refined_path)

    top = []
    for result in (results or [])[:CARD_DIAGNOSTICS_TOP_N]:
        top.append({
            "id": str(result.get("id")),
            "score": float(result.get("score", 0.0)),
            "rotation": result.get("rotation"),
            "refine_reason": result.get("refine_reason"),
        })

    report = {
        "side": side,
        "candidate_index": candidate_index,
        "candidate_source": source,
        "candidate_area": float(candidate.get("area", 0.0)),
        "candidate_box": box.tolist() if box is not None else None,
        "refined_box": refined_box.tolist() if refined_box is not None else None,
        "refine_reason": refined.get("reason") if refined else None,
        "sharpness": refined.get("sharpness") if refined else None,
        "confidence": refined.get("confidence") if refined else None,
        "used_fallback": bool(refined.get("fallback_to_opencv", False)) if refined else None,
        "detail": detail,
        "best": {
            "id": best.get("id") if best else None,
            "score": float(best.get("score", 0.0)) if best else 0.0,
            "margin": best.get("margin") if best else None,
            "refine_reason": best.get("refine_reason") if best else None,
        },
        "top": top,
        "files": {
            "frame": str(raw_path),
            "crop": str(crop_path),
            "refined": str(refined_path),
            "overlay": str(overlay_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return folder
