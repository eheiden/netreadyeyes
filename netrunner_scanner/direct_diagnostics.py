"""Live, directed CollectorVision diagnostics.

This module is deliberately separate from normal recognition/tracking.  It lets
us ask one narrow question: "when CollectorVision is given exactly this patch of
the camera frame, what corners and card match does it see?"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .corner_refine import dewarp_manual_region_with_collectorvision
from .recognition import _recognize_from_refined


OUT_DIR = Path("debug") / "direct_collectorvision_diagnostics"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _pil_to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    return Image.fromarray(rgb)


def _result_summary(best: Optional[Dict[str, Any]], reason: str, refined: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if best is None:
        return {
            "id": None,
            "score": 0.0,
            "margin": None,
            "reason": reason,
            "alternatives": [],
            "sharpness": refined.get("sharpness") if refined else None,
            "confidence": refined.get("confidence") if refined else None,
        }
    return {
        "id": best.get("id"),
        "score": _safe_float(best.get("score")),
        "margin": best.get("margin"),
        "rotation": best.get("rotation"),
        "reason": best.get("refine_reason") or best.get("manual_direct_reason") or reason,
        "sharpness": best.get("sharpness", refined.get("sharpness") if refined else None),
        "confidence": best.get("confidence", refined.get("confidence") if refined else None),
        "alternatives": best.get("alternatives") or [],
    }


def _find_catalog_image(card_id: Optional[str]) -> Optional[str]:
    if not card_id or card_id in ("unknown", "card_back"):
        return None

    candidates = []
    cid = str(card_id)
    roots = [
        Path("downloaded_cards"),
        Path("public") / "cards",
        Path("cards"),
        Path("alt_arts"),
        Path("alt_arts") / "display",
        Path("alt_arts") / "small",
    ]
    stems = [cid]
    # Also try the base card name for places where generated public images drop
    # the printing number suffix.
    import re
    base = re.sub(r"_\d{5}$", "", cid)
    if base != cid:
        stems.append(base)

    for root in roots:
        for stem in stems:
            for ext in IMAGE_EXTENSIONS:
                candidates.append(root / f"{stem}{ext}")

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def format_summary(summary: Dict[str, Any]) -> str:
    margin = summary.get("margin")
    margin_text = "n/a" if margin is None else f"{_safe_float(margin):.3f}"
    lines = [
        "CollectorVision directed scan",
        f"result: {summary.get('id')}  score={_safe_float(summary.get('score')):.3f}  margin={margin_text}",
        f"reason: {summary.get('reason')}",
    ]
    if summary.get("sharpness") is not None or summary.get("confidence") is not None:
        lines.append(f"sharpness={summary.get('sharpness')}  confidence={summary.get('confidence')}")
    alts = summary.get("alternatives") or []
    if alts:
        lines.append("top alternatives:")
        for alt in alts[:8]:
            lines.append(f"  {alt.get('id')}  {_safe_float(alt.get('score')):.3f}  rot={alt.get('rotation')}")
    return "\n".join(lines)


def run_direct_diagnostic(frame_getter, frame, x1, y1, x2, y2, side, catalog, save_artifacts=True, live_tag=None) -> Dict[str, Any]:
    """Run CollectorVision directly on a user-drawn rectangle.

    When save_artifacts is False this overwrites a small fixed set of live files
    instead of creating a new timestamped artifact set every refresh.
    """
    h, w = frame.shape[:2]
    left = max(0, min(int(x1), int(x2), w - 1))
    right = min(w, max(int(x1), int(x2)))
    top = max(0, min(int(y1), int(y2), h - 1))
    bottom = min(h, max(int(y1), int(y2)))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if save_artifacts:
        base = f"{stamp}_{side}_{left}_{top}_{right}_{bottom}"
    else:
        safe_tag = live_tag or "live"
        base = f"{safe_tag}_{side}"

    if right - left < 10 or bottom - top < 10:
        report = {
            "ok": False,
            "region": [left, top, right, bottom],
            "refined_box": None,
            "summary": {"id": None, "score": 0.0, "margin": None, "reason": "region_too_small", "alternatives": []},
            "collectorvision_reason": "region_too_small",
        }
        return report

    source_crop = frame[top:bottom, left:right].copy()
    source_crop_path = OUT_DIR / f"{base}_source_crop.jpg" if save_artifacts else None
    if save_artifacts:
        cv2.imwrite(str(source_crop_path), source_crop)

    best = None
    candidate = None
    refined = None
    reason = "unknown"
    try:
        candidate, refined, reason = dewarp_manual_region_with_collectorvision(
            frame, left, top, right, bottom, source="live_direct_diagnostic"
        )
        if candidate is not None and refined is not None:
            best = _recognize_from_refined(
                frame=frame,
                candidate=candidate,
                side=side,
                candidate_index=9000,
                catalog=catalog,
                refined=refined,
                force_diagnostics=True,
                allow_card_back=False,
            )
            if best is not None:
                best["manual_candidate_source"] = "live_direct_diagnostic"
                best["manual_direct_reason"] = reason
    except Exception as exc:
        reason = f"collectorvision_exception:{exc}"

    summary = _result_summary(best, reason, refined)

    overlay = frame.copy()
    crop_overlay = source_crop.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 255, 255), 2)
    cv2.putText(crop_overlay, "user DIRECT CV region", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    refined_box = None
    if candidate is not None and candidate.get("box") is not None:
        refined_box = np.asarray(candidate.get("box"), dtype=np.int32)
        cv2.polylines(overlay, [refined_box], True, (0, 255, 255), 3)
        local_box = refined_box.copy()
        local_box[:, 0] -= left
        local_box[:, 1] -= top
        cv2.polylines(crop_overlay, [local_box], True, (0, 255, 255), 3)

    label = f"{summary.get('id')} {summary.get('score', 0.0):.2f} {summary.get('reason')}"
    cv2.putText(overlay, label[:120], (left, max(18, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(crop_overlay, label[:90], (8, max(42, crop_overlay.shape[0] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    overlay_path = OUT_DIR / f"{base}_full_frame_overlay.jpg" if save_artifacts else None
    crop_overlay_path = OUT_DIR / f"{base}_crop_with_cv_box.jpg" if save_artifacts else None
    if save_artifacts:
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(crop_overlay_path), crop_overlay)

    dewarped_path = None
    rotations_path = None
    if refined is not None and refined.get("image") is not None:
        dewarped_path = OUT_DIR / f"{base}_collectorvision_dewarped.jpg" if save_artifacts else None
        dewarped_bgr = _pil_to_bgr(refined["image"])
        if save_artifacts:
            cv2.imwrite(str(dewarped_path), dewarped_bgr)

        # Show the same dewarped crop in all four cardinal orientations.
        # Recognition already tests these, but seeing them makes upside-down /
        # handed-off-rotated crops obvious during live diagnostics.
        rotations_path = OUT_DIR / f"{base}_cardinal_rotations.jpg" if save_artifacts else None
        tiles = []
        for label_text, pil_image in (
            ("0", refined["image"]),
            ("90", refined["image"].rotate(90, expand=True)),
            ("180", refined["image"].rotate(180, expand=True)),
            ("270", refined["image"].rotate(270, expand=True)),
        ):
            tile = _pil_to_bgr(pil_image)
            target_h = 210
            scale = target_h / max(tile.shape[0], 1)
            tile = cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((240, 170, 3), dtype=np.uint8)
            xoff = max(0, (canvas.shape[1] - tile.shape[1]) // 2)
            yoff = 24
            tile = tile[:, :canvas.shape[1]]
            canvas[yoff:yoff + tile.shape[0], xoff:xoff + tile.shape[1]] = tile
            cv2.putText(canvas, f"rot {label_text}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            if str(summary.get("rotation")) == label_text:
                cv2.rectangle(canvas, (1, 1), (canvas.shape[1]-2, canvas.shape[0]-2), (0,255,255), 2)
            tiles.append(canvas)
        rotations_preview = np.hstack(tiles)
        if save_artifacts:
            cv2.imwrite(str(rotations_path), rotations_preview)

    matched_card_path = _find_catalog_image(summary.get("id"))

    report = {
        "ok": candidate is not None,
        "region": [left, top, right, bottom],
        "refined_box": refined_box.tolist() if refined_box is not None else None,
        "summary": summary,
        "collectorvision_reason": reason,
        "source_crop": str(source_crop_path) if source_crop_path else None,
        "crop_overlay": str(crop_overlay_path) if crop_overlay_path else None,
        "overlay": str(overlay_path) if overlay_path else None,
        "dewarped": str(dewarped_path) if dewarped_path else None,
        "matched_card": matched_card_path,
        "rotations": str(rotations_path) if rotations_path else None,
        "live": not save_artifacts,
    }
    if not save_artifacts:
        report["_preview_images"] = {
            "overlay": _bgr_to_pil(overlay),
            "crop_overlay": _bgr_to_pil(crop_overlay),
            "source_crop": _bgr_to_pil(source_crop),
            "dewarped": _bgr_to_pil(dewarped_bgr) if refined is not None and refined.get("image") is not None else None,
            "rotations": _bgr_to_pil(rotations_preview) if refined is not None and refined.get("image") is not None else None,
        }
    if save_artifacts:
        (OUT_DIR / f"{base}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def draw_direct_diagnostic_overlay(display_frame, report, scale=1.0):
    """Legacy helper kept for compatibility.

    The live GUI no longer calls this after a diagnostic scan because persistent
    DIRECT CV boxes were confusing. Diagnostic results are shown in their own
    window instead.
    """
    return
