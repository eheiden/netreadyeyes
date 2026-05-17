
import cv2
import numpy as np
from .config import HANDLE_SIZE, DRAW_LABEL_SHADOWS, ROI_LEFT_LABEL, ROI_RIGHT_LABEL
from .roi import roi_points, roi_to_rect, roi_enabled, roi_color, hover_state, drag_state, roi_edit_enabled, show_roi_labels

ROI_LEFT_COLOR = (0, 255, 0)
ROI_RIGHT_COLOR = (255, 0, 0)

PROPOSAL_COLOR = (110, 110, 110)
REFINED_COLOR = (255, 255, 0)

DISPLAYED_COLOR = (0, 255, 255)
QUEUED_COLOR = (0, 165, 255)
UNDISPLAYED_COLOR = (255, 0, 255)
CARD_BACK_COLOR = (200, 200, 200)


def draw_label(frame, text, origin, color, scale=0.58, thickness=2):
    x, y = origin
    if DRAW_LABEL_SHADOWS:
        cv2.putText(
            frame,
            text,
            (x + 2, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _lighten_bgr(color, amount=80):
    return tuple(min(255, int(c) + amount) for c in color)


def draw_roi(frame, side, roi):
    if roi is None:
        return

    color = roi_color(roi, ROI_LEFT_COLOR if side == "left" else ROI_RIGHT_COLOR)
    label = ROI_LEFT_LABEL if side == "left" else ROI_RIGHT_LABEL
    enabled = roi_enabled(roi)
    pts = roi_points(roi)

    if pts is None:
        return

    pts_i = pts.astype(np.int32)
    x, y, w, h = roi_to_rect(roi)

    edit_mode = roi_edit_enabled()
    is_hovered = edit_mode and hover_state.get("side") == side
    is_dragged = edit_mode and drag_state.get("active") and drag_state.get("side") == side
    active = is_hovered or is_dragged
    draw_color = _lighten_bgr(color, 55) if active else color
    thickness = 3 if active else 2

    if not enabled:
        draw_color = (90, 90, 90)
        thickness = 1

    cv2.polylines(frame, [pts_i], True, draw_color, thickness, cv2.LINE_AA)

    # Soft fill while hovered/dragged so it is obvious the mouse is in an editable spot.
    if active and enabled:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_i], draw_color)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    for idx, (hx, hy) in enumerate(pts_i):
        radius = max(4, HANDLE_SIZE // 2) if edit_mode else 2
        if active and hover_state.get("mode", "").startswith("corner"):
            radius += 2
        cv2.circle(frame, (int(hx), int(hy)), radius, draw_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(hx), int(hy)), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

    mode = hover_state.get("mode") if is_hovered else None
    hint = ""
    if edit_mode and mode == "move":
        hint = "  drag to move"
    elif edit_mode and mode and mode.startswith("corner"):
        hint = "  drag corner to dewarp"
    elif edit_mode and mode and mode.startswith("edge"):
        hint = "  drag edge"

    if show_roi_labels():
        disabled = " OFF" if not enabled else ""
        draw_label(frame, f"{label}{disabled}{hint}", (x + 12, y + 32), draw_color, scale=0.82, thickness=2)


def is_known_card_label(label):
    return label not in (None, "", "unknown", "tracking", "card_back")


def state_color(match):
    label = match.get("label")

    if label == "card_back":
        return CARD_BACK_COLOR

    if match.get("queued"):
        return QUEUED_COLOR

    if match.get("displayed"):
        return DISPLAYED_COLOR

    # If a known card is stable and not in the FIFO, treat it as already handled
    # visually. This avoids the "pink forever" state for cached/reacquired cards.
    if is_known_card_label(label) and match.get("stationary"):
        return DISPLAYED_COLOR

    return UNDISPLAYED_COLOR


def draw_card_matches(frame, matches):
    for match in matches:
        proposal_box = match.get("proposal_box")
        refined_box = match.get("refined_box")
        active_box = match.get("box")
        used_fallback = match.get("used_fallback", False)

        # Draw the rough proposal only if it is meaningfully different from the
        # active box. This keeps stale-looking gray boxes from cluttering the UI.
        if proposal_box is not None and active_box is not None:
            px, py, pw, ph = cv2.boundingRect(proposal_box)
            ax, ay, aw, ah = cv2.boundingRect(active_box)
            center_delta = abs((px + pw / 2) - (ax + aw / 2)) + abs((py + ph / 2) - (ay + ah / 2))
            size_delta = abs(pw - aw) + abs(ph - ah)

            if center_delta > 18 or size_delta > 24:
                cv2.drawContours(frame, [proposal_box], 0, PROPOSAL_COLOR, 1)
        elif proposal_box is not None:
            cv2.drawContours(frame, [proposal_box], 0, PROPOSAL_COLOR, 1)

        if refined_box is not None and not used_fallback:
            cv2.drawContours(frame, [refined_box], 0, REFINED_COLOR, 2)

        color = state_color(match)

        if active_box is not None:
            cv2.drawContours(frame, [active_box], 0, color, 2)

        box_for_label = active_box if active_box is not None else proposal_box

        if box_for_label is None:
            continue

        x, y, w, h = cv2.boundingRect(box_for_label)

        label = match["label"]
        score = match["score"]
        suffix = " stable" if match.get("stationary") else ""
        mode = "opencv" if used_fallback else "cv"

        sharpness = match.get("detector_sharpness")
        margin = match.get("margin")
        detail = match.get("detail_metrics")

        text = f"{label} {score:.2f} {mode}"

        if margin is not None:
            text += f" m:{margin:.2f}"

        if sharpness is not None:
            text += f" s:{sharpness:.3f}"

        if label == "card_back" and detail is not None:
            edge_value = detail.get("edge_ratio", detail.get("center_edge_ratio", 0.0))
            text += f" edge:{edge_value:.3f}"

        text += suffix

        text_x = x
        text_y = y - 8

        if text_y < 20:
            text_y = y + h + 22

        draw_label(frame, text, (text_x, text_y), color, scale=0.58, thickness=2)
