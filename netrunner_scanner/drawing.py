import cv2
from .config import HANDLE_SIZE

ROI_LEFT_COLOR = (0, 255, 0)
ROI_RIGHT_COLOR = (255, 0, 0)

PROPOSAL_COLOR = (180, 180, 180)
REFINED_COLOR = (255, 255, 0)

DISPLAYED_COLOR = (0, 255, 255)
QUEUED_COLOR = (0, 165, 255)
UNDISPLAYED_COLOR = (255, 0, 255)
CARD_BACK_COLOR = (200, 200, 200)


def draw_roi(frame, side, roi):
    x, y, w, h = roi

    if side == "left":
        color = ROI_LEFT_COLOR
        label = "LEFT / PINK"
    else:
        color = ROI_RIGHT_COLOR
        label = "RIGHT / BLUE"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    for hx, hy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.rectangle(
            frame,
            (hx - HANDLE_SIZE // 2, hy - HANDLE_SIZE // 2),
            (hx + HANDLE_SIZE // 2, hy + HANDLE_SIZE // 2),
            color,
            -1,
        )

    cv2.putText(
        frame,
        label,
        (x + 12, y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


def state_color(match):
    if match.get("label") == "card_back":
        return CARD_BACK_COLOR
    if match.get("displayed"):
        return DISPLAYED_COLOR
    if match.get("queued"):
        return QUEUED_COLOR
    return UNDISPLAYED_COLOR


def draw_card_matches(frame, matches):
    for match in matches:
        proposal_box = match.get("proposal_box")
        refined_box = match.get("refined_box")
        active_box = match.get("box")
        used_fallback = match.get("used_fallback", False)

        if proposal_box is not None:
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
            text += f" edge:{detail['edge_ratio']:.3f}"

        text += suffix

        text_x = x
        text_y = y - 8

        if text_y < 20:
            text_y = y + h + 22

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
