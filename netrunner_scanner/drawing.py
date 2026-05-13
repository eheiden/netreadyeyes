import cv2

from .config import HANDLE_SIZE

def draw_roi(frame, side, roi):
    x, y, w, h = roi

    if side == "left":
        color = (0, 255, 0)
        label = "LEFT / PINK"
    else:
        color = (255, 0, 0)
        label = "RIGHT / BLUE"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    for hx, hy in [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h),
    ]:
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

def draw_card_matches(frame, matches):
    color = (0, 255, 255)

    for match in matches:
        box = match["box"]
        label = match["label"]
        score = match["score"]

        cv2.drawContours(frame, [box], 0, color, 2)

        x, y, w, h = cv2.boundingRect(box)

        text = f"{label} {score:.2f}"

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
