import json

import cv2

from .config import (
    ROI_SETTINGS_FILE,
    HANDLE_SIZE,
    MIN_ROI_SIZE,
)

rois = {
    "left": None,
    "right": None,
}

drag_state = {
    "active": False,
    "side": None,
    "mode": None,
    "start_mouse": (0, 0),
    "start_roi": None,
}

def default_rois(frame_width, frame_height):
    split_x = frame_width // 2

    return {
        "left": [0, 0, split_x, frame_height],
        "right": [split_x, 0, frame_width - split_x, frame_height],
    }

def clamp_roi(roi, frame_width, frame_height):
    x, y, w, h = roi

    w = max(MIN_ROI_SIZE, w)
    h = max(MIN_ROI_SIZE, h)

    x = max(0, min(x, frame_width - MIN_ROI_SIZE))
    y = max(0, min(y, frame_height - MIN_ROI_SIZE))

    if x + w > frame_width:
        w = frame_width - x

    if y + h > frame_height:
        h = frame_height - y

    return [int(x), int(y), int(w), int(h)]

def load_rois(frame_width, frame_height):
    if not ROI_SETTINGS_FILE.exists():
        print("No ROI settings file found; using defaults.")
        return default_rois(frame_width, frame_height)

    try:
        data = json.loads(ROI_SETTINGS_FILE.read_text())

        loaded = {
            "left": clamp_roi(data["left"], frame_width, frame_height),
            "right": clamp_roi(data["right"], frame_width, frame_height),
        }

        print(f"Loaded ROI settings from {ROI_SETTINGS_FILE}")
        return loaded

    except Exception as e:
        print("Could not load ROI settings; using defaults.")
        print(e)
        return default_rois(frame_width, frame_height)

def save_rois():
    ROI_SETTINGS_FILE.write_text(json.dumps(rois, indent=2))
    print(f"\nSaved ROI settings to {ROI_SETTINGS_FILE}")

def point_in_roi(px, py, roi):
    x, y, w, h = roi
    return x <= px <= x + w and y <= py <= y + h

def get_resize_mode(px, py, roi):
    x, y, w, h = roi

    left = abs(px - x) <= HANDLE_SIZE
    right = abs(px - (x + w)) <= HANDLE_SIZE
    top = abs(py - y) <= HANDLE_SIZE
    bottom = abs(py - (y + h)) <= HANDLE_SIZE

    if left and top:
        return "resize_tl"
    if right and top:
        return "resize_tr"
    if left and bottom:
        return "resize_bl"
    if right and bottom:
        return "resize_br"
    if left:
        return "resize_l"
    if right:
        return "resize_r"
    if top:
        return "resize_t"
    if bottom:
        return "resize_b"

    return None

def update_roi_from_drag(side, mode, start_roi, dx, dy, frame_width, frame_height):
    x, y, w, h = start_roi

    if mode == "move":
        x += dx
        y += dy

    elif mode == "resize_l":
        x += dx
        w -= dx

    elif mode == "resize_r":
        w += dx

    elif mode == "resize_t":
        y += dy
        h -= dy

    elif mode == "resize_b":
        h += dy

    elif mode == "resize_tl":
        x += dx
        y += dy
        w -= dx
        h -= dy

    elif mode == "resize_tr":
        y += dy
        w += dx
        h -= dy

    elif mode == "resize_bl":
        x += dx
        w -= dx
        h += dy

    elif mode == "resize_br":
        w += dx
        h += dy

    rois[side] = clamp_roi([x, y, w, h], frame_width, frame_height)

def on_mouse(event, x, y, flags, param):
    frame_width, frame_height = param

    if event == cv2.EVENT_LBUTTONDOWN:
        for side in ["right", "left"]:
            roi = rois[side]

            resize_mode = get_resize_mode(x, y, roi)

            if resize_mode:
                drag_state["active"] = True
                drag_state["side"] = side
                drag_state["mode"] = resize_mode
                drag_state["start_mouse"] = (x, y)
                drag_state["start_roi"] = roi.copy()
                return

            if point_in_roi(x, y, roi):
                drag_state["active"] = True
                drag_state["side"] = side
                drag_state["mode"] = "move"
                drag_state["start_mouse"] = (x, y)
                drag_state["start_roi"] = roi.copy()
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_state["active"]:
            sx, sy = drag_state["start_mouse"]
            dx = x - sx
            dy = y - sy

            update_roi_from_drag(
                drag_state["side"],
                drag_state["mode"],
                drag_state["start_roi"],
                dx,
                dy,
                frame_width,
                frame_height,
            )

    elif event == cv2.EVENT_LBUTTONUP:
        drag_state["active"] = False
        drag_state["side"] = None
        drag_state["mode"] = None
        drag_state["start_roi"] = None
