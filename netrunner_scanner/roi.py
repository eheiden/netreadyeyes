import json
import math

import cv2
import numpy as np

from .config import ROI_SETTINGS_FILE, HANDLE_SIZE, MIN_ROI_SIZE

DEFAULT_LEFT_COLOR = (0, 255, 0)
DEFAULT_RIGHT_COLOR = (255, 0, 0)
ROI_COLOR_PRESETS = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 220, 255),    # yellow/cyan-ish in BGR display
    (255, 0, 255),    # magenta
    (255, 255, 255),  # white
    (0, 0, 255),      # red
]

rois = {"left": None, "right": None}

hover_state = {
    "side": None,
    "mode": None,
    "point": None,
}

drag_state = {
    "active": False,
    "side": None,
    "mode": None,
    "start_mouse": (0, 0),
    "start_roi": None,
}

roi_edit_state = {
    # Locked by default so normal click-drag remains manual operation.
    "enabled": False,
}

roi_display_options = {
    # These are visual-only labels drawn over each playmat ROI.
    "show_roi_labels": True,
}


def show_roi_labels():
    return bool(roi_display_options.get("show_roi_labels", True))


def set_show_roi_labels(enabled):
    roi_display_options["show_roi_labels"] = bool(enabled)
    return roi_display_options["show_roi_labels"]


def roi_edit_enabled():
    return bool(roi_edit_state.get("enabled", False))


def set_roi_edit_enabled(enabled):
    roi_edit_state["enabled"] = bool(enabled)
    if not enabled:
        drag_state["active"] = False
        drag_state["side"] = None
        drag_state["mode"] = None
        drag_state["start_roi"] = None
        hover_state["side"] = None
        hover_state["mode"] = None
        hover_state["point"] = None
    return roi_edit_state["enabled"]


def toggle_roi_edit_enabled():
    return set_roi_edit_enabled(not roi_edit_enabled())


def _as_bgr_tuple(value, fallback):
    if value is None:
        return fallback
    try:
        if isinstance(value, str):
            value = value.strip().lstrip("#")
            if len(value) == 6:
                r = int(value[0:2], 16)
                g = int(value[2:4], 16)
                b = int(value[4:6], 16)
                return (b, g, r)
        b, g, r = value
        return (int(b), int(g), int(r))
    except Exception:
        return fallback


def default_rois(frame_width, frame_height):
    split_x = frame_width // 2
    return {
        "left": make_rect_roi(0, 0, split_x, frame_height, enabled=True, color=DEFAULT_LEFT_COLOR),
        "right": make_rect_roi(split_x, 0, frame_width - split_x, frame_height, enabled=True, color=DEFAULT_RIGHT_COLOR),
    }


def make_rect_roi(x, y, w, h, enabled=True, color=DEFAULT_LEFT_COLOR):
    x, y, w, h = int(x), int(y), int(w), int(h)
    points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return {
        "type": "quad",
        "points": points,
        "enabled": bool(enabled),
        "color": [int(color[0]), int(color[1]), int(color[2])],
    }


def normalize_roi(roi, frame_width=None, frame_height=None, default_color=DEFAULT_LEFT_COLOR):
    if roi is None:
        return None

    if isinstance(roi, dict):
        enabled = bool(roi.get("enabled", True))
        color = _as_bgr_tuple(roi.get("color"), default_color)
        if "points" in roi:
            points = np.asarray(roi.get("points"), dtype=np.float32).reshape(4, 2)
            normalized = {
                "type": "quad",
                "points": [[int(round(x)), int(round(y))] for x, y in points],
                "enabled": enabled,
                "color": [int(color[0]), int(color[1]), int(color[2])],
            }
        else:
            x, y, w, h = roi_to_rect(roi)
            normalized = make_rect_roi(x, y, w, h, enabled=enabled, color=color)
    else:
        x, y, w, h = roi
        normalized = make_rect_roi(x, y, w, h, enabled=True, color=default_color)

    if frame_width is not None and frame_height is not None:
        normalized["points"] = clamp_points(normalized["points"], frame_width, frame_height)
        if rect_too_small(normalized):
            x, y, w, h = roi_to_rect(normalized)
            normalized = make_rect_roi(x, y, max(MIN_ROI_SIZE, w), max(MIN_ROI_SIZE, h), enabled=normalized.get("enabled", True), color=roi_color(normalized, default_color))
            normalized["points"] = clamp_points(normalized["points"], frame_width, frame_height)

    return normalized


def clamp_points(points, frame_width, frame_height):
    clamped = []
    for x, y in points:
        clamped.append([
            int(max(0, min(round(x), frame_width - 1))),
            int(max(0, min(round(y), frame_height - 1))),
        ])
    return clamped


def rect_too_small(roi):
    _x, _y, w, h = roi_to_rect(roi)
    return w < MIN_ROI_SIZE or h < MIN_ROI_SIZE


def roi_points(roi):
    normalized = normalize_roi(roi)
    if normalized is None:
        return None
    return np.asarray(normalized["points"], dtype=np.float32)


def roi_to_rect(roi):
    if isinstance(roi, dict) and "points" in roi:
        pts = np.asarray(roi["points"], dtype=np.float32)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        return [int(x), int(y), int(w), int(h)]
    x, y, w, h = roi
    return [int(x), int(y), int(w), int(h)]


def roi_enabled(roi):
    if roi is None:
        return False
    if isinstance(roi, dict):
        return bool(roi.get("enabled", True))
    return True


def roi_color(roi, fallback=DEFAULT_LEFT_COLOR):
    if isinstance(roi, dict):
        return _as_bgr_tuple(roi.get("color"), fallback)
    return fallback


def set_roi_enabled(side, enabled):
    roi = rois.get(side)
    if roi is not None:
        roi["enabled"] = bool(enabled)


def toggle_roi_enabled(side):
    roi = rois.get(side)
    if roi is None:
        return False
    roi["enabled"] = not roi_enabled(roi)
    return roi["enabled"]


def cycle_roi_color(side):
    roi = rois.get(side)
    if roi is None:
        return None
    current = roi_color(roi, DEFAULT_LEFT_COLOR if side == "left" else DEFAULT_RIGHT_COLOR)
    try:
        idx = ROI_COLOR_PRESETS.index(current)
    except ValueError:
        idx = -1
    new_color = ROI_COLOR_PRESETS[(idx + 1) % len(ROI_COLOR_PRESETS)]
    roi["color"] = [int(new_color[0]), int(new_color[1]), int(new_color[2])]
    return new_color


def load_rois(frame_width, frame_height):
    if not ROI_SETTINGS_FILE.exists():
        print("No ROI settings file found; using defaults.")
        return default_rois(frame_width, frame_height)

    try:
        data = json.loads(ROI_SETTINGS_FILE.read_text())
        options = data.get("options", {}) if isinstance(data, dict) else {}
        if isinstance(options, dict):
            roi_display_options["show_roi_labels"] = bool(options.get("show_roi_labels", True))
        loaded = {
            "left": normalize_roi(data.get("left"), frame_width, frame_height, DEFAULT_LEFT_COLOR),
            "right": normalize_roi(data.get("right"), frame_width, frame_height, DEFAULT_RIGHT_COLOR),
        }
        if loaded["left"] is None or loaded["right"] is None:
            raise ValueError("ROI settings file is missing left or right ROI")
        print(f"Loaded ROI settings from {ROI_SETTINGS_FILE}")
        return loaded
    except Exception as e:
        print("Could not load ROI settings; using defaults.")
        print(e)
        return default_rois(frame_width, frame_height)


def save_rois():
    payload = {
        "left": rois.get("left"),
        "right": rois.get("right"),
        "options": dict(roi_display_options),
    }
    ROI_SETTINGS_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved ROI settings to {ROI_SETTINGS_FILE}")


def point_in_roi(px, py, roi):
    if roi is None or not roi_enabled(roi):
        return False
    pts = roi_points(roi)
    if pts is None:
        return False
    return cv2.pointPolygonTest(pts.astype(np.float32), (float(px), float(py)), False) >= 0


def _distance_to_segment(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def get_resize_mode(px, py, roi):
    pts = roi_points(roi)
    if pts is None:
        return None

    names = ["tl", "tr", "br", "bl"]
    handle_radius = max(HANDLE_SIZE, 14)
    for idx, (hx, hy) in enumerate(pts):
        if math.hypot(px - hx, py - hy) <= handle_radius:
            return f"corner_{names[idx]}"

    edge_names = ["top", "right", "bottom", "left"]
    for idx in range(4):
        ax, ay = pts[idx]
        bx, by = pts[(idx + 1) % 4]
        if _distance_to_segment(px, py, ax, ay, bx, by) <= HANDLE_SIZE:
            return f"edge_{edge_names[idx]}"

    return None


def _move_points(points, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in points]


def _resize_edge(points, mode, dx, dy):
    pts = np.asarray(points, dtype=np.float32).copy()
    if mode == "edge_top":
        pts[0, 1] += dy
        pts[1, 1] += dy
    elif mode == "edge_right":
        pts[1, 0] += dx
        pts[2, 0] += dx
    elif mode == "edge_bottom":
        pts[2, 1] += dy
        pts[3, 1] += dy
    elif mode == "edge_left":
        pts[0, 0] += dx
        pts[3, 0] += dx
    return [[int(round(x)), int(round(y))] for x, y in pts]


def update_roi_from_drag(side, mode, start_roi, dx, dy, frame_width, frame_height):
    roi = normalize_roi(start_roi, frame_width, frame_height, DEFAULT_LEFT_COLOR if side == "left" else DEFAULT_RIGHT_COLOR)
    points = [p[:] for p in roi["points"]]

    corner_indexes = {"corner_tl": 0, "corner_tr": 1, "corner_br": 2, "corner_bl": 3}

    if mode == "move":
        points = _move_points(points, dx, dy)
    elif mode in corner_indexes:
        idx = corner_indexes[mode]
        points[idx] = [int(round(points[idx][0] + dx)), int(round(points[idx][1] + dy))]
    elif mode.startswith("edge_"):
        points = _resize_edge(points, mode, dx, dy)

    roi["points"] = clamp_points(points, frame_width, frame_height)
    rois[side] = roi


def update_hover(px, py):
    hover_state["side"] = None
    hover_state["mode"] = None
    hover_state["point"] = None

    if not roi_edit_enabled():
        return

    for side in ["right", "left"]:
        roi = rois.get(side)
        if roi is None or not roi_enabled(roi):
            continue
        mode = get_resize_mode(px, py, roi)
        if mode:
            hover_state.update({"side": side, "mode": mode, "point": (px, py)})
            return
        if point_in_roi(px, py, roi):
            hover_state.update({"side": side, "mode": "move", "point": (px, py)})
            return


def active_or_hover_side(default="left"):
    return drag_state.get("side") or hover_state.get("side") or default


def square_up_roi(side):
    roi = rois.get(side)
    if roi is None:
        return
    x, y, w, h = roi_to_rect(roi)
    rois[side] = make_rect_roi(x, y, w, h, enabled=roi_enabled(roi), color=roi_color(roi, DEFAULT_LEFT_COLOR if side == "left" else DEFAULT_RIGHT_COLOR))


def on_mouse(event, x, y, flags, param):
    if not roi_edit_enabled():
        return

    frame_width, frame_height = param

    if event == cv2.EVENT_MOUSEMOVE and not drag_state["active"]:
        update_hover(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        update_hover(x, y)
        for side in ["right", "left"]:
            roi = rois.get(side)
            if roi is None or not roi_enabled(roi):
                continue
            resize_mode = get_resize_mode(x, y, roi)

            if resize_mode:
                drag_state.update({
                    "active": True,
                    "side": side,
                    "mode": resize_mode,
                    "start_mouse": (x, y),
                    "start_roi": json.loads(json.dumps(roi)),
                })
                return

            if point_in_roi(x, y, roi):
                drag_state.update({
                    "active": True,
                    "side": side,
                    "mode": "move",
                    "start_mouse": (x, y),
                    "start_roi": json.loads(json.dumps(roi)),
                })
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
        update_hover(x, y)


def roi_is_axis_aligned(roi, tolerance_px=2.0):
    """Return True when a quad ROI is really just a normal rectangle.

    This matters because the default saved ROI format may now be a quad, but
    forcing every rectangular playmat through warpPerspective can soften edges
    enough to make contour detection stop finding cards.  Only perspective-warp
    when the user has actually dragged corners into a non-rectangular quad.
    """
    pts = roi_points(roi)
    if pts is None:
        return True
    x, y, w, h = roi_to_rect(roi)
    rect = np.asarray(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )
    return bool(np.max(np.abs(pts.astype(np.float32) - rect)) <= float(tolerance_px))


def roi_output_size(roi):
    pts = roi_points(roi)
    if pts is None:
        return None
    tl, tr, br, bl = pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(MIN_ROI_SIZE, int(max(width_a, width_b)))
    height = max(MIN_ROI_SIZE, int(max(height_a, height_b)))
    return width, height


def dewarp_roi_for_scan(frame, roi):
    """Return (scan_frame, scan_roi, to_source_matrix, is_dewarped).

    Rectangular ROIs stay on the original frame.  Only edited, non-rectangular
    quadrilateral ROIs are perspective-warped.  This preserves the original
    contour/recognition behavior for normal overhead camera setups while still
    supporting parallax correction when the ROI corners have actually been
    adjusted.
    """
    if roi is None or not roi_enabled(roi):
        return frame, None, None, False

    normalized = normalize_roi(roi, frame.shape[1], frame.shape[0])
    pts = roi_points(normalized)
    size = roi_output_size(normalized)
    if pts is None or size is None or roi_is_axis_aligned(normalized):
        return frame, roi_to_rect(normalized), None, False

    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    inverse = cv2.getPerspectiveTransform(dst, pts.astype(np.float32))
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped, [0, 0, width, height], inverse, True


def transform_box(box, matrix):
    if box is None or matrix is None:
        return box
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return np.round(mapped).astype(np.int32)


def transform_match_for_display(match, matrix):
    if matrix is None:
        return dict(match)
    out = dict(match)
    for key in ("box", "proposal_box", "refined_box"):
        if out.get(key) is not None:
            out[key] = transform_box(out[key], matrix)
    return out
