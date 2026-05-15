import time

import cv2

from .recognition import latest_matches, tracker, obs_queue, scan_point_for_card, scan_box_for_card
from .roi import rois, point_in_roi, on_mouse as roi_on_mouse
from .display_utils import display_to_source
from .config import GUI_DISPLAY_SCALE, LEFT_CLICK_FORCE_OBS, RIGHT_CLICK_CARD_MENU, MANUAL_DRAG_SCAN_ENABLED, MANUAL_DRAG_MIN_WIDTH_PX, MANUAL_DRAG_MIN_HEIGHT_PX, MANUAL_DRAG_PROCESS_SYNC, MANUAL_POINT_SCAN_ON_CLICK

menu_state = {
    "active": False,
    "side": None,
    "track_id": None,
    "x": 0,
    "y": 0,
    "hover": None,
    "buttons": {},
    "dragging_box": False,
    "drag_start": None,
    "drag_current": None,
    "drag_status": "",
    "drag_status_until": 0.0,
    "pending_click_scan": False,
    "pending_click_start": None,
    "pending_click_side": None,
}

runtime = {
    "frame_getter": None,
    "catalog": None,
    "frame_size_getter": None,
    "point_submitter": None,
    "box_submitter": None,
}


def display_click_to_frame_coords(x, y):
    scaled_x, scaled_y = display_to_source(x, y)

    frame_size_getter = runtime.get("frame_size_getter")

    if frame_size_getter is None:
        return int(scaled_x), int(scaled_y)

    frame_width, frame_height = frame_size_getter()

    scale = float(GUI_DISPLAY_SCALE)
    if scale <= 0:
        scale = 1.0

    scaled_frame_w = int(frame_width * scale)
    scaled_frame_h = int(frame_height * scale)

    # Ignore clicks in the status sidebar or outside the camera preview.
    if scaled_x < 0 or scaled_y < 0:
        return None

    if scaled_x >= scaled_frame_w or scaled_y >= scaled_frame_h:
        return None

    source_x = int(scaled_x / scale)
    source_y = int(scaled_y / scale)

    source_x = max(0, min(source_x, frame_width - 1))
    source_y = max(0, min(source_y, frame_height - 1))

    return source_x, source_y


def point_in_box(px, py, box):
    return cv2.pointPolygonTest(box, (float(px), float(py)), False) >= 0


def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def force_match_to_front(side, match):
    track_id = match.get("track_id")
    if track_id is None:
        return False
    return force_track_to_front(side, track_id)


def draw_manual_drag_box(frame):
    if menu_state.get("dragging_box"):
        start = menu_state.get("drag_start")
        current = menu_state.get("drag_current")

        if start is None or current is None:
            return

        x1, y1 = start
        x2, y2 = current

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            frame,
            "manual scan area",
            (min(x1, x2), max(18, min(y1, y2) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return

    if menu_state.get("drag_status") and time.time() < menu_state.get("drag_status_until", 0.0):
        cv2.putText(
            frame,
            menu_state["drag_status"],
            (28, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )


def find_clicked_match(x, y):
    for side in ["left", "right"]:
        for match in latest_matches.get(side, []):
            box = match.get("box")
            if box is not None and point_in_box(x, y, box):
                return side, match

    return None, None


def side_for_point(x, y):
    for side in ["left", "right"]:
        roi = rois.get(side)
        if roi is not None and point_in_roi(x, y, roi):
            return side

    return None


def clear_track_identification(side, track_id):
    for track in tracker.tracks.get(side, []):
        if track.get("track_id") == track_id:
            track["label"] = "unknown"
            track["score"] = 0.0
            track["margin"] = None
            track["displayed"] = False
            track["queued"] = False
            track["last_processed_at"] = 0.0
            track["visual_signature"] = None
            print(f"Cleared identification for {side} track {track_id}")
            return True

    return False


def force_track_to_front(side, track_id):
    for track in tracker.tracks.get(side, []):
        if track.get("track_id") != track_id:
            continue

        label = track.get("label")

        if label in (None, "", "unknown", "tracking", "card_back"):
            print(f"Cannot send track {track_id}; label is {label}")
            return False

        obs_queue.enqueue_front(
            side=side,
            card_id=label,
            score=float(track.get("score", 1.0)),
            track=track,
        )
        print(f"Forced to front of OBS queue: {side} {label}")
        return True

    return False


def run_menu_action(action):
    if not menu_state["active"]:
        return False

    side = menu_state["side"]
    track_id = menu_state["track_id"]

    if action == "clear":
        clear_track_identification(side, track_id)
        menu_state["active"] = False
        return True

    if action == "force":
        force_track_to_front(side, track_id)
        menu_state["active"] = False
        return True

    return False


def update_hover(x, y):
    menu_state["hover"] = None

    if not menu_state["active"]:
        return

    for name, rect in menu_state.get("buttons", {}).items():
        if point_in_rect(x, y, rect):
            menu_state["hover"] = name
            return


def draw_button(frame, name, label, rect):
    x, y, w, h = rect
    hovered = menu_state.get("hover") == name

    bg = (70, 70, 70) if hovered else (42, 42, 42)
    border = (0, 220, 255) if hovered else (150, 150, 150)

    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)
    cv2.putText(frame, label, (x + 12, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)


def draw_card_menu(frame):
    if not menu_state["active"]:
        return

    x = int(menu_state["x"])
    y = int(menu_state["y"])

    w = 330
    h = 130

    frame_h, frame_w = frame.shape[:2]
    x = min(x, frame_w - w - 4)
    y = min(y, frame_h - h - 4)
    x = max(4, x)
    y = max(4, y)

    menu_state["x"] = x
    menu_state["y"] = y
    menu_state["buttons"] = {
        "clear": (x + 14, y + 40, w - 28, 34),
        "force": (x + 14, y + 82, w - 28, 34),
    }

    cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), 1)

    cv2.putText(frame, "Card actions", (x + 14, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

    draw_button(frame, "clear", "Clear ID + refresh", menu_state["buttons"]["clear"])
    draw_button(frame, "force", "Force to front of OBS queue", menu_state["buttons"]["force"])


def handle_card_menu_key(key):
    if not menu_state["active"]:
        return False

    if key in (ord("c"), ord("C")):
        return run_menu_action("clear")

    if key in (ord("f"), ord("F")):
        return run_menu_action("force")

    if key == 27:
        menu_state["active"] = False
        return True

    return False


def scan_unrecognized_spot(x, y):
    side = side_for_point(x, y)

    if side is None:
        return False

    frame_getter = runtime.get("frame_getter")
    catalog = runtime.get("catalog")

    if frame_getter is None or catalog is None:
        return False

    frame = frame_getter()

    if frame is None:
        return False

    print(f"Manual click scan: {side} at ({x}, {y})")
    point_submitter = runtime.get("point_submitter")
    if point_submitter is not None:
        point_submitter(frame, x, y, side)
    else:
        scan_point_for_card(frame, x, y, side, catalog)
    return True


def make_mouse_handler(frame_size_getter, frame_getter=None, catalog=None, point_submitter=None, box_submitter=None):
    runtime["frame_size_getter"] = frame_size_getter
    runtime["frame_getter"] = frame_getter
    runtime["catalog"] = catalog
    runtime["point_submitter"] = point_submitter
    runtime["box_submitter"] = box_submitter

    def on_mouse(event, x, y, flags, param):
        mapped = display_click_to_frame_coords(x, y)

        if mapped is None:
            return

        source_x, source_y = mapped
        frame_width, frame_height = frame_size_getter()

        if event == cv2.EVENT_MOUSEMOVE:
            update_hover(source_x, source_y)

            if menu_state.get("pending_click_scan"):
                start = menu_state.get("pending_click_start")
                if start is not None:
                    dx = abs(source_x - start[0])
                    dy = abs(source_y - start[1])
                    if dx >= MANUAL_DRAG_MIN_WIDTH_PX or dy >= MANUAL_DRAG_MIN_HEIGHT_PX:
                        menu_state["dragging_box"] = True
                        menu_state["drag_start"] = start
                        menu_state["drag_current"] = (source_x, source_y)
                        menu_state["pending_click_scan"] = False

            if menu_state.get("dragging_box"):
                menu_state["drag_current"] = (source_x, source_y)
            else:
                roi_on_mouse(event, source_x, source_y, flags, (frame_width, frame_height))
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            side, match = find_clicked_match(source_x, source_y)

            if match is not None and RIGHT_CLICK_CARD_MENU:
                menu_state["active"] = True
                menu_state["side"] = side
                menu_state["track_id"] = match.get("track_id")
                menu_state["x"] = source_x + 12
                menu_state["y"] = source_y + 12
                menu_state["hover"] = None
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            if menu_state["active"]:
                update_hover(source_x, source_y)
                hover = menu_state.get("hover")
                if hover is not None:
                    run_menu_action(hover)
                    return
                menu_state["active"] = False

            side, match = find_clicked_match(source_x, source_y)

            if match is not None and LEFT_CLICK_FORCE_OBS:
                force_match_to_front(side, match)
                # Fast visual feedback.
                match["queued"] = True
                match["displayed"] = False
                return

            side = side_for_point(source_x, source_y)

            if side is not None and MANUAL_DRAG_SCAN_ENABLED:
                menu_state["pending_click_scan"] = True
                menu_state["pending_click_start"] = (source_x, source_y)
                menu_state["pending_click_side"] = side
                return

            roi_on_mouse(event, source_x, source_y, flags, (frame_width, frame_height))
            return

        if event == cv2.EVENT_LBUTTONUP:
            if menu_state.get("pending_click_scan"):
                menu_state["pending_click_scan"] = False
                start = menu_state.get("pending_click_start") or (source_x, source_y)
                menu_state["pending_click_start"] = None
                menu_state["pending_click_side"] = None

                if MANUAL_POINT_SCAN_ON_CLICK:
                    scan_unrecognized_spot(start[0], start[1])
                    return

            if menu_state.get("dragging_box"):
                start = menu_state.get("drag_start")
                current = menu_state.get("drag_current") or (source_x, source_y)

                menu_state["dragging_box"] = False
                menu_state["drag_start"] = None
                menu_state["drag_current"] = None

                if start is not None:
                    x1, y1 = start
                    x2, y2 = current
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)

                    if w >= MANUAL_DRAG_MIN_WIDTH_PX and h >= MANUAL_DRAG_MIN_HEIGHT_PX:
                        side = side_for_point((x1 + x2) // 2, (y1 + y2) // 2)
                        frame_getter = runtime.get("frame_getter")
                        frame = frame_getter() if frame_getter is not None else None
                        if side is not None and frame is not None:
                            box_submitter = runtime.get("box_submitter")
                            menu_state["drag_status"] = f"manual scan: {side}"
                            menu_state["drag_status_until"] = time.time() + 1.25

                            if MANUAL_DRAG_PROCESS_SYNC:
                                scan_box_for_card(frame, x1, y1, x2, y2, side, runtime.get("catalog"))
                            elif box_submitter is not None:
                                box_submitter(frame, x1, y1, x2, y2, side)
                            else:
                                scan_box_for_card(frame, x1, y1, x2, y2, side, runtime.get("catalog"))
                            return

                    # Treat a tiny drag as a point scan.
                    scan_unrecognized_spot(source_x, source_y)
                    return

            roi_on_mouse(event, source_x, source_y, flags, (frame_width, frame_height))
            return

        roi_on_mouse(event, source_x, source_y, flags, (frame_width, frame_height))

    return on_mouse
