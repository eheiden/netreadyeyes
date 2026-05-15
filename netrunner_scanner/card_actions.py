import time

import cv2

from .recognition import latest_matches, tracker, obs_queue, scan_point_for_card, scan_box_for_card, manual_choice_state, apply_manual_choice, close_manual_choices, open_manual_choices_for_track
from .roi import rois, point_in_roi, on_mouse as roi_on_mouse
from .display_utils import display_to_source
from .runtime_controls import manual_click_respects_queue, handle_control_click
from .config import GUI_DISPLAY_SCALE, LEFT_CLICK_FORCE_OBS, RIGHT_CLICK_CARD_MENU, MANUAL_DRAG_SCAN_ENABLED, MANUAL_DRAG_MIN_WIDTH_PX, MANUAL_DRAG_MIN_HEIGHT_PX, MANUAL_DRAG_PROCESS_SYNC, MANUAL_POINT_SCAN_ON_CLICK, RIGHT_CLICK_CLEAR_DELETES_TRACK, MANUAL_CHOICE_WIDTH, MANUAL_CHOICE_ROW_HEIGHT, MANUAL_CHOICE_MARGIN, MANUAL_CHOICE_TITLE

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
    "choice_buttons": {},
}

runtime = {
    "frame_getter": None,
    "catalog": None,
    "frame_size_getter": None,
    "point_submitter": None,
    "box_submitter": None,
    "status_controls_getter": None,
}


def handle_status_panel_click(x, y):
    scaled_x, scaled_y = display_to_source(x, y)

    frame_size_getter = runtime.get("frame_size_getter")
    controls_getter = runtime.get("status_controls_getter")

    if frame_size_getter is None or controls_getter is None:
        return False

    frame_width, _frame_height = frame_size_getter()
    scale = float(GUI_DISPLAY_SCALE)
    if scale <= 0:
        scale = 1.0

    scaled_frame_w = int(frame_width * scale)

    if scaled_x < scaled_frame_w:
        return False

    sidebar_x = int(scaled_x - scaled_frame_w)
    sidebar_y = int(scaled_y)

    return handle_control_click(sidebar_x, sidebar_y, controls_getter())


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

def draw_manual_choice_overlay(frame):
    if not manual_choice_state.get("active"):
        menu_state["choice_buttons"] = {}
        return

    choices = manual_choice_state.get("choices") or []
    if not choices:
        menu_state["choice_buttons"] = {}
        return

    frame_h, frame_w = frame.shape[:2]
    w = int(MANUAL_CHOICE_WIDTH)
    row_h = int(MANUAL_CHOICE_ROW_HEIGHT)
    margin = int(MANUAL_CHOICE_MARGIN)
    title_h = 38
    cancel_h = 34
    h = title_h + row_h * len(choices) + cancel_h + margin * 2

    x = int(manual_choice_state.get("x", 28))
    y = int(manual_choice_state.get("y", 92))

    # Keep the selector near the card, but clamp it into the visible camera frame.
    if x + w > frame_w - margin:
        x = max(margin, frame_w - w - margin)
    if y + h > frame_h - margin:
        y = max(margin, frame_h - h - margin)
    x = max(margin, x)
    y = max(margin, y)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (24, 24, 28), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 1)

    title = manual_choice_state.get("title") or MANUAL_CHOICE_TITLE
    cv2.putText(
        frame,
        title,
        (x + 12, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    buttons = {}

    for i, choice in enumerate(choices):
        by = y + title_h + i * row_h
        rect = (x + 10, by, w - 20, row_h - 6)
        buttons[str(i)] = rect
        card_id = str(choice.get("id", "unknown"))
        score = float(choice.get("score", 0.0))
        label = f"{i + 1}. {card_id}   {score:.2f}"

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (44, 44, 50), -1)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (90, 90, 98), 1)
        cv2.putText(
            frame,
            label[:58],
            (rect[0] + 8, rect[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    cancel_y = y + title_h + row_h * len(choices) + 6
    cancel_rect = (x + 10, cancel_y, 120, 26)
    buttons["cancel"] = cancel_rect
    cv2.rectangle(frame, (cancel_rect[0], cancel_rect[1]), (cancel_rect[0] + cancel_rect[2], cancel_rect[1] + cancel_rect[3]), (50, 36, 36), -1)
    cv2.rectangle(frame, (cancel_rect[0], cancel_rect[1]), (cancel_rect[0] + cancel_rect[2], cancel_rect[1] + cancel_rect[3]), (120, 90, 90), 1)
    cv2.putText(frame, "Cancel", (cancel_rect[0] + 10, cancel_rect[1] + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    menu_state["choice_buttons"] = buttons


def handle_manual_choice_click(x, y):
    if not manual_choice_state.get("active"):
        return False

    for key, rect in menu_state.get("choice_buttons", {}).items():
        if not point_in_rect(x, y, rect):
            continue

        if key == "cancel":
            close_manual_choices()
            return True

        try:
            return apply_manual_choice(int(key))
        except ValueError:
            return False

    return False


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
    tracks = tracker.tracks.get(side, [])

    if RIGHT_CLICK_CLEAR_DELETES_TRACK:
        before = len(tracks)
        tracker.tracks[side] = [track for track in tracks if track.get("track_id") != track_id]
        latest_matches[side] = [match for match in latest_matches.get(side, []) if match.get("track_id") != track_id]
        print(f"Deleted {side} track {track_id}")
        return len(tracker.tracks[side]) != before

    for track in tracks:
        if track.get("track_id") == track_id:
            track["label"] = "unknown"
            track["score"] = 0.0
            track["margin"] = None
            track["displayed"] = False
            track["queued"] = False
            track["last_processed_at"] = 0.0
            track["visual_signature"] = None
            track["last_raw_signature"] = None
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

    if action == "selector":
        open_manual_choices_for_track(side, track_id)
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

    w = 360
    h = 174

    frame_h, frame_w = frame.shape[:2]
    x = min(x, frame_w - w - 4)
    y = min(y, frame_h - h - 4)
    x = max(4, x)
    y = max(4, y)

    menu_state["x"] = x
    menu_state["y"] = y
    menu_state["buttons"] = {
        "selector": (x + 14, y + 40, w - 28, 34),
        "force": (x + 14, y + 82, w - 28, 34),
        "clear": (x + 14, y + 124, w - 28, 34),
    }

    cv2.rectangle(frame, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), 1)

    cv2.putText(frame, "Card actions", (x + 14, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

    draw_button(frame, "selector", "Open manual card selector", menu_state["buttons"]["selector"])
    draw_button(frame, "force", "Force to front of OBS queue", menu_state["buttons"]["force"])
    draw_button(frame, "clear", "Delete track + rescan", menu_state["buttons"]["clear"])


def handle_card_menu_key(key):
    if manual_choice_state.get("active"):
        if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            return apply_manual_choice(key - ord("1"))
        if key == 27:
            return close_manual_choices()

    if not menu_state["active"]:
        return False

    if key in (ord("c"), ord("C")):
        return run_menu_action("clear")

    if key in (ord("f"), ord("F")):
        return run_menu_action("force")

    if key in (ord("m"), ord("M")):
        return run_menu_action("selector")

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


def make_mouse_handler(frame_size_getter, frame_getter=None, catalog=None, point_submitter=None, box_submitter=None, status_controls_getter=None):
    runtime["frame_size_getter"] = frame_size_getter
    runtime["frame_getter"] = frame_getter
    runtime["catalog"] = catalog
    runtime["point_submitter"] = point_submitter
    runtime["box_submitter"] = box_submitter
    runtime["status_controls_getter"] = status_controls_getter

    def on_mouse(event, x, y, flags, param):
        mapped = display_click_to_frame_coords(x, y)

        if mapped is None:
            if event == cv2.EVENT_LBUTTONDOWN:
                handle_status_panel_click(x, y)
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
            if handle_manual_choice_click(source_x, source_y):
                return

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
