
import cv2
import numpy as np

from .runtime_controls import get_settings, is_dirty

last_controls = {}


def get_last_controls():
    return dict(last_controls)

from .config import (
    GUI_STATUS_MAX_QUEUE_ITEMS,
    GUI_STATUS_MAX_RECOGNIZED_ITEMS,
    GUI_STATUS_SIDEBAR_WIDTH,
    STATUS_TEXT_TRUNCATION_SUFFIX,
    STATUS_LOG_WRAP_WIDTH,
    STATUS_MAX_STABILITY_EVENTS,
    STATUS_PANEL_COMPACT_MODE,
    STATUS_PANEL_SHOW_RECOGNIZED_LISTS,
    STATUS_PANEL_POLISHED,
    STATUS_PANEL_SHOW_LEGEND,
    STATUS_PANEL_SHOW_LAST_SENT,
    APP_NAME,
    APP_VERSION,
    STATUS_PANEL_SHOW_CONTROLS,
    CONTROL_QUEUE_SECONDS_MIN,
    CONTROL_QUEUE_SECONDS_MAX,
)


def trim_id(card_id, max_len=34):
    card_id = str(card_id)

    if len(card_id) <= max_len:
        return card_id

    return card_id[: max_len - len(STATUS_TEXT_TRUNCATION_SUFFIX)] + STATUS_TEXT_TRUNCATION_SUFFIX


def draw_text(panel, text, x, y, scale=0.48, color=(230, 230, 230), thickness=1):
    if y < 0 or y >= panel.shape[0] - 4:
        return
    cv2.putText(
        panel,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_section_header(panel, title, y):
    if STATUS_PANEL_POLISHED:
        cv2.rectangle(panel, (0, y - 20), (panel.shape[1], y + 9), (38, 38, 42), -1)
        cv2.line(panel, (0, y + 9), (panel.shape[1], y + 9), (62, 62, 66), 1)
        draw_text(panel, title, 14, y, scale=0.50, color=(245, 245, 245), thickness=1)
    else:
        cv2.rectangle(panel, (0, y - 18), (panel.shape[1], y + 8), (45, 45, 45), -1)
        draw_text(panel, title, 12, y, scale=0.52, color=(255, 255, 255), thickness=1)


def draw_legend(panel, y):
    draw_section_header(panel, "Legend", y)
    y += 26

    rows = [
        ("yellow", "sent / handled"),
        ("orange", "queued"),
        ("magenta", "new / unknown"),
        ("gray", "card_back"),
        ("opencv", "rough crop"),
        ("cv", "refined crop"),
        ("m", "match gap"),
        ("s", "corner sharpness"),
        ("stable", "cached / not moving"),
    ]

    color_map = {
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "magenta": (255, 0, 255),
        "gray": (200, 200, 200),
    }

    for key, meaning in rows:
        color = color_map.get(key, (210, 210, 210))
        draw_text(panel, key, 16, y, scale=0.40, color=color)
        draw_text(panel, meaning, 100, y, scale=0.40, color=(165, 165, 165))
        y += 16

    return y + 8



def draw_perf(panel, status, y):
    draw_section_header(panel, "Performance", y)
    y += 28
    perf = status.get("perf", {})

    rows = [
        ("CPU", f"{perf.get('process_cpu_percent', 0.0):.1f}%"),
        ("gui", f"{perf.get('gui_frame_ms', 0.0):.1f} ms"),
        ("left scan", f"{perf.get('worker_left_ms', 0.0):.1f} ms"),
        ("right scan", f"{perf.get('worker_right_ms', 0.0):.1f} ms"),
        ("point scan", f"{perf.get('worker_point_ms', 0.0):.1f} ms"),
        ("jobs", f"q {perf.get('queued_jobs', 0):.0f} / drop {perf.get('dropped_jobs', 0):.0f}"),
    ]
    for key, val in rows:
        draw_text(panel, key, 16, y, scale=0.40, color=(180, 180, 180))
        draw_text(panel, val, 140, y, scale=0.40, color=(210, 210, 210))
        y += 17

    gpu = perf.get("gpu", {})
    if gpu:
        provider = trim_id(gpu.get("active", "unknown"), 26)
        enabled = "on" if gpu.get("gpu_enabled") else "off"
        draw_text(panel, "GPU", 16, y, scale=0.40, color=(180, 180, 180))
        draw_text(panel, f"{enabled} / {provider}", 80, y, scale=0.36, color=(210, 210, 210))
        y += 17
        bench = gpu.get("last_benchmark_ms")
        if bench is not None:
            draw_text(panel, "bench", 16, y, scale=0.36, color=(160, 160, 160))
            draw_text(panel, f"{float(bench):.1f} ms/img", 80, y, scale=0.36, color=(190, 190, 190))
            y += 15

    threads = perf.get("thread_cpu", [])
    if threads:
        draw_text(panel, "top threads", 16, y, scale=0.40, color=(210, 210, 210))
        y += 17
        for row in threads[:4]:
            name = trim_id(row.get("name", "?"), 24)
            cpu = float(row.get("cpu_percent", 0.0))
            draw_text(panel, name, 16, y, scale=0.36, color=(160, 160, 160))
            draw_text(panel, f"{cpu:.1f}%", 220, y, scale=0.36, color=(190, 190, 190))
            y += 15
    elif not perf.get("psutil_available", True):
        draw_text(panel, "install psutil for thread CPU", 16, y, scale=0.38, color=(180, 150, 150))
        y += 16

    return y + 8


def draw_match_list(panel, title, matches, y):
    draw_section_header(panel, title, y)
    y += 30

    if not matches:
        draw_text(panel, "none", 16, y, color=(150, 150, 150))
        return y + 24

    for match in matches[:GUI_STATUS_MAX_RECOGNIZED_ITEMS]:
        label = trim_id(match.get("label", "unknown"), 28)
        score = float(match.get("score", 0.0))
        state = []

        if match.get("queued"):
            state.append("queued")
        if match.get("displayed"):
            state.append("sent")
        if match.get("stationary"):
            state.append("stable")

        state_text = ",".join(state) if state else "new"

        color = (230, 230, 230)
        if label == "card_back":
            color = (190, 190, 190)
        elif label == "unknown":
            color = (255, 100, 255)
        elif match.get("queued"):
            color = (0, 190, 255)
        elif match.get("displayed") or match.get("stationary"):
            color = (0, 255, 255)

        draw_text(panel, f"{label}", 16, y, color=color)
        diag = ""
        if match.get("unknown_streak", 0):
            diag += f" u{match.get('unknown_streak')}"
        if match.get("force_new_identification"):
            diag += " force"
        if match.get("needs_same_spot_signature_check"):
            diag += " sig?"
        if match.get("last_signature_distance") is not None:
            diag += f" sig{match.get('last_signature_distance'):.2f}"
        if match.get("last_raw_visual_diff") is not None:
            diag += f" raw{match.get('last_raw_visual_diff'):.2f}"
        if match.get("raw_visual_change_pending"):
            diag += " raw!"
        if match.get("missing_seconds", 0.0) > 0.2:
            diag += f" miss{match.get('missing_seconds', 0.0):.1f}"

        draw_text(panel, f"{score:.2f} {state_text}{diag}", 16, y + 17, scale=0.40, color=(160, 160, 160))
        decision = str(match.get("last_decision", ""))
        if decision:
            draw_text(panel, decision[:36], 16, y + 33, scale=0.34, color=(120, 120, 120))
            y += 58
        else:
            y += 42

    return y + 8


def draw_queue(panel, queue_snapshot, y):
    draw_section_header(panel, "OBS FIFO", y)
    y += 30

    queue = queue_snapshot.get("queue", [])
    next_send = queue_snapshot.get("next_send_in", {})

    if isinstance(next_send, dict):
        draw_text(
            panel,
            f"next L:{next_send.get('left', 0.0):.1f}s R:{next_send.get('right', 0.0):.1f}s",
            16,
            y,
            scale=0.45,
            color=(170, 170, 170),
        )
    else:
        draw_text(panel, f"next send: {float(next_send):.1f}s", 16, y, scale=0.45, color=(170, 170, 170))

    y += 24

    if not queue:
        draw_text(panel, "queue empty", 16, y, color=(150, 150, 150))
        return y + 30

    for idx, item in enumerate(queue[:GUI_STATUS_MAX_QUEUE_ITEMS], start=1):
        side = item.get("side", "?")
        card_id = trim_id(item.get("card_id", ""), 27)
        score = float(item.get("score", 0.0))
        age = float(item.get("age", 0.0))

        draw_text(panel, f"{idx}. [{side}] {card_id}", 16, y, color=(0, 190, 255))
        draw_text(panel, f"score {score:.2f}  age {age:.1f}s", 34, y + 17, scale=0.42, color=(160, 160, 160))
        y += 42

    return y + 8


def draw_last_sent(panel, queue_snapshot, y):
    draw_section_header(panel, "Last Sent", y)
    y += 30

    last_sent = queue_snapshot.get("last_sent", {})

    for side in ["left", "right"]:
        item = last_sent.get(side)

        if item:
            card_id = trim_id(item.get("card_id", ""), 28)
            score = float(item.get("score", 0.0))
            draw_text(panel, f"{side}: {card_id}", 16, y, color=(0, 255, 255))
            draw_text(panel, f"score {score:.2f}", 34, y + 17, scale=0.42, color=(160, 160, 160))
        else:
            draw_text(panel, f"{side}: none", 16, y, color=(150, 150, 150))

        y += 42

    return y + 8


def clean_event_text(event):
    text = str(event)

    if " message=" in text:
        text = text.split(" message=", 1)[1]

    return text.replace(" event=human_", " ")


def wrap_words(text, width):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [""]


def draw_stability_events(panel, events, y):
    draw_section_header(panel, "Stability Log", y)
    y += 26

    if not events:
        draw_text(panel, "no recent events", 16, y, scale=0.38, color=(150, 150, 150))
        return y + 24

    shown = 0
    # Prefer human-readable events if present.
    human_events = [event for event in events if "human_" in str(event)]
    source_events = human_events if human_events else events

    for event in source_events[-STATUS_MAX_STABILITY_EVENTS:]:
        text = clean_event_text(event)
        for line in wrap_words(text, STATUS_LOG_WRAP_WIDTH)[:3]:
            draw_text(panel, line, 16, y, scale=0.34, color=(170, 170, 170))
            y += 15
        y += 4
        shown += 1

    return y + 8



def draw_radio(panel, label, x, y, active):
    color = (0, 220, 255) if active else (120, 120, 120)
    cv2.circle(panel, (x + 8, y - 5), 7, color, 1, cv2.LINE_AA)
    if active:
        cv2.circle(panel, (x + 8, y - 5), 4, color, -1, cv2.LINE_AA)
    draw_text(panel, label, x + 24, y, scale=0.40, color=(220, 220, 220))


def draw_button_panel(panel, label, rect, active=False):
    x, y, w, h = rect
    bg = (48, 54, 58) if active else (38, 38, 42)
    border = (0, 220, 255) if active else (90, 90, 98)
    cv2.rectangle(panel, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), border, 1)
    draw_text(panel, label, x + 8, y + 21, scale=0.40, color=(230, 230, 230))


def draw_controls(panel, y):
    draw_section_header(panel, "Controls", y)
    y += 30

    settings = get_settings()
    mode = settings.get("mode", "automatic")
    respect_queue = bool(settings.get("manual_click_respect_queue", False))
    wait = float(settings.get("queue_wait_seconds", 8.0))
    gpu_on = bool(settings.get("gpu_enabled", True))

    buttons = {}

    draw_text(panel, "Recognition mode", 16, y, scale=0.40, color=(170, 170, 170))
    y += 23

    buttons["mode_auto"] = (18, y - 20, 132, 26)
    buttons["mode_manual"] = (160, y - 20, 132, 26)
    draw_radio(panel, "Automatic", 20, y, mode == "automatic")
    draw_radio(panel, "Manual", 162, y, mode == "manual")
    y += 34

    draw_text(panel, "Left-click OBS send", 16, y, scale=0.40, color=(170, 170, 170))
    y += 23

    buttons["click_instant"] = (18, y - 20, 132, 26)
    buttons["click_queue"] = (160, y - 20, 132, 26)
    draw_radio(panel, "Instant", 20, y, not respect_queue)
    draw_radio(panel, "Use queue", 162, y, respect_queue)
    y += 34

    draw_text(panel, "Queue wait seconds", 16, y, scale=0.40, color=(170, 170, 170))
    y += 24

    buttons["wait_minus"] = (18, y - 21, 30, 26)
    buttons["wait_plus"] = (130, y - 21, 30, 26)

    draw_button_panel(panel, "-", buttons["wait_minus"], active=False)
    cv2.rectangle(panel, (56, y - 21), (122, y + 5), (30, 30, 34), -1)
    cv2.rectangle(panel, (56, y - 21), (122, y + 5), (90, 90, 98), 1)
    draw_text(panel, f"{wait:.1f}", 72, y, scale=0.45, color=(235, 235, 235))
    draw_button_panel(panel, "+", buttons["wait_plus"], active=False)
    draw_text(panel, f"range {CONTROL_QUEUE_SECONDS_MIN:.0f}-{CONTROL_QUEUE_SECONDS_MAX:.0f}", 176, y, scale=0.36, color=(135, 135, 135))
    y += 38

    draw_text(panel, "GPU acceleration", 16, y, scale=0.40, color=(170, 170, 170))
    y += 23

    buttons["gpu_on"] = (18, y - 20, 132, 26)
    buttons["gpu_off"] = (160, y - 20, 132, 26)
    draw_radio(panel, "On", 20, y, gpu_on)
    draw_radio(panel, "Off", 162, y, not gpu_on)
    y += 34

    save_label = "Save" + (" *" if is_dirty() else "")
    buttons["save"] = (18, y - 20, 120, 28)
    draw_button_panel(panel, save_label, buttons["save"], active=is_dirty())
    y += 42

    controls = {
        "buttons": buttons,
    }

    return y + 8, controls


def make_status_sidebar(height, status):
    panel = np.zeros((height, GUI_STATUS_SIDEBAR_WIDTH, 3), dtype=np.uint8)
    panel[:] = (19, 20, 23) if STATUS_PANEL_POLISHED else (24, 24, 24)

    if STATUS_PANEL_POLISHED:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (26, 27, 31), -1)
        cv2.line(panel, (0, 44), (panel.shape[1], 44), (55, 55, 60), 1)

    y = 28
    draw_text(panel, f"{APP_NAME} v{APP_VERSION}", 14, y, scale=0.60, color=(248, 248, 248), thickness=1)
    y += 30

    global last_controls
    # Runtime controls moved to the real Tk menu/ribbon.  Keep the drawn
    # sidebar focused on stream-facing information only.
    last_controls = {}

    if STATUS_PANEL_SHOW_LEGEND and not STATUS_PANEL_COMPACT_MODE:
        y = draw_legend(panel, y)

    y = draw_perf(panel, status, y)
    y = draw_stability_events(panel, status.get("stability_events", []), y)

    queue_snapshot = status.get("obs_queue", {})
    y = draw_queue(panel, queue_snapshot, y)

    if STATUS_PANEL_SHOW_LAST_SENT and not STATUS_PANEL_COMPACT_MODE:
        y = draw_last_sent(panel, queue_snapshot, y)

    if STATUS_PANEL_SHOW_RECOGNIZED_LISTS:
        matches = status.get("matches", {})
        if y < height - 80:
            y = draw_match_list(panel, "Left Recognized", matches.get("left", []), y)
        if y < height - 80:
            y = draw_match_list(panel, "Right Recognized", matches.get("right", []), y)

    return panel
