
import cv2
import numpy as np

from .config import (
    GUI_STATUS_MAX_QUEUE_ITEMS,
    GUI_STATUS_MAX_RECOGNIZED_ITEMS,
    GUI_STATUS_SIDEBAR_WIDTH,
    STATUS_TEXT_TRUNCATION_SUFFIX,
)


def trim_id(card_id, max_len=34):
    card_id = str(card_id)

    if len(card_id) <= max_len:
        return card_id

    return card_id[: max_len - len(STATUS_TEXT_TRUNCATION_SUFFIX)] + STATUS_TEXT_TRUNCATION_SUFFIX


def draw_text(panel, text, x, y, scale=0.48, color=(230, 230, 230), thickness=1):
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


def draw_stability_events(panel, events, y):
    draw_section_header(panel, "Stability Log", y)
    y += 26

    if not events:
        draw_text(panel, "no recent events", 16, y, scale=0.38, color=(150, 150, 150))
        return y + 24

    for event in events[-5:]:
        draw_text(panel, trim_id(event, 48), 16, y, scale=0.34, color=(155, 155, 155))
        y += 15

    return y + 8


def make_status_sidebar(height, status):
    panel = np.zeros((height, GUI_STATUS_SIDEBAR_WIDTH, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)

    y = 28
    draw_text(panel, "CollectorVision Status", 12, y, scale=0.62, color=(255, 255, 255), thickness=1)
    y += 28

    y = draw_legend(panel, y)
    y = draw_perf(panel, status, y)
    y = draw_stability_events(panel, status.get("stability_events", []), y)

    queue_snapshot = status.get("obs_queue", {})
    y = draw_queue(panel, queue_snapshot, y)
    y = draw_last_sent(panel, queue_snapshot, y)

    matches = status.get("matches", {})
    y = draw_match_list(panel, "Left / Pink Recognized", matches.get("left", []), y)
    y = draw_match_list(panel, "Right / Blue Recognized", matches.get("right", []), y)

    return panel
