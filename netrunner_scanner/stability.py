
from pathlib import Path
from time import strftime

from .logging_utils import pretty_card_id, explain_reason, console_event

from .config import (
    STABILITY_DEBUG_ENABLED,
    STABILITY_EVENT_LOG_FILE,
    STABILITY_EVENT_LOG_MAX_LINES,
)

_recent_events = []


def log_event(side, track_id, event, **fields):
    if not STABILITY_DEBUG_ENABLED:
        return

    parts = [
        strftime("%H:%M:%S"),
        f"side={side}",
        f"track={track_id}",
        f"event={event}",
    ]

    for key, value in fields.items():
        parts.append(f"{key}={value}")

    line = " ".join(parts)
    _recent_events.append(line)

    if len(_recent_events) > STABILITY_EVENT_LOG_MAX_LINES:
        del _recent_events[: len(_recent_events) - STABILITY_EVENT_LOG_MAX_LINES]

    try:
        with Path(STABILITY_EVENT_LOG_FILE).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def recent_events(limit=8):
    return list(_recent_events[-limit:])



def log_human_event(side, track_id, event, **fields):
    if not STABILITY_DEBUG_ENABLED:
        return

    side_text = str(side).upper()
    label = pretty_card_id(fields.get("label") or fields.get("new") or fields.get("old"))
    reason = explain_reason(fields.get("reason") or fields.get("best_reason"))

    if event == "label_change":
        old_label = pretty_card_id(fields.get("old"))
        new_label = pretty_card_id(fields.get("new"))
        message = f"[{side_text}] Track {track_id}: {old_label} -> {new_label}. Reason: {reason}."
    elif event == "held_unknown":
        message = f"[{side_text}] Track {track_id}: held {label}; weak read ignored. Reason: {reason}."
    elif event == "same_spot_reuse":
        message = f"[{side_text}] Track {track_id}: kept {label}; same spot and crop still matches."
    elif event == "same_spot_changed":
        message = f"[{side_text}] Track {track_id}: same spot but crop changed; refreshing ID."
    elif event == "replacement_check":
        message = (
            f"[{side_text}] Track {track_id}: checking possible replacement for {label} "
            f"(missing {fields.get('missing')}s, moved {fields.get('moved')} px)."
        )
    elif event == "scan_summary":
        message = (
            f"[{side_text}] Scan: {fields.get('candidates')} candidates, "
            f"{fields.get('processed')} recognized, {fields.get('label_changes')} label changes, "
            f"{fields.get('held_unknown')} weak reads held."
        )
    else:
        message = f"[{side_text}] Track {track_id}: {event} {fields}"

    console_event(message)
    log_event(side, track_id, f"human_{event}", message=message)
