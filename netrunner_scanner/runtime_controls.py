
import json
from pathlib import Path

from .config import (
    CONTROL_SETTINGS_PATH,
    CONTROL_MODE_DEFAULT,
    CONTROL_MANUAL_CLICK_RESPECT_QUEUE_DEFAULT,
    CONTROL_QUEUE_SECONDS_DEFAULT,
    CONTROL_QUEUE_SECONDS_MIN,
    CONTROL_QUEUE_SECONDS_MAX,
)


_state = {
    "mode": CONTROL_MODE_DEFAULT,
    "manual_click_respect_queue": bool(CONTROL_MANUAL_CLICK_RESPECT_QUEUE_DEFAULT),
    "queue_wait_seconds": float(CONTROL_QUEUE_SECONDS_DEFAULT),
}

_dirty = False


def clamp_wait(value):
    return max(float(CONTROL_QUEUE_SECONDS_MIN), min(float(CONTROL_QUEUE_SECONDS_MAX), float(value)))


def load_settings():
    global _dirty

    path = Path(CONTROL_SETTINGS_PATH)

    if not path.exists():
        _state["queue_wait_seconds"] = clamp_wait(_state["queue_wait_seconds"])
        return dict(_state)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not load {path}: {exc}")
        return dict(_state)

    mode = str(data.get("mode", _state["mode"])).lower()
    if mode not in ("automatic", "manual"):
        mode = CONTROL_MODE_DEFAULT

    _state["mode"] = mode
    _state["manual_click_respect_queue"] = bool(data.get(
        "manual_click_respect_queue",
        _state["manual_click_respect_queue"],
    ))
    _state["queue_wait_seconds"] = clamp_wait(data.get(
        "queue_wait_seconds",
        _state["queue_wait_seconds"],
    ))
    _dirty = False
    return dict(_state)


def save_settings():
    global _dirty

    path = Path(CONTROL_SETTINGS_PATH)
    path.write_text(json.dumps(_state, indent=2), encoding="utf-8")
    _dirty = False
    print(f"Saved Net Ready Eyes settings to {path}")


def get_settings():
    return dict(_state)


def is_manual_mode():
    return _state["mode"] == "manual"


def auto_send_enabled():
    return _state["mode"] == "automatic"


def manual_click_respects_queue():
    return bool(_state["manual_click_respect_queue"])


def get_queue_wait_seconds():
    return clamp_wait(_state["queue_wait_seconds"])


def set_mode(mode):
    global _dirty

    mode = str(mode).lower()
    if mode not in ("automatic", "manual"):
        return False

    if _state["mode"] != mode:
        _state["mode"] = mode
        _dirty = True

    return True


def set_manual_click_respect_queue(value):
    global _dirty

    value = bool(value)
    if _state["manual_click_respect_queue"] != value:
        _state["manual_click_respect_queue"] = value
        _dirty = True

    return True


def set_queue_wait_seconds(value):
    global _dirty

    value = clamp_wait(value)
    if abs(_state["queue_wait_seconds"] - value) > 0.001:
        _state["queue_wait_seconds"] = value
        _dirty = True

    return True


def is_dirty():
    return _dirty


def handle_control_click(x, y, controls):
    # x/y are sidebar-local display coordinates.
    if not controls:
        return False

    buttons = controls.get("buttons", {})

    for name, rect in buttons.items():
        rx, ry, rw, rh = rect
        if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
            continue

        if name == "mode_auto":
            return set_mode("automatic")
        if name == "mode_manual":
            return set_mode("manual")
        if name == "click_queue":
            return set_manual_click_respect_queue(True)
        if name == "click_instant":
            return set_manual_click_respect_queue(False)
        if name == "save":
            save_settings()
            return True

    slider = controls.get("slider")
    if slider:
        sx, sy, sw, sh = slider
        if sx <= x <= sx + sw and sy - 10 <= y <= sy + sh + 10:
            frac = (x - sx) / max(1, sw)
            seconds = CONTROL_QUEUE_SECONDS_MIN + frac * (CONTROL_QUEUE_SECONDS_MAX - CONTROL_QUEUE_SECONDS_MIN)
            return set_queue_wait_seconds(seconds)

    return False


# Load as soon as the module is imported so runtime behavior matches saved UI.
load_settings()
