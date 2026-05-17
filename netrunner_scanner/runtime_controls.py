import json
import sys
from pathlib import Path

from . import config as _config
from .config import (
    CONTROL_SETTINGS_PATH,
    CONTROL_MODE_DEFAULT,
    CONTROL_MANUAL_CLICK_RESPECT_QUEUE_DEFAULT,
    CONTROL_QUEUE_SECONDS_DEFAULT,
    CONTROL_QUEUE_SECONDS_MIN,
    CONTROL_QUEUE_SECONDS_MAX,
    CONTROL_GPU_ENABLED_DEFAULT,
)

# Runtime-tweakable settings shown in the GUI.  Each entry is:
# key, label, config constant, type, min, max, step, group, help
_THRESHOLD_DEFINITIONS = [
    {"key": "track_expire_seconds", "label": "Track memory age-out", "constant": "TRACK_EXPIRE_SECONDS", "type": "float", "min": 0.5, "max": 120.0, "step": 0.5, "group": "Track age-out", "help": "How long an unmatched normal track remains in memory."},
    {"key": "track_visible_missing_seconds", "label": "Visible missing age-out", "constant": "TRACK_VISIBLE_MISSING_SECONDS", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "group": "Track age-out", "help": "How quickly a missing normal card disappears from the overlay."},
    {"key": "card_back_visible_missing_seconds", "label": "Card-back visible age-out", "constant": "CARD_BACK_VISIBLE_MISSING_SECONDS", "type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "group": "Track age-out", "help": "How quickly card_back boxes disappear when not seen."},
    {"key": "card_back_expire_seconds", "label": "Card-back memory age-out", "constant": "CARD_BACK_EXPIRE_SECONDS", "type": "float", "min": 0.1, "max": 30.0, "step": 0.1, "group": "Track age-out", "help": "How long card_back tracks remain in memory."},
    {"key": "track_center_match_threshold_px", "label": "Track center match px", "constant": "TRACK_CENTER_MATCH_THRESHOLD_PX", "type": "float", "min": 5.0, "max": 250.0, "step": 5.0, "group": "Track matching", "help": "How far a proposal can move and still match an existing track."},
    {"key": "track_reacquire_center_threshold_px", "label": "Reacquire same-card px", "constant": "TRACK_REACQUIRE_CENTER_THRESHOLD_PX", "type": "float", "min": 5.0, "max": 300.0, "step": 5.0, "group": "Track matching", "help": "Same-spot reacquire distance for known cards."},
    {"key": "track_requeue_center_threshold_px", "label": "Requeue movement px", "constant": "TRACK_REQUEUE_CENTER_THRESHOLD_PX", "type": "float", "min": 5.0, "max": 400.0, "step": 5.0, "group": "Track matching", "help": "How far a known card must move before it can queue again."},
    {"key": "track_reuse_after_missing_seconds", "label": "Reuse after missing seconds", "constant": "TRACK_REUSE_AFTER_MISSING_SECONDS", "type": "float", "min": 0.0, "max": 120.0, "step": 1.0, "group": "Track matching", "help": "After this, re-recognize even if the proposal is in the same spot."},
    {"key": "min_card_short_side_px", "label": "Minimum card short side px", "constant": "MIN_CARD_SHORT_SIDE_PX", "type": "int", "min": 20, "max": 250, "step": 1, "group": "Detection candidates", "help": "Reject tiny card candidates below this short-side size."},
    {"key": "edge_proposal_min_area_ratio", "label": "Edge proposal min area", "constant": "EDGE_PROPOSAL_MIN_AREA_RATIO", "type": "float", "min": 0.0005, "max": 0.05, "step": 0.0005, "group": "Detection candidates", "help": "Minimum ROI-area fraction for internal edge proposals."},
    {"key": "max_candidates_per_side", "label": "Max candidates per side", "constant": "MAX_CANDIDATES_PER_SIDE", "type": "int", "min": 1, "max": 30, "step": 1, "group": "Detection candidates", "help": "Limit card proposals per playmat scan."},
    {"key": "relative_card_size_min_area_fraction", "label": "Relative size min area", "constant": "RELATIVE_CARD_SIZE_MIN_AREA_FRACTION", "type": "float", "min": 0.1, "max": 1.2, "step": 0.01, "group": "Detection candidates", "help": "Reject candidates much smaller than other visible cards."},
    {"key": "partial_card_min_area_fraction", "label": "Partial-card min area", "constant": "PARTIAL_CARD_MIN_AREA_FRACTION", "type": "float", "min": 0.1, "max": 1.2, "step": 0.01, "group": "Detection candidates", "help": "Reject partial/fragment boxes relative to reference cards."},
    {"key": "partial_card_min_short_side_fraction", "label": "Partial-card min short side", "constant": "PARTIAL_CARD_MIN_SHORT_SIDE_FRACTION", "type": "float", "min": 0.1, "max": 1.2, "step": 0.01, "group": "Detection candidates", "help": "Reject partial boxes with too-small short side."},
    {"key": "card_back_confirmations", "label": "Card-back confirmations", "constant": "CARD_BACK_CONFIRMATIONS", "type": "int", "min": 1, "max": 8, "step": 1, "group": "Card backs / ghosts", "help": "Number of repeated card_back reads before accepting it."},
    {"key": "card_back_confirm_window_seconds", "label": "Card-back confirm window", "constant": "CARD_BACK_CONFIRM_WINDOW_SECONDS", "type": "float", "min": 0.5, "max": 20.0, "step": 0.5, "group": "Card backs / ghosts", "help": "Time window for card_back confirmation."},
    {"key": "auto_accept_new_card_min_score", "label": "New auto min score", "constant": "AUTO_ACCEPT_NEW_CARD_MIN_SCORE", "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "group": "Auto acceptance", "help": "Minimum score for a new automatic track."},
    {"key": "auto_accept_new_card_min_margin", "label": "New auto min margin", "constant": "AUTO_ACCEPT_NEW_CARD_MIN_MARGIN", "type": "float", "min": 0.0, "max": 0.6, "step": 0.01, "group": "Auto acceptance", "help": "Top-match margin required for a new automatic track."},
    {"key": "auto_accept_bad_geometry_min_score", "label": "Bad geometry min score", "constant": "AUTO_ACCEPT_BAD_GEOMETRY_MIN_SCORE", "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "group": "Auto acceptance", "help": "Score required when geometry warnings are present."},
    {"key": "auto_accept_bad_geometry_min_margin", "label": "Bad geometry min margin", "constant": "AUTO_ACCEPT_BAD_GEOMETRY_MIN_MARGIN", "type": "float", "min": 0.0, "max": 0.6, "step": 0.01, "group": "Auto acceptance", "help": "Margin required when geometry warnings are present."},
    {"key": "auto_accept_weak_geometry_if_score_at_least", "label": "Weak geometry rescue score", "constant": "AUTO_ACCEPT_WEAK_GEOMETRY_IF_SCORE_AT_LEAST", "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "group": "Auto acceptance", "help": "Let real cards through despite geometry warnings at/above this score."},
    {"key": "auto_accept_weak_geometry_if_margin_at_least", "label": "Weak geometry rescue margin", "constant": "AUTO_ACCEPT_WEAK_GEOMETRY_IF_MARGIN_AT_LEAST", "type": "float", "min": 0.0, "max": 0.6, "step": 0.01, "group": "Auto acceptance", "help": "Let real cards through despite geometry warnings at/above this margin."},
    {"key": "ambiguous_match_score_threshold", "label": "Ambiguous score threshold", "constant": "AMBIGUOUS_MATCH_SCORE_THRESHOLD", "type": "float", "min": 0.1, "max": 0.95, "step": 0.01, "group": "Match quality", "help": "Below/near this score, ambiguity checks matter more."},
    {"key": "ambiguous_match_min_margin", "label": "Ambiguous min margin", "constant": "AMBIGUOUS_MATCH_MIN_MARGIN", "type": "float", "min": 0.0, "max": 0.6, "step": 0.01, "group": "Match quality", "help": "Minimum top-vs-runner-up gap before accepting a match."},
    {"key": "raw_visual_diff_threshold", "label": "Raw visual diff threshold", "constant": "RAW_VISUAL_DIFF_THRESHOLD", "type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "group": "Visual recheck", "help": "How much a stationary crop must change before forced recheck."},
    {"key": "raw_visual_change_confirmations", "label": "Raw change confirmations", "constant": "RAW_VISUAL_CHANGE_CONFIRMATIONS", "type": "int", "min": 1, "max": 10, "step": 1, "group": "Visual recheck", "help": "Number of raw-diff hits needed before treating a card as changed."},
    {"key": "raw_visual_change_confirm_window_seconds", "label": "Raw change confirm window", "constant": "RAW_VISUAL_CHANGE_CONFIRM_WINDOW_SECONDS", "type": "float", "min": 0.5, "max": 30.0, "step": 0.5, "group": "Visual recheck", "help": "Time window for raw visual-change confirmation."},
    {"key": "moved_sanity_rescan_px", "label": "Moved sanity rescan px", "constant": "MOVED_SANITY_RESCAN_PX", "type": "float", "min": 1.0, "max": 200.0, "step": 1.0, "group": "Visual recheck", "help": "Movement before checking whether a known card changed."},
    {"key": "moved_sanity_rescan_cooldown_seconds", "label": "Moved sanity cooldown", "constant": "MOVED_SANITY_RESCAN_COOLDOWN_SECONDS", "type": "float", "min": 0.0, "max": 20.0, "step": 0.25, "group": "Visual recheck", "help": "Minimum seconds between movement sanity checks."},
]

_BOOL_THRESHOLD_DEFINITIONS = [
    {
        "key": "auto_reject_weak_new_tracks",
        "label": "Reject weak new automatic tracks",
        "constant": "AUTO_REJECT_WEAK_NEW_TRACKS",
        "type": "bool",
        "group": "Boolean guards",
        "help": "When on, brand-new automatic reads with weak score/margin or bad geometry are held as unknown instead of immediately becoming visible tracks.",
    },
    {
        "key": "hide_unconfirmed_unknown_tracks",
        "label": "Hide unconfirmed unknown tracks",
        "constant": "HIDE_UNCONFIRMED_UNKNOWN_TRACKS",
        "type": "bool",
        "group": "Boolean guards",
        "help": "When on, unknown tracks are kept internally but not drawn until they become a real card. This reduces ghost boxes.",
    },
    {
        "key": "hide_card_back_tracks",
        "label": "Hide card_back tracks",
        "constant": "HIDE_CARD_BACK_TRACKS",
        "type": "bool",
        "group": "Boolean guards",
        "help": "When on, card_back detections do not draw persistent boxes. Useful when blank playmat areas are being mistaken for card backs.",
    },
    {
        "key": "card_back_confirmation_enabled",
        "label": "Require card_back confirmation",
        "constant": "CARD_BACK_CONFIRMATION_ENABLED",
        "type": "bool",
        "group": "Boolean guards",
        "help": "Requires repeated card_back reads before accepting card_back. Turn down the confirmation count if real face-down cards are slow to appear.",
    },
    {
        "key": "last_known_reacquire_enabled",
        "label": "Last-known reacquire enabled",
        "constant": "LAST_KNOWN_REACQUIRE_ENABLED",
        "type": "bool",
        "group": "Boolean guards",
        "help": "Allows a recently missing card to be reattached near its old position. Helpful for jitter, but can create ghosts after a card is removed.",
    },
    {
        "key": "partial_card_reject_enabled",
        "label": "Reject partial-card candidates",
        "constant": "PARTIAL_CARD_REJECT_ENABLED",
        "type": "bool",
        "group": "Boolean guards",
        "help": "Rejects candidate boxes that look too small relative to the other cards. Helps ignore fragments, chips, and leftover edges.",
    },
    {
        "key": "relative_card_size_filter_enabled",
        "label": "Relative card-size filter",
        "constant": "RELATIVE_CARD_SIZE_FILTER_ENABLED",
        "type": "bool",
        "group": "Boolean guards",
        "help": "Filters proposals that are much smaller than established cards on the same playmat. Turn off if cards at different depths look very different in size.",
    },
    {
        "key": "always_raw_visual_diff_enabled",
        "label": "Raw visual diff recheck",
        "constant": "ALWAYS_RAW_VISUAL_DIFF_ENABLED",
        "type": "bool",
        "group": "Boolean guards",
        "help": "Allows stationary cards to be rechecked when their crop changes visually. Helpful for flips, but too sensitive can cause repeated relabeling.",
    },
]

_THRESHOLD_DEFINITIONS.extend(_BOOL_THRESHOLD_DEFINITIONS)

_DEF_BY_KEY = {d["key"]: d for d in _THRESHOLD_DEFINITIONS}
_DEF_BY_CONSTANT = {d["constant"]: d for d in _THRESHOLD_DEFINITIONS}


def _default_for(defn):
    return getattr(_config, defn["constant"])


def _coerce(defn, value):
    kind = defn.get("type")
    if kind == "bool":
        return bool(value)
    if kind == "int":
        value = int(round(float(value)))
    else:
        value = float(value)
    if "min" in defn:
        value = max(defn["min"], value)
    if "max" in defn:
        value = min(defn["max"], value)
    if kind == "int":
        return int(value)
    return float(value)


def _default_thresholds():
    return {d["key"]: _coerce(d, _default_for(d)) for d in _THRESHOLD_DEFINITIONS}


_state = {
    "mode": CONTROL_MODE_DEFAULT,
    "manual_click_respect_queue": bool(CONTROL_MANUAL_CLICK_RESPECT_QUEUE_DEFAULT),
    "queue_wait_seconds": float(CONTROL_QUEUE_SECONDS_DEFAULT),
    "gpu_enabled": bool(CONTROL_GPU_ENABLED_DEFAULT),
    "thresholds": _default_thresholds(),
}

_dirty = False


def clamp_wait(value):
    return max(float(CONTROL_QUEUE_SECONDS_MIN), min(float(CONTROL_QUEUE_SECONDS_MAX), float(value)))


def threshold_definitions():
    return [dict(d) for d in _THRESHOLD_DEFINITIONS]


def get_threshold_values():
    return dict(_state.get("thresholds", {}))


def _apply_threshold_to_loaded_modules(defn, value):
    constant = defn["constant"]
    setattr(_config, constant, value)

    # Many scanner modules imported config constants with `from config import X`,
    # so update those module globals too when values change at runtime.
    for module_name in (
        "netrunner_scanner.recognition",
        "netrunner_scanner.tracking",
        "netrunner_scanner.detection",
        "netrunner_scanner.display_utils",
        "netrunner_scanner.drawing",
        "netrunner_scanner.motion",
        "netrunner_scanner.scanner_worker",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, constant):
            setattr(module, constant, value)

    # Keep the already-created CardTracker instance in sync for constructor args.
    recognition = sys.modules.get("netrunner_scanner.recognition")
    if recognition is not None:
        tracker = getattr(recognition, "tracker", None)
        if tracker is not None:
            mapping = {
                "TRACK_IOU_THRESHOLD": "iou_threshold",
                "STATIONARY_CENTER_THRESHOLD_PX": "stationary_center_threshold_px",
                "STATIONARY_REFRESH_SECONDS": "stationary_refresh_seconds",
                "TRACK_EXPIRE_SECONDS": "expire_seconds",
                "REFINED_BOX_SMOOTHING_ALPHA": "refined_box_smoothing_alpha",
            }
            attr = mapping.get(constant)
            if attr is not None:
                try:
                    setattr(tracker, attr, value)
                except Exception:
                    pass


def apply_thresholds():
    thresholds = _state.setdefault("thresholds", _default_thresholds())
    for defn in _THRESHOLD_DEFINITIONS:
        value = _coerce(defn, thresholds.get(defn["key"], _default_for(defn)))
        thresholds[defn["key"]] = value
        _apply_threshold_to_loaded_modules(defn, value)


def set_threshold_value(key, value):
    global _dirty
    defn = _DEF_BY_KEY.get(key)
    if defn is None:
        return False
    value = _coerce(defn, value)
    thresholds = _state.setdefault("thresholds", _default_thresholds())
    old_value = thresholds.get(key, _default_for(defn))
    thresholds[key] = value
    _apply_threshold_to_loaded_modules(defn, value)
    if old_value != value:
        _dirty = True
    return True


def reset_threshold_values():
    global _dirty
    _state["thresholds"] = _default_thresholds()
    apply_thresholds()
    _dirty = True
    return True


def load_settings():
    global _dirty

    path = Path(CONTROL_SETTINGS_PATH)

    if not path.exists():
        _state["queue_wait_seconds"] = clamp_wait(_state["queue_wait_seconds"])
        apply_thresholds()
        return dict(_state)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not load {path}: {exc}")
        apply_thresholds()
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
    _state["gpu_enabled"] = bool(data.get("gpu_enabled", _state["gpu_enabled"]))

    incoming_thresholds = data.get("thresholds", {})
    thresholds = _default_thresholds()
    if isinstance(incoming_thresholds, dict):
        for key, value in incoming_thresholds.items():
            defn = _DEF_BY_KEY.get(key)
            if defn is not None:
                try:
                    thresholds[key] = _coerce(defn, value)
                except Exception:
                    pass
    _state["thresholds"] = thresholds
    apply_thresholds()
    _dirty = False
    return dict(_state)


def save_settings():
    global _dirty

    path = Path(CONTROL_SETTINGS_PATH)
    payload = dict(_state)
    payload["thresholds"] = dict(_state.get("thresholds", {}))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


def gpu_enabled():
    return bool(_state.get("gpu_enabled", True))


def set_gpu_enabled(value):
    global _dirty

    value = bool(value)
    if bool(_state.get("gpu_enabled", True)) != value:
        _state["gpu_enabled"] = value
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
        if name == "gpu_on":
            return set_gpu_enabled(True)
        if name == "gpu_off":
            return set_gpu_enabled(False)
        if name == "save":
            save_settings()
            return True

    for name, delta in (("wait_minus", -1.0), ("wait_plus", 1.0)):
        rect = buttons.get(name)
        if not rect:
            continue
        rx, ry, rw, rh = rect
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return set_queue_wait_seconds(_state["queue_wait_seconds"] + delta)

    return False


# Load as soon as the module is imported so runtime behavior matches saved UI.
load_settings()
