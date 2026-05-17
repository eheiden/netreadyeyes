
import time
import cv2
import numpy as np

from .config import (
    TRACK_CENTER_MATCH_THRESHOLD_PX,
    TRACK_VISIBLE_MISSING_SECONDS,
    TRACK_REACQUIRE_CENTER_THRESHOLD_PX,
    TRACK_REQUEUE_CENTER_THRESHOLD_PX,
    TRACK_REUSE_AFTER_MISSING_SECONDS,
    MOVED_SANITY_RESCAN_ENABLED,
    MOVED_SANITY_RESCAN_PX,
    MOVED_SANITY_RESCAN_COOLDOWN_SECONDS,
    MOVED_REIDENTIFY_ENABLED,
    MOVED_REIDENTIFY_MIN_PX,
    MOVED_REIDENTIFY_COOLDOWN_SECONDS,
    CARD_BACK_VISIBLE_MISSING_SECONDS,
    CARD_BACK_EXPIRE_SECONDS,
    HIDE_UNCONFIRMED_UNKNOWN_TRACKS,
    HIDE_CARD_BACK_TRACKS,
)
from .stability import log_event, log_human_event


def box_to_rect(box):
    return cv2.boundingRect(box)


def rect_center(rect):
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def center_distance(a, b):
    ax, ay = rect_center(a)
    bx, by = rect_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def smooth_box(old_box, new_box, alpha=0.75):
    if new_box is None:
        return old_box

    if old_box is None:
        return new_box

    return (
        old_box.astype(np.float32) * alpha
        + new_box.astype(np.float32) * (1.0 - alpha)
    ).astype(np.intp)


class CardTracker:
    def __init__(
        self,
        iou_threshold,
        stationary_center_threshold_px,
        stationary_refresh_seconds,
        expire_seconds,
        refined_box_smoothing_alpha=0.75,
    ):
        self.iou_threshold = iou_threshold
        self.stationary_center_threshold_px = stationary_center_threshold_px
        self.stationary_refresh_seconds = stationary_refresh_seconds
        self.expire_seconds = expire_seconds
        self.refined_box_smoothing_alpha = refined_box_smoothing_alpha
        self.next_track_id = 1
        self.tracks = {"left": [], "right": []}

    def current_time(self):
        return time.time()

    def expire_old_tracks(self, side, now=None):
        if now is None:
            now = time.time()

        kept = []
        for track in self.tracks[side]:
            label = track.get("label")
            max_age = CARD_BACK_EXPIRE_SECONDS if label == "card_back" else self.expire_seconds
            if now - track["last_seen_at"] <= max_age:
                kept.append(track)

        self.tracks[side] = kept

    def match_track(self, side, box):
        rect = box_to_rect(box)
        best_track = None
        best_score = -1.0

        for track in self.tracks[side]:
            old_rect = track["rect"]
            iou_score = rect_iou(rect, old_rect)
            distance = center_distance(rect, old_rect)
            has_known_label = track.get("label") not in (None, "", "unknown")

            if iou_score >= self.iou_threshold:
                score = 1000.0 + iou_score
            elif has_known_label and distance <= TRACK_REACQUIRE_CENTER_THRESHOLD_PX:
                score = 700.0 - distance
            elif distance <= TRACK_CENTER_MATCH_THRESHOLD_PX:
                score = 500.0 - distance
            else:
                continue

            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    def should_reprocess(self, track, new_rect, missing_seconds=0.0, now=None):
        if now is None:
            now = time.time()

        label = track.get("label")

        if not label or label == "unknown":
            return True

        moved_px = center_distance(track["rect"], new_rect)

        # Only treat this as a possible replacement if it was genuinely missing
        # longer than the normal scan cadence. A too-short value here caused the
        # old recognized -> unknown -> recognized flicker.
        if missing_seconds > TRACK_REUSE_AFTER_MISSING_SECONDS:
            return True

        if moved_px <= TRACK_REACQUIRE_CENTER_THRESHOLD_PX:
            return False

        if moved_px >= TRACK_REQUEUE_CENTER_THRESHOLD_PX:
            return True

        return False

    def update_or_create_track(self, side, box):
        now = time.time()
        self.expire_old_tracks(side, now)

        rect = box_to_rect(box)
        track = self.match_track(side, box)

        if track is None:
            track = {
                "track_id": self.next_track_id,
                "side": side,
                "box": box,
                "rect": rect,
                "label": None,
                "score": 0.0,
                "margin": None,
                "rotation": "?",
                "last_seen_at": now,
                "last_processed_at": 0.0,
                "stationary": False,
                "displayed": False,
                "queued": False,
                "refined_box": None,
                "proposal_box": box,
                "detector_sharpness": None,
                "detector_confidence": None,
                "used_fallback": False,
                "refine_reason": None,
                "detail_metrics": None,
                "visual_signature": None,
                "missing_seconds": 0.0,
                "unknown_streak": 0,
                "force_new_identification": False,
                "needs_same_spot_signature_check": False,
                "last_decision": "new_track",
                "last_moved_rescan_at": 0.0,
                "last_visual_recheck_at": 0.0,
                "last_signature_distance": None,
                "last_raw_signature": None,
                "pending_raw_signature": None,
                "last_raw_visual_diff": None,
                "last_raw_visual_diff_at": 0.0,
                "raw_visual_change_pending": False,
                "raw_visual_change_count": 0,
                "raw_visual_change_started_at": 0.0,
                "pending_replacement_label": None,
                "pending_replacement_count": 0,
                "pending_replacement_started_at": 0.0,
                "last_alternatives": [],
                "forced_visual_change_at": 0.0,
                "pending_card_back_count": 0,
                "pending_card_back_started_at": 0.0,
            }
            self.next_track_id += 1
            self.tracks[side].append(track)
            log_event(side, track["track_id"], "new_track")
            return track, True

        previous_rect = track["rect"]
        missing_seconds = now - track["last_seen_at"]
        moved_px = center_distance(previous_rect, rect)
        needs_processing = self.should_reprocess(
            track,
            rect,
            missing_seconds=missing_seconds,
            now=now,
        )

        track["box"] = box
        track["rect"] = rect
        track["last_seen_at"] = now
        track["missing_seconds"] = missing_seconds
        track["stationary"] = moved_px <= self.stationary_center_threshold_px
        track["proposal_box"] = box

        moved_replacement_like = moved_px >= TRACK_REQUEUE_CENTER_THRESHOLD_PX
        stale_same_spot_check = (
            missing_seconds > TRACK_REUSE_AFTER_MISSING_SECONDS
            and moved_px < TRACK_REQUEUE_CENTER_THRESHOLD_PX
        )

        moved_sanity_check = (
            MOVED_SANITY_RESCAN_ENABLED
            and moved_px >= MOVED_SANITY_RESCAN_PX
            and now - float(track.get("last_moved_rescan_at", 0.0)) >= MOVED_SANITY_RESCAN_COOLDOWN_SECONDS
        )

        moved_reidentify_check = (
            MOVED_REIDENTIFY_ENABLED
            and moved_px >= MOVED_REIDENTIFY_MIN_PX
            and now - float(track.get("last_moved_rescan_at", 0.0)) >= MOVED_REIDENTIFY_COOLDOWN_SECONDS
        )

        if moved_replacement_like or moved_sanity_check or moved_reidentify_check:
            # Keep the old overlay visible while the fresh ID is pending. Clearing
            # it here made normal movement look like the track had dropped.
            track["force_new_identification"] = True
            track["needs_same_spot_signature_check"] = False
            track["last_moved_rescan_at"] = now
            if moved_replacement_like:
                kind = "moved_replacement_check"
            elif moved_sanity_check:
                kind = "moved_sanity_rescan"
            else:
                kind = "moved_reidentify"
            track["last_decision"] = f"{kind} missing={missing_seconds:.2f} moved={moved_px:.1f}"
            log_event(
                side,
                track["track_id"],
                "replacement_check",
                label=track.get("label"),
                missing=f"{missing_seconds:.2f}",
                moved=f"{moved_px:.1f}",
                kind=kind,
            )
            log_human_event(
                side,
                track["track_id"],
                "replacement_check",
                label=track.get("label"),
                missing=f"{missing_seconds:.2f}",
                moved=f"{moved_px:.1f}",
            )
        elif stale_same_spot_check:
            # A side may simply not have been scanned for a while. Do not drop
            # the old ID yet. Recognition will compare a lightweight visual
            # signature and only refresh the ID if the crop is genuinely different.
            track["force_new_identification"] = False
            track["needs_same_spot_signature_check"] = True
            track["last_decision"] = f"same_spot_verify missing={missing_seconds:.2f} moved={moved_px:.1f}"
            log_event(
                side,
                track["track_id"],
                "same_spot_verify",
                label=track.get("label"),
                missing=f"{missing_seconds:.2f}",
                moved=f"{moved_px:.1f}",
            )
        elif not needs_processing and track.get("label") not in (None, "", "unknown", "tracking", "card_back"):
            if not track.get("queued", False):
                track["displayed"] = True
            track["last_decision"] = f"reuse_known missing={missing_seconds:.2f} moved={moved_px:.1f}"
        else:
            track["last_decision"] = f"needs_processing missing={missing_seconds:.2f} moved={moved_px:.1f}"

        return track, needs_processing

    def apply_recognition_result(self, track, best):
        now = time.time()
        old_label = track.get("label")

        if best is None:
            track["label"] = "unknown"
            track["score"] = 0.0
            track["margin"] = None
            track["rotation"] = "?"
            track["used_fallback"] = True
            track["refine_reason"] = "no_match"
            track["last_decision"] = "applied_unknown_no_match"
        else:
            track["label"] = best["id"]
            track["score"] = float(best["score"])
            track["margin"] = best.get("margin")
            track["rotation"] = best.get("rotation", "?")

            proposal_box = best.get("proposal_box")
            if proposal_box is not None:
                track["proposal_box"] = proposal_box
            elif track.get("proposal_box") is not None:
                track["proposal_box"] = track.get("proposal_box")
            else:
                track["proposal_box"] = track["box"]

            track["detector_sharpness"] = best.get("sharpness")
            track["detector_confidence"] = best.get("confidence")
            track["used_fallback"] = best.get("used_fallback", False)
            track["refine_reason"] = best.get("refine_reason")
            track["detail_metrics"] = best.get("detail_metrics")
            track["last_decision"] = f"applied_{best.get('id')} score={float(best.get('score', 0.0)):.2f}"

            if best.get("visual_signature") is not None:
                track["visual_signature"] = best.get("visual_signature")

            new_refined = best.get("refined_box")
            if new_refined is not None:
                track["refined_box"] = smooth_box(
                    old_box=track.get("refined_box"),
                    new_box=new_refined,
                    alpha=self.refined_box_smoothing_alpha,
                )
            elif track["used_fallback"]:
                track["refined_box"] = None

        if old_label != track.get("label"):
            log_event(
                track.get("side"),
                track.get("track_id"),
                "label_change",
                old=old_label,
                new=track.get("label"),
                reason=track.get("refine_reason"),
                score=f"{track.get('score', 0.0):.2f}",
            )
            log_human_event(
                track.get("side"),
                track.get("track_id"),
                "label_change",
                old=old_label,
                new=track.get("label"),
                reason=track.get("refine_reason"),
                score=f"{track.get('score', 0.0):.2f}",
            )

        track["last_processed_at"] = now

    def get_visible_matches(self, side):
        now = time.time()
        self.expire_old_tracks(side, now)
        matches = []

        for track in self.tracks[side]:
            label = track.get("label")

            # Do not draw uncertain bookkeeping tracks. These are useful internally
            # for retry/cooldown logic, but displaying them creates stale-looking
            # boxes after a card has been picked up or moved.
            if HIDE_UNCONFIRMED_UNKNOWN_TRACKS and label in (None, "", "unknown", "tracking"):
                continue

            # Card-back reads are intentionally conservative because smooth
            # playmat/table patches are easy false positives. Keep them out of
            # the overlay unless explicitly re-enabled in config.py.
            if HIDE_CARD_BACK_TRACKS and label == "card_back":
                continue

            visible_missing_seconds = (
                CARD_BACK_VISIBLE_MISSING_SECONDS
                if label == "card_back"
                else TRACK_VISIBLE_MISSING_SECONDS
            )
            if now - track["last_seen_at"] > visible_missing_seconds:
                continue

            active_box = track.get("refined_box") if (
                track.get("refined_box") is not None and not track.get("used_fallback")
            ) else track["box"]

            proposal_box = track.get("proposal_box")
            if proposal_box is None:
                proposal_box = track["box"]

            matches.append({
                "box": active_box,
                "proposal_box": proposal_box,
                "refined_box": track.get("refined_box"),
                "label": track.get("label") or "tracking",
                "score": float(track.get("score", 0.0)),
                "margin": track.get("margin"),
                "rotation": track.get("rotation", "?"),
                "track_id": track["track_id"],
                "stationary": track.get("stationary", False),
                "displayed": track.get("displayed", False),
                "queued": track.get("queued", False),
                "detector_sharpness": track.get("detector_sharpness"),
                "detector_confidence": track.get("detector_confidence"),
                "used_fallback": track.get("used_fallback", False),
                "refine_reason": track.get("refine_reason"),
                "detail_metrics": track.get("detail_metrics"),
                "missing_seconds": track.get("missing_seconds", 0.0),
                "unknown_streak": track.get("unknown_streak", 0),
                "force_new_identification": track.get("force_new_identification", False),
                "last_decision": track.get("last_decision", ""),
                "needs_same_spot_signature_check": track.get("needs_same_spot_signature_check", False),
                "last_moved_rescan_at": track.get("last_moved_rescan_at", 0.0),
                "last_visual_recheck_at": track.get("last_visual_recheck_at", 0.0),
                "last_signature_distance": track.get("last_signature_distance"),
                "last_raw_visual_diff": track.get("last_raw_visual_diff"),
                "raw_visual_change_pending": track.get("raw_visual_change_pending", False),
                "raw_visual_change_count": track.get("raw_visual_change_count", 0),
                "pending_replacement_label": track.get("pending_replacement_label"),
                "pending_replacement_count": track.get("pending_replacement_count", 0),
                "forced_visual_change_at": track.get("forced_visual_change_at", 0.0),
            })

        return matches
