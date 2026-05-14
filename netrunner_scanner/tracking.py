
import time
import cv2
import numpy as np


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

    def expire_old_tracks(self, side, now=None):
        if now is None:
            now = time.time()

        self.tracks[side] = [
            track for track in self.tracks[side]
            if now - track["last_seen_at"] <= self.expire_seconds
        ]

    def match_track(self, side, box):
        rect = box_to_rect(box)
        best_track = None
        best_iou = 0.0

        for track in self.tracks[side]:
            score = rect_iou(rect, track["rect"])

            if score > best_iou:
                best_iou = score
                best_track = track

        if best_track is not None and best_iou >= self.iou_threshold:
            return best_track

        return None

    def should_reprocess(self, track, new_rect, now=None):
        if now is None:
            now = time.time()

        if not track.get("label") or track.get("label") == "unknown":
            return True

        moved_px = center_distance(track["rect"], new_rect)

        if moved_px > self.stationary_center_threshold_px:
            return True

        if track.get("label") == "card_back":
            return False

        if now - track.get("last_processed_at", 0.0) >= self.stationary_refresh_seconds:
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
            }
            self.next_track_id += 1
            self.tracks[side].append(track)
            return track, True

        moved_px = center_distance(track["rect"], rect)
        needs_processing = self.should_reprocess(track, rect, now)

        track["box"] = box
        track["rect"] = rect
        track["last_seen_at"] = now
        track["stationary"] = moved_px <= self.stationary_center_threshold_px

        if moved_px > self.stationary_center_threshold_px:
            track["displayed"] = False
            track["queued"] = False
            track["refined_box"] = None

        return track, needs_processing

    def apply_recognition_result(self, track, best):
        now = time.time()

        if best is None:
            track["label"] = "unknown"
            track["score"] = 0.0
            track["margin"] = None
            track["rotation"] = "?"
            track["used_fallback"] = True
            track["refine_reason"] = "no_match"
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

            new_refined = best.get("refined_box")
            if new_refined is not None:
                track["refined_box"] = smooth_box(
                    old_box=track.get("refined_box"),
                    new_box=new_refined,
                    alpha=self.refined_box_smoothing_alpha,
                )
            elif track["used_fallback"]:
                track["refined_box"] = None

        track["last_processed_at"] = now

    def get_visible_matches(self, side):
        now = time.time()
        self.expire_old_tracks(side, now)

        matches = []

        for track in self.tracks[side]:
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
            })

        return matches
