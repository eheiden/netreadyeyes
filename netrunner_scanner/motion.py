
import time
import cv2
import numpy as np

from .roi import dewarp_roi_for_scan

from .config import (
    MOTION_DOWNSAMPLE_WIDTH,
    MOTION_SCAN_THRESHOLD,
    MOTION_FORCE_SCAN_SECONDS,
)


class MotionGate:
    def __init__(self):
        self.previous = {
            "left": None,
            "right": None,
        }
        self.last_scan_time = {
            "left": 0.0,
            "right": 0.0,
        }

    def _prepare_roi(self, frame, roi):
        scan_frame, scan_roi, _matrix, _dewarped = dewarp_roi_for_scan(frame, roi)

        if scan_roi is None or scan_frame is None or scan_frame.size == 0:
            return None

        x, y, w, h = [int(v) for v in scan_roi]
        if w <= 0 or h <= 0:
            return None

        crop = scan_frame[y:y + h, x:x + w]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        scale = MOTION_DOWNSAMPLE_WIDTH / max(1, gray.shape[1])
        new_w = MOTION_DOWNSAMPLE_WIDTH
        new_h = max(1, int(gray.shape[0] * scale))

        small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        return small

    def should_scan(self, frame, roi, side):
        now = time.time()
        prepared = self._prepare_roi(frame, roi)

        if prepared is None:
            return True, "empty_roi"

        previous = self.previous.get(side)

        self.previous[side] = prepared

        if previous is None:
            self.last_scan_time[side] = now
            return True, "first_frame"

        if previous.shape != prepared.shape:
            self.last_scan_time[side] = now
            return True, "shape_changed"

        diff = cv2.absdiff(previous, prepared)
        changed = np.count_nonzero(diff > 18) / diff.size

        if changed >= MOTION_SCAN_THRESHOLD:
            self.last_scan_time[side] = now
            return True, f"motion:{changed:.4f}"

        if now - self.last_scan_time[side] >= MOTION_FORCE_SCAN_SECONDS:
            self.last_scan_time[side] = now
            return True, "periodic_refresh"

        return False, f"still:{changed:.4f}"
