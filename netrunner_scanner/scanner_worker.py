
import queue
import threading
import copy

from .config import (
    SCANNER_WORKER_MAX_QUEUE_SIZE,
    SCANNER_WORKER_DROP_DUPLICATE_SIDE_REQUESTS,
    SIDE_SCAN_COOLDOWN_SECONDS,
)
from .recognition import scan_side_for_matches, scan_point_for_card, scan_box_for_card
from .perf import now, record, increment
from .console_utils import print_error
from .scan_diagnostics import log as diag_log, log_exception as diag_exception, log_throttled


class ScannerWorker:
    def __init__(self, catalog):
        self.catalog = catalog
        self.jobs = queue.Queue(maxsize=SCANNER_WORKER_MAX_QUEUE_SIZE)
        self.pending_sides = set()
        self.last_accepted_side_scan = {"left": 0.0, "right": 0.0}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="ScannerWorkerRecognition", daemon=True)
        diag_log("worker_created", max_queue=SCANNER_WORKER_MAX_QUEUE_SIZE, cooldown_seconds=SIDE_SCAN_COOLDOWN_SECONDS)
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        diag_log("worker_start")
        self.thread.start()

    def stop(self):
        self.running = False
        diag_log("worker_stop_requested", queue_size=self.jobs.qsize())
        try:
            self.jobs.put_nowait(("stop", None))
        except queue.Full:
            pass
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
                diag_log("worker_stop_joined", alive=self.thread.is_alive())
        except RuntimeError:
            pass

    def queue_size(self):
        return self.jobs.qsize()

    def request_side_scan(self, frame, roi, side):
        side = str(side)
        current = now()

        with self.lock:
            if current - self.last_accepted_side_scan.get(side, 0.0) < SIDE_SCAN_COOLDOWN_SECONDS:
                increment("dropped_jobs")
                log_throttled(f"side_scan_cooldown_{side}", 2.0, "side_scan_rejected_cooldown", side=side, queue_size=self.jobs.qsize(), elapsed_seconds=current - self.last_accepted_side_scan.get(side, 0.0))
                return False

            if SCANNER_WORKER_DROP_DUPLICATE_SIDE_REQUESTS and side in self.pending_sides:
                increment("dropped_jobs")
                log_throttled(f"side_scan_duplicate_{side}", 2.0, "side_scan_rejected_duplicate", side=side, queue_size=self.jobs.qsize(), pending=list(self.pending_sides))
                return False

            self.pending_sides.add(side)
            self.last_accepted_side_scan[side] = current

        # Do not copy unless we actually accept the job. This avoids repeated
        # 1080p frame copies during motion bursts.
        job = {
            "frame": frame.copy(),
            "roi": copy.deepcopy(roi),
            "side": side,
        }

        try:
            self.jobs.put_nowait(("side", job))
            diag_log("side_scan_enqueued", side=side, queue_size=self.jobs.qsize(), roi=str(job.get("roi"))[:240], frame_shape=str(getattr(frame, "shape", None)))
            return True
        except queue.Full:
            with self.lock:
                self.pending_sides.discard(side)
            increment("dropped_jobs")
            diag_log("side_scan_rejected_queue_full", side=side, queue_size=self.jobs.qsize())
            return False

    def request_point_scan(self, frame, x, y, side):
        job = {
            "frame": frame.copy(),
            "x": int(x),
            "y": int(y),
            "side": str(side),
        }

        try:
            self.jobs.put_nowait(("point", job))
            return True
        except queue.Full:
            increment("dropped_jobs")
            return False


    def request_box_scan(self, frame, x1, y1, x2, y2, side):
        job = {
            "frame": frame.copy(),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "side": str(side),
        }

        try:
            self.jobs.put_nowait(("box", job))
            return True
        except queue.Full:
            increment("dropped_jobs")
            return False

    def _run(self):
        while self.running:
            kind, job = self.jobs.get()

            if kind == "stop":
                diag_log("worker_stop_job_received")
                return

            start = now()
            diag_log("worker_job_start", kind=kind, side=(job or {}).get("side"), queue_size=self.jobs.qsize())

            try:
                if kind == "side":
                    side = job["side"]
                    try:
                        scan_side_for_matches(job["frame"], job["roi"], side, self.catalog)
                    finally:
                        elapsed_ms = (now() - start) * 1000.0
                        if side == "left":
                            record("worker_left_ms", elapsed_ms)
                        else:
                            record("worker_right_ms", elapsed_ms)
                        with self.lock:
                            self.pending_sides.discard(side)
                        diag_log("worker_side_finish", side=side, elapsed_ms=round(elapsed_ms, 1), queue_size=self.jobs.qsize())

                elif kind == "point":
                    scan_point_for_card(
                        job["frame"],
                        job["x"],
                        job["y"],
                        job["side"],
                        self.catalog,
                    )
                    record("worker_point_ms", (now() - start) * 1000.0)

                elif kind == "box":
                    scan_box_for_card(
                        job["frame"],
                        job["x1"],
                        job["y1"],
                        job["x2"],
                        job["y2"],
                        job["side"],
                        self.catalog,
                    )
                    record("worker_point_ms", (now() - start) * 1000.0)

                increment("processed_jobs")
                record("queued_jobs", self.jobs.qsize())

            except Exception as exc:
                diag_exception("worker_job_exception", exc, kind=kind, side=(job or {}).get("side"), queue_size=self.jobs.qsize())
                print_error(f"Scanner worker error during {kind} scan: {exc}", exc)
                if kind == "side":
                    elapsed_ms = (now() - start) * 1000.0
                    side = job.get("side")
                    if side == "left":
                        record("worker_left_ms", elapsed_ms)
                    elif side == "right":
                        record("worker_right_ms", elapsed_ms)
                    with self.lock:
                        self.pending_sides.discard(side)
