import time
from collections import deque
import cv2
import numpy as np

from netrunner_scanner.config import (
    APP_NAME,
    APP_VERSION,
    AUTO_SCAN_ENABLED,
    AUTO_SCAN_INTERVAL_SECONDS,
    MOTION_GATING_ENABLED,
    OVERLAY_SERVER_ENABLED,
    GUI_STATUS_SIDEBAR_ENABLED,
    GUI_DISPLAY_SCALE,
    GUI_TARGET_FPS,
    OPENCV_NUM_THREADS,
    SCANNER_WORKER_ENABLED,
)
from netrunner_scanner.camera import open_camera, open_camera_by_index, get_camera_capture_info, format_camera_info
from netrunner_scanner.catalog import CardCatalog
from netrunner_scanner.roi import rois, load_rois, save_rois, default_rois, roi_enabled, toggle_roi_edit_enabled
from netrunner_scanner.recognition import latest_matches, scan_side_for_matches, tick_obs_queue, get_scanner_status
from netrunner_scanner.drawing import draw_roi, draw_card_matches
from netrunner_scanner.motion import MotionGate
from netrunner_scanner.overlay_server import start_overlay_server
from netrunner_scanner.status_panel import make_status_sidebar, get_last_controls
from netrunner_scanner.card_actions import (
    make_mouse_handler,
    draw_card_menu,
    draw_manual_drag_box,
    draw_manual_choice_overlay,
    draw_track_selection,
    handle_card_menu_key,
    clear_all_tracks,
    delete_selected_tracks,
    rescan_selected_tracks,
)
from netrunner_scanner.scanner_worker import ScannerWorker
from netrunner_scanner.settings_window import NetReadyEyesWindow
from netrunner_scanner.perf import now as perf_now, record as perf_record, start_thread_cpu_monitor
from netrunner_scanner.console_utils import print_error
from netrunner_scanner.runtime_controls import load_settings, gpu_enabled, get_settings
from netrunner_scanner.gpu_status import configure_gpu
from netrunner_scanner.scan_diagnostics import init_diagnostics, log as diag_log, log_exception as diag_exception, log_throttled, close_diagnostics
from netrunner_scanner.direct_diagnostics import run_direct_diagnostic, format_summary


def main():
    log_path = init_diagnostics()
    print(f"Diagnostics log: {log_path}")
    diag_log("app_main_start")
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
    except cv2.error:
        pass

    load_settings()
    diag_log("runtime_settings_loaded", settings=get_settings())
    configure_gpu(gpu_enabled())
    start_thread_cpu_monitor()

    if OVERLAY_SERVER_ENABLED:
        start_overlay_server()

    cap = open_camera()
    video_source_meta = get_camera_capture_info(cap)
    diag_log("camera_opened", video_source=format_camera_info(video_source_meta))
    catalog = CardCatalog("netrunner-catalog.npz")
    diag_log("catalog_loaded", catalog_path="netrunner-catalog.npz", catalog_type=type(catalog).__name__)

    worker = ScannerWorker(catalog) if SCANNER_WORKER_ENABLED else None
    if worker is not None:
        worker.start()

    initialized = False
    app = None
    key_queue = []
    should_quit = [False]

    last_auto_scan_time = 0.0
    next_scan_side = "left"
    target_frame_seconds = 1.0 / max(float(GUI_TARGET_FPS), 1.0)
    current_frame = [None]
    current_frame_size = [0, 0]
    frame_time_samples = deque(maxlen=45)
    last_frame_read_time = [None]
    measured_source_fps = [0.0]
    motion_gate = MotionGate()

    def _format_video_metadata():
        width = int(video_source_meta.get("width") or current_frame_size[0] or 0)
        height = int(video_source_meta.get("height") or current_frame_size[1] or 0)
        reported_fps = float(video_source_meta.get("fps") or 0.0)
        live_fps = float(measured_source_fps[0] or 0.0)
        fps = live_fps if live_fps > 0 else reported_fps
        if width > 0 and height > 0 and fps > 0:
            return f"Video source: {width}x{height}  {fps:.1f} fps"
        if width > 0 and height > 0:
            return f"Video source: {width}x{height}"
        return "Video source metadata unavailable"

    def _draw_video_metadata(frame):
        text = _format_video_metadata()
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x = 10
        pad_y = 8
        x2 = w - 10
        y2 = h - 10
        x1 = max(0, x2 - tw - pad_x * 2)
        y1 = max(0, y2 - th - baseline - pad_y * 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y - baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def get_current_frame():
        return current_frame[0]

    def get_current_frame_size():
        return current_frame_size[0], current_frame_size[1]

    def submit_box_scan(frame, x1, y1, x2, y2, side):
        if worker is not None:
            worker.request_box_scan(frame, x1, y1, x2, y2, side)
        else:
            from netrunner_scanner.recognition import scan_box_for_card
            scan_box_for_card(frame, x1, y1, x2, y2, side, catalog)

    def submit_point_scan(frame, x, y, side):
        if worker is not None:
            worker.request_point_scan(frame, x, y, side)
        else:
            from netrunner_scanner.recognition import scan_point_for_card
            scan_point_for_card(frame, x, y, side, catalog)

    def _display_box_to_raw(x1, y1, x2, y2, frame):
        scale = float(GUI_DISPLAY_SCALE) if GUI_DISPLAY_SCALE else 1.0
        raw_x1 = int(round(float(x1) / scale))
        raw_y1 = int(round(float(y1) / scale))
        raw_x2 = int(round(float(x2) / scale))
        raw_y2 = int(round(float(y2) / scale))

        frame_h, frame_w = frame.shape[:2]
        raw_x1 = max(0, min(raw_x1, frame_w))
        raw_x2 = max(0, min(raw_x2, frame_w))
        raw_y1 = max(0, min(raw_y1, frame_h))
        raw_y2 = max(0, min(raw_y2, frame_h))
        return raw_x1, raw_y1, raw_x2, raw_y2

    def _run_direct_diagnostic_raw(frame, raw_x1, raw_y1, raw_x2, raw_y2, save_artifacts=True, announce=True):
        frame_h, frame_w = frame.shape[:2]
        cx = (raw_x1 + raw_x2) / 2.0
        side = "left" if cx < frame_w / 2.0 else "right"
        if announce:
            print(f"\nDirected CollectorVision diagnostic scan: {side} region=({raw_x1}, {raw_y1})-({raw_x2}, {raw_y2})")
        try:
            report = run_direct_diagnostic(
                get_current_frame,
                frame.copy(),
                raw_x1, raw_y1, raw_x2, raw_y2,
                side,
                catalog,
                save_artifacts=save_artifacts,
                live_tag="live_direct_cv",
            )
        except Exception as exc:
            diag_exception("direct_diagnostic_exception", exc)
            print_error("Directed diagnostic failed", exc)
            return None
        if announce:
            print(format_summary(report.get("summary") or {}))
            print("Note: the diagnostic recognition path tests 0/90/180/270 degree rotations of the dewarped crop.")
            print(f"Diagnostic files: {report.get('overlay')} and {report.get('source_crop')}")
        if app is not None:
            app.show_direct_diagnostic_report(report)
        return report

    def run_direct_diagnostic_from_display_coords(x1, y1, x2, y2):
        frame = current_frame[0]
        if frame is None:
            print("No current frame for diagnostic scan.")
            return
        raw = _display_box_to_raw(x1, y1, x2, y2, frame)
        _run_direct_diagnostic_raw(frame, *raw, save_artifacts=False, announce=True)

    def save_final_direct_diagnostic_from_region(region):
        frame = current_frame[0]
        if frame is None:
            print("No current frame available to save final directed diagnostic frame.")
            return
        raw = _display_box_to_raw(*region, frame)
        print("Saving final directed CollectorVision diagnostic frame...")
        _run_direct_diagnostic_raw(frame, *raw, save_artifacts=True, announce=True)

    def scan_side(side):
        frame = current_frame[0]
        if frame is None:
            diag_log("scan_request_skipped_no_frame", side=side)
            return False
        if not roi_enabled(rois.get(side)):
            diag_log("scan_request_skipped_roi_disabled", side=side, roi=str(rois.get(side))[:240])
            return False
        diag_log("scan_request", side=side, worker_enabled=worker is not None, frame_shape=str(getattr(frame, "shape", None)), roi=str(rois.get(side))[:240])
        if worker is not None:
            accepted = worker.request_side_scan(frame, rois[side], side)
            diag_log("scan_request_worker_result", side=side, accepted=bool(accepted), queue_size=worker.queue_size())
            return accepted
        else:
            scan_side_for_matches(frame, rois[side], side, catalog)
            return True

    def on_key(value):
        key_queue.append(value)

    def on_quit():
        should_quit[0] = True

    def load_catalog(path):
        nonlocal catalog, worker
        print(f"\nLoading catalog: {path}")
        new_catalog = CardCatalog(path)
        catalog = new_catalog
        # Mouse/action callbacks keep their catalog in card_actions.runtime; update it
        # so manual rescans use the newly loaded catalog immediately.
        from netrunner_scanner.card_actions import runtime as card_action_runtime
        card_action_runtime["catalog"] = catalog
        if worker is not None:
            worker.stop()
        worker = ScannerWorker(catalog) if SCANNER_WORKER_ENABLED else None
        if worker is not None:
            worker.start()
        print("Catalog loaded.")

    def switch_camera(index):
        nonlocal cap, video_source_meta
        print(f"\nSwitching video source to index {index}...")
        new_cap = open_camera_by_index(index)
        try:
            cap.release()
        except Exception:
            pass
        cap = new_cap
        video_source_meta = get_camera_capture_info(cap)
        frame_time_samples.clear()
        last_frame_read_time[0] = None
        measured_source_fps[0] = 0.0
        print(f"Active video source: {format_camera_info(video_source_meta)}")

    def process_key(value):
        if value is None:
            return

        if value in ("Escape", "esc"):
            should_quit[0] = True
            return

        if value in ("Delete", "BackSpace"):
            delete_selected_tracks()
            return

        if len(value) == 1:
            key = ord(value.lower())
        else:
            key = 0

        # First give overlays / card context menus a chance to consume text keys.
        if key and handle_card_menu_key(key):
            return

        if value.lower() == "q":
            scan_side("left")
        elif value.lower() == "e":
            scan_side("right")
        elif value.lower() == "c":
            clear_all_tracks()
        elif value.lower() == "m":
            rescan_selected_tracks()
        elif value.lower() == "i":
            enabled = toggle_roi_edit_enabled()
            print("\nROI edit mode on." if enabled else "\nROI edit mode off.")
        elif value.lower() == "s":
            save_rois()
        elif value.lower() == "l":
            frame_width, frame_height = get_current_frame_size()
            loaded = load_rois(frame_width, frame_height)
            rois["left"] = loaded["left"]
            rois["right"] = loaded["right"]
            print("\nLoaded ROI settings.")
        elif value.lower() == "r":
            frame_width, frame_height = get_current_frame_size()
            rois.update(default_rois(frame_width, frame_height))
            print("\nReset ROIs to default left/right split.")

    print(f"\n{APP_NAME} v{APP_VERSION}")
    print("Controls:")
    print("File > Load Card Catalog = switch CollectorVision/Net Ready Eyes catalog")
    print("Edit > Video Source = switch camera/video input")
    print("Edit > Settings > ROI = ROI edit mode, colors, square/reset, second ROI")
    print("Normal left-drag = select tracks")
    print("Delete = delete selected tracks")
    print("M = manually rescan selected tracks")
    print("Right-click a card track = actions menu, including manual rescan")
    print("Automatic scanning runs continuously on enabled playmat ROIs")
    print("C = clear all tracks on both playmats")
    print("ESC = quit\n")

    try:
        while not should_quit[0]:
            gui_start = perf_now()
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame")
                break

            read_time = time.time()
            if last_frame_read_time[0] is not None:
                dt = read_time - last_frame_read_time[0]
                if dt > 0:
                    frame_time_samples.append(dt)
                    total = sum(frame_time_samples)
                    if total > 0:
                        measured_source_fps[0] = len(frame_time_samples) / total
            last_frame_read_time[0] = read_time

            frame_height, frame_width = frame.shape[:2]
            current_frame[0] = frame
            current_frame_size[0] = frame_width
            current_frame_size[1] = frame_height

            if not initialized:
                print("\nFrame shape:", frame.shape)
                diag_log("first_frame", frame_shape=str(frame.shape), frame_width=frame_width, frame_height=frame_height)

                loaded = load_rois(frame_width, frame_height)
                rois["left"] = loaded["left"]
                rois["right"] = loaded["right"]
                diag_log("rois_loaded", left=str(rois["left"]), right=str(rois["right"]))

                mouse_handler = make_mouse_handler(
                    get_current_frame_size,
                    get_current_frame,
                    catalog,
                    submit_point_scan,
                    submit_box_scan,
                    get_last_controls,
                )

                app = NetReadyEyesWindow(
                    frame_size_getter=get_current_frame_size,
                    mouse_handler=mouse_handler,
                    load_catalog_callback=load_catalog,
                    switch_camera_callback=switch_camera,
                    clear_tracks_callback=clear_all_tracks,
                    scan_side_callback=scan_side,
                    diagnostic_scan_callback=run_direct_diagnostic_from_display_coords,
                    diagnostic_stop_callback=save_final_direct_diagnostic_from_region,
                    save_roi_callback=save_rois,
                    key_callback=on_key,
                    quit_callback=on_quit,
                )
                app.start()
                initialized = True

            current_time = time.time()

            if AUTO_SCAN_ENABLED and current_time - last_auto_scan_time >= AUTO_SCAN_INTERVAL_SECONDS:
                # Recognition should be automatic and dependable.  Motion gating is
                # intentionally not allowed to suppress the only scan request path; it
                # can be reintroduced later as an optimization once the core pipeline is
                # verified.  Scan one enabled playmat per interval to avoid spikes.
                side_to_scan = next_scan_side
                next_scan_side = "right" if next_scan_side == "left" else "left"

                enabled = roi_enabled(rois.get(side_to_scan))
                log_throttled(
                    f"auto_scan_tick_{side_to_scan}",
                    2.0,
                    "auto_scan_tick",
                    side=side_to_scan,
                    enabled=enabled,
                    worker_queue=worker.queue_size() if worker is not None else 0,
                )
                if enabled:
                    scan_side(side_to_scan)

                last_auto_scan_time = current_time

            tick_obs_queue()

            if app is not None and app.live_direct_diagnostic_due(current_time):
                region = app.get_active_direct_diagnostic_region()
                if region is not None:
                    raw = _display_box_to_raw(*region, frame)
                    _run_direct_diagnostic_raw(frame, *raw, save_artifacts=False, announce=False)

            display_frame = frame.copy()
            draw_roi(display_frame, "left", rois["left"])
            draw_roi(display_frame, "right", rois["right"])
            draw_card_matches(display_frame, latest_matches["left"])
            draw_card_matches(display_frame, latest_matches["right"])
            draw_manual_drag_box(display_frame)
            draw_track_selection(display_frame)
            draw_manual_choice_overlay(display_frame)
            draw_card_menu(display_frame)

            if GUI_DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(
                    display_frame,
                    None,
                    fx=GUI_DISPLAY_SCALE,
                    fy=GUI_DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            _draw_video_metadata(display_frame)

            if GUI_STATUS_SIDEBAR_ENABLED:
                sidebar = make_status_sidebar(display_frame.shape[0], get_scanner_status())
                display_frame = np.hstack([display_frame, sidebar])

            if app is not None and not app.update_frame(display_frame):
                break

            while key_queue:
                process_key(key_queue.pop(0))

            gui_elapsed = perf_now() - gui_start
            perf_record("gui_frame_ms", gui_elapsed * 1000.0)

            remaining = target_frame_seconds - gui_elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        diag_log("app_shutdown_start")
        if worker is not None:
            worker.stop()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        diag_log("app_shutdown_complete")
        close_diagnostics()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        diag_exception("fatal_scanner_error", exc)
        print_error("Fatal scanner error", exc)
        raise
