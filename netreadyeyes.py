
import time
import cv2
import numpy as np

from netrunner_scanner.config import (
    WINDOW_NAME,
    APP_NAME,
    APP_VERSION,
    AUTO_SCAN_ENABLED,
    AUTO_SCAN_INTERVAL_SECONDS,
    MOTION_GATING_ENABLED,
    OVERLAY_SERVER_ENABLED,
    GUI_STATUS_SIDEBAR_ENABLED,
    GUI_WINDOW_RESIZABLE,
    GUI_LETTERBOX_RESIZED_WINDOW,
    SCANNER_WORKER_ENABLED,
    GUI_DISPLAY_SCALE,
    GUI_TARGET_FPS,
    OPENCV_NUM_THREADS,
    EXIT_ON_WINDOW_CLOSE,
)
from netrunner_scanner.camera import open_camera
from netrunner_scanner.catalog import CardCatalog
from netrunner_scanner.roi import rois, load_rois, save_rois, default_rois
from netrunner_scanner.recognition import latest_matches, scan_side_for_matches, tick_obs_queue, get_scanner_status
from netrunner_scanner.drawing import draw_roi, draw_card_matches
from netrunner_scanner.motion import MotionGate
from netrunner_scanner.overlay_server import start_overlay_server
from netrunner_scanner.status_panel import make_status_sidebar, get_last_controls
from netrunner_scanner.display_utils import fit_image_to_window
from netrunner_scanner.card_actions import make_mouse_handler, draw_card_menu, draw_manual_drag_box, draw_manual_choice_overlay, handle_card_menu_key
from netrunner_scanner.scanner_worker import ScannerWorker
from netrunner_scanner.perf import now as perf_now, record as perf_record, start_thread_cpu_monitor
from netrunner_scanner.console_utils import print_error
from netrunner_scanner.runtime_controls import load_settings, gpu_enabled
from netrunner_scanner.gpu_status import configure_gpu


def main():
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
    except cv2.error:
        pass

    load_settings()
    configure_gpu(gpu_enabled())

    start_thread_cpu_monitor()

    if OVERLAY_SERVER_ENABLED:
        start_overlay_server()

    cap = open_camera()
    catalog = CardCatalog("netrunner-catalog.npz")

    worker = ScannerWorker(catalog) if SCANNER_WORKER_ENABLED else None
    if worker is not None:
        worker.start()

    window_mode = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO if GUI_WINDOW_RESIZABLE else cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(WINDOW_NAME, window_mode)

    initialized = False
    last_auto_scan_time = 0.0
    next_scan_side = "left"
    target_frame_seconds = 1.0 / max(float(GUI_TARGET_FPS), 1.0)
    current_frame = [None]
    current_frame_size = [0, 0]
    motion_gate = MotionGate()

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

    print(f"\\n{APP_NAME} v{APP_VERSION}")
    print("Controls:")
    print("Mouse drag inside box = move ROI")
    print("Mouse drag edge/corner = resize ROI")
    print("Left-click card = force to OBS front")
    print("Right-click card = actions menu")
    print("Left-click/drag empty ROI area = manual box scan")
    print("Q = manually scan LEFT playmat")
    print("E = manually scan RIGHT playmat")
    print("S = save ROI settings")
    print("L = load ROI settings")
    print("R = reset ROIs to left/right split")
    print("ESC = quit\\n")

    try:
        while True:
            gui_start = perf_now()
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame")
                break

            frame_height, frame_width = frame.shape[:2]
            current_frame[0] = frame
            current_frame_size[0] = frame_width
            current_frame_size[1] = frame_height

            if not initialized:
                print("\\nFrame shape:", frame.shape)

                loaded = load_rois(frame_width, frame_height)
                rois["left"] = loaded["left"]
                rois["right"] = loaded["right"]

                cv2.setMouseCallback(
                    WINDOW_NAME,
                    make_mouse_handler(
                        get_current_frame_size,
                        get_current_frame,
                        catalog,
                        submit_point_scan,
                        submit_box_scan,
                        get_last_controls,
                    ),
                )

                initialized = True

            current_time = time.time()

            if AUTO_SCAN_ENABLED and current_time - last_auto_scan_time >= AUTO_SCAN_INTERVAL_SECONDS:
                # Scan only one side per interval. This halves worst-case spikes
                # compared with scanning both playmats together.
                side_to_scan = next_scan_side
                next_scan_side = "right" if next_scan_side == "left" else "left"

                if MOTION_GATING_ENABLED:
                    should_scan, _motion_reason = motion_gate.should_scan(frame, rois[side_to_scan], side_to_scan)
                else:
                    should_scan = True

                if should_scan:
                    if worker is not None:
                        worker.request_side_scan(frame, rois[side_to_scan], side_to_scan)
                    else:
                        scan_side_for_matches(frame, rois[side_to_scan], side_to_scan, catalog)

                last_auto_scan_time = current_time

            tick_obs_queue()

            display_frame = frame.copy()
            draw_roi(display_frame, "left", rois["left"])
            draw_roi(display_frame, "right", rois["right"])
            draw_card_matches(display_frame, latest_matches["left"])
            draw_card_matches(display_frame, latest_matches["right"])
            draw_manual_drag_box(display_frame)
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

            if GUI_STATUS_SIDEBAR_ENABLED:
                sidebar = make_status_sidebar(display_frame.shape[0], get_scanner_status())
                display_frame = np.hstack([display_frame, sidebar])

            if GUI_WINDOW_RESIZABLE and GUI_LETTERBOX_RESIZED_WINDOW:
                display_frame = fit_image_to_window(WINDOW_NAME, display_frame)

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if EXIT_ON_WINDOW_CLOSE:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            if handle_card_menu_key(key):
                continue

            if key == 27:
                break

            if key == ord("q"):
                if worker is not None:
                    worker.request_side_scan(frame, rois["left"], "left")
                else:
                    scan_side_for_matches(frame, rois["left"], "left", catalog)

            elif key == ord("e"):
                if worker is not None:
                    worker.request_side_scan(frame, rois["right"], "right")
                else:
                    scan_side_for_matches(frame, rois["right"], "right", catalog)

            elif key == ord("s"):
                save_rois()

            elif key == ord("l"):
                loaded = load_rois(frame_width, frame_height)
                rois["left"] = loaded["left"]
                rois["right"] = loaded["right"]
                print("\\nLoaded ROI settings.")

            elif key == ord("r"):
                rois.update(default_rois(frame_width, frame_height))
                print("\\nReset ROIs to default left/right split.")

            gui_elapsed = perf_now() - gui_start
            perf_record("gui_frame_ms", gui_elapsed * 1000.0)

            remaining = target_frame_seconds - gui_elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        if worker is not None:
            worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_error("Fatal scanner error", exc)
        raise
