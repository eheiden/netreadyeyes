import time
import cv2

from netrunner_scanner.config import (
    WINDOW_NAME,
    AUTO_SCAN_ENABLED,
    AUTO_SCAN_INTERVAL_SECONDS,
)
from netrunner_scanner.camera import open_camera
from netrunner_scanner.catalog import CardCatalog
from netrunner_scanner.roi import (
    rois,
    load_rois,
    save_rois,
    default_rois,
    on_mouse,
)
from netrunner_scanner.recognition import (
    latest_matches,
    scan_side_for_matches,
)
from netrunner_scanner.drawing import draw_roi, draw_card_matches


def main():
    cap = open_camera()
    catalog = CardCatalog("netrunner-catalog.npz")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    initialized = False
    last_auto_scan_time = 0

    print("\nControls:")
    print("Mouse drag inside box = move ROI")
    print("Mouse drag edge/corner = resize ROI")
    print("Q = manually scan LEFT / pink playmat")
    print("E = manually scan RIGHT / blue playmat")
    print("S = save ROI settings")
    print("L = load ROI settings")
    print("R = reset ROIs to left/right split")
    print("ESC = quit\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        frame_height, frame_width = frame.shape[:2]

        if not initialized:
            print("\nFrame shape:", frame.shape)

            loaded = load_rois(frame_width, frame_height)
            rois["left"] = loaded["left"]
            rois["right"] = loaded["right"]

            cv2.setMouseCallback(
                WINDOW_NAME,
                on_mouse,
                param=(frame_width, frame_height),
            )

            initialized = True

        now = time.time()

        if AUTO_SCAN_ENABLED and now - last_auto_scan_time >= AUTO_SCAN_INTERVAL_SECONDS:
            scan_side_for_matches(frame, rois["left"], "left", catalog)
            scan_side_for_matches(frame, rois["right"], "right", catalog)
            last_auto_scan_time = now

        display_frame = frame.copy()

        draw_roi(display_frame, "left", rois["left"])
        draw_roi(display_frame, "right", rois["right"])

        draw_card_matches(display_frame, latest_matches["left"])
        draw_card_matches(display_frame, latest_matches["right"])

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord("q"):
            scan_side_for_matches(frame, rois["left"], "left", catalog)

        elif key == ord("e"):
            scan_side_for_matches(frame, rois["right"], "right", catalog)

        elif key == ord("s"):
            save_rois()

        elif key == ord("l"):
            loaded = load_rois(frame_width, frame_height)
            rois["left"] = loaded["left"]
            rois["right"] = loaded["right"]
            print("\nLoaded ROI settings.")

        elif key == ord("r"):
            rois.update(default_rois(frame_width, frame_height))
            print("\nReset ROIs to default left/right split.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
