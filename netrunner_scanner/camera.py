import cv2
from pygrabber.dshow_graph import FilterGraph

from .config import CAMERA_NAME, REQUEST_WIDTH, REQUEST_HEIGHT, REQUEST_FPS, OPENCV_NUM_THREADS


def open_camera():
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
    except cv2.error:
        pass

    graph = FilterGraph()
    devices = graph.get_input_devices()

    print("\nAvailable cameras:\n")

    for i, name in enumerate(devices):
        print(f"{i}: {name}")

    camera_index = None

    for i, name in enumerate(devices):
        if CAMERA_NAME.lower() in name.lower():
            camera_index = i
            break

    if camera_index is None:
        raise RuntimeError(f"Could not find camera: {CAMERA_NAME}")

    print(f"\nUsing camera {camera_index}: {devices[camera_index]}")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)

    print("\nActual camera settings:")
    print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")

    return cap
