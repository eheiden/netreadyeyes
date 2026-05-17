import cv2

try:
    from pygrabber.dshow_graph import FilterGraph
except Exception:  # pragma: no cover - exercised on systems without pygrabber
    FilterGraph = None

from .config import CAMERA_NAME, REQUEST_WIDTH, REQUEST_HEIGHT, REQUEST_FPS, OPENCV_NUM_THREADS


def list_cameras():
    if FilterGraph is None:
        raise RuntimeError("pygrabber is required to enumerate Windows video sources")
    graph = FilterGraph()
    return graph.get_input_devices()


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    if value != value:
        return default
    return value


def get_camera_capture_info(cap):
    """Return the currently-open capture's width/height/fps metadata.

    FPS reported by webcams/virtual cameras is sometimes approximate or 0, but
    it is still useful as source metadata in the UI.
    """
    if cap is None:
        return {"width": 0, "height": 0, "fps": 0.0}
    return {
        "width": int(round(_safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))),
        "height": int(round(_safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))),
        "fps": _safe_float(cap.get(cv2.CAP_PROP_FPS)),
    }


def probe_camera_info(camera_index, requested=True):
    """Probe a camera and return best-effort resolution/FPS metadata.

    This opens the device briefly.  DirectShow generally reports the active mode
    after width/height/fps are requested, which is exactly what we want for the
    source picker.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0}
        if requested:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)
        return get_camera_capture_info(cap)
    except Exception:
        return {"width": 0, "height": 0, "fps": 0.0}
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def format_camera_info(info):
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    fps = _safe_float(info.get("fps"), 0.0)
    if width > 0 and height > 0 and fps > 0:
        return f"{width}x{height} @ {fps:.1f} fps"
    if width > 0 and height > 0:
        return f"{width}x{height}"
    if fps > 0:
        return f"{fps:.1f} fps"
    return ""


def list_video_sources(probe=False):
    devices = list_cameras()
    sources = []
    for i, name in enumerate(devices):
        # Probing every device can freeze or spam driver warnings on Windows,
        # especially with virtual cameras.  The picker therefore lists sources
        # immediately by default and only includes metadata when the caller
        # explicitly asks for a probe.
        info = probe_camera_info(i) if probe else {"width": 0, "height": 0, "fps": 0.0}
        info_text = format_camera_info(info)
        label = f"{i}: {name}"
        if info_text:
            label = f"{label} — {info_text}"
        sources.append({
            "index": i,
            "name": name,
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "fps": _safe_float(info.get("fps"), 0.0),
            "label": label,
        })
    return sources


def open_camera_by_index(camera_index):
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
    except cv2.error:
        pass

    devices = list_cameras()
    if camera_index < 0 or camera_index >= len(devices):
        raise RuntimeError(f"Invalid camera index: {camera_index}")

    print(f"\nUsing camera {camera_index}: {devices[camera_index]}")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)

    info = get_camera_capture_info(cap)
    print("\nActual video source settings:")
    print(f"  Resolution: {format_camera_info(info)}")

    return cap


def open_camera():
    try:
        cv2.setNumThreads(OPENCV_NUM_THREADS)
    except cv2.error:
        pass

    devices = list_cameras()

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

    return open_camera_by_index(camera_index)
