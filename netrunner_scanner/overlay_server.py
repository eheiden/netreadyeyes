
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .config import (
    OVERLAY_HOST,
    OVERLAY_PORT,
    CARD_IMAGE_DIRS,
    CARD_IMAGE_EXTENSIONS,
    DEFAULT_CARD_BACK_IMAGE,
    OBS_TEXT_SYNC_ENABLED,
    OBS_TEXT_SYNC_FILE,
    OBS_TEXT_SYNC_FILE_LEFT,
    OBS_TEXT_SYNC_FILE_RIGHT,
    OBS_TEXT_SYNC_INCLUDE_LABEL,
    OBS_IMAGE_UPDATE_DELAY_SECONDS,
)


_state_lock = threading.Lock()
_update_generation = {
    "left": 0,
    "right": 0,
}
_state = {
    "left": {
        "card_id": "",
        "src": "",
        "updated_at": 0.0,
        "counter": 0,
        "pending_card_id": "",
        "pending_until": 0.0,
    },
    "right": {
        "card_id": "",
        "src": "",
        "updated_at": 0.0,
        "counter": 0,
        "pending_card_id": "",
        "pending_until": 0.0,
    },
}

_server = None
_server_thread = None


def normalize_side(side):
    side = (side or "").lower().strip()

    if side in ("left", "pink", "l"):
        return "left"

    if side in ("right", "blue", "r"):
        return "right"

    return "left"


def card_image_url(card_id):
    if not card_id:
        card_id = DEFAULT_CARD_BACK_IMAGE

    return f"/cards/{card_id}"


def sync_file_for_side(side):
    side = normalize_side(side)

    if side == "left":
        return OBS_TEXT_SYNC_FILE_LEFT

    if side == "right":
        return OBS_TEXT_SYNC_FILE_RIGHT

    return OBS_TEXT_SYNC_FILE


def sync_text_for_card(side, card_id):
    card_id = str(card_id).strip()

    if OBS_TEXT_SYNC_INCLUDE_LABEL:
        return f"{side} {card_id}.jpg"

    return f"{card_id}.jpg"


def write_sync_file(side, card_id):
    if not OBS_TEXT_SYNC_ENABLED:
        return

    path = Path(sync_file_for_side(side))
    text = sync_text_for_card(side, card_id)

    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def _set_overlay_state(side, card_id, generation):
    side = normalize_side(side)
    card_id = str(card_id).strip()

    with _state_lock:
        if generation != _update_generation[side]:
            return

        side_state = _state[side]
        side_state["card_id"] = card_id
        side_state["src"] = card_image_url(card_id)
        side_state["updated_at"] = time.time()
        side_state["counter"] += 1
        side_state["pending_card_id"] = ""
        side_state["pending_until"] = 0.0
        counter = side_state["counter"]

    print(f"Overlay image update: {side} {card_id} ({counter})")


def _delayed_set_overlay_state(side, card_id, delay_seconds, generation):
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    _set_overlay_state(side, card_id, generation)


def publish_card(side, card_id):
    side = normalize_side(side)
    card_id = str(card_id).strip()

    if not card_id:
        return

    write_sync_file(side, card_id)

    delay_seconds = max(0.0, float(OBS_IMAGE_UPDATE_DELAY_SECONDS))

    with _state_lock:
        _update_generation[side] += 1
        generation = _update_generation[side]

        side_state = _state[side]
        side_state["pending_card_id"] = card_id
        side_state["pending_until"] = time.time() + delay_seconds

    print(f"Overlay trigger written: {side} {card_id}; image updates in {delay_seconds:.2f}s")

    thread = threading.Thread(
        target=_delayed_set_overlay_state,
        args=(side, card_id, delay_seconds, generation),
        daemon=True,
    )
    thread.start()


def clear_side(side):
    publish_card(normalize_side(side), DEFAULT_CARD_BACK_IMAGE)


def get_state(side=None):
    with _state_lock:
        if side is not None:
            side = normalize_side(side)
            return dict(_state[side])

        return {
            "left": dict(_state["left"]),
            "right": dict(_state["right"]),
        }


def find_card_image(card_id):
    raw_id = unquote(str(card_id)).strip()

    if not raw_id:
        raw_id = DEFAULT_CARD_BACK_IMAGE

    raw_path = Path(raw_id)

    names_to_try = []

    if raw_path.suffix:
        names_to_try.append(raw_path.name)
    else:
        for ext in CARD_IMAGE_EXTENSIONS:
            names_to_try.append(raw_path.name + ext)

    if DEFAULT_CARD_BACK_IMAGE not in names_to_try and raw_id == DEFAULT_CARD_BACK_IMAGE:
        names_to_try.append(DEFAULT_CARD_BACK_IMAGE)

    for folder in CARD_IMAGE_DIRS:
        base = Path(folder)

        for name in names_to_try:
            candidate = base / name

            if candidate.exists() and candidate.is_file():
                return candidate

    return None


def overlay_html(side):
    side = normalize_side(side)
    title = f"Card View {side.title()}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
html, body {{
  margin: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: transparent;
}}
#wrap {{
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}}
#card {{
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: 100%;
  object-fit: contain;
  border-radius: 20px;
}}
</style>
</head>
<body>
<div id="wrap">
  <img id="card" src="/cards/{DEFAULT_CARD_BACK_IMAGE}" />
</div>
<script>
const SIDE = "{side}";
let lastCounter = -1;

async function poll() {{
  try {{
    const response = await fetch(`/api/state?side=${{SIDE}}&t=${{Date.now()}}`, {{
      cache: "no-store"
    }});
    const data = await response.json();

    if (data.counter !== lastCounter) {{
      lastCounter = data.counter;
      const src = data.src || "/cards/{DEFAULT_CARD_BACK_IMAGE}";
      document.getElementById("card").src = `${{src}}?v=${{data.counter}}`;
    }}
  }} catch (err) {{
    console.log("overlay poll failed", err);
  }}
}}

setInterval(poll, 250);
poll();
</script>
</body>
</html>
"""


class OverlayRequestHandler(BaseHTTPRequestHandler):
    server_version = "CollectorVisionOverlay/1.0"

    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, status=200, content_type="text/plain"):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path):
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/card-view.html"):
            self.send_text(overlay_html("left"), content_type="text/html")
            return

        if path == "/card-view-left.html":
            self.send_text(overlay_html("left"), content_type="text/html")
            return

        if path == "/card-view-right.html":
            self.send_text(overlay_html("right"), content_type="text/html")
            return

        if path == "/api/state":
            side = query.get("side", [None])[0]
            self.send_json(get_state(side))
            return

        if path.startswith("/cardmatch/"):
            parts = [part for part in path.split("/") if part]

            if len(parts) == 2:
                side = "left"
                card_id = parts[1]
            elif len(parts) >= 3:
                side = parts[1]
                card_id = parts[2]
            else:
                self.send_json({"error": "missing card id"}, status=400)
                return

            publish_card(side, card_id)
            self.send_json({"ok": True, "side": normalize_side(side), "card_id": card_id})
            return

        if path.startswith("/cards/"):
            card_id = path.split("/cards/", 1)[1]
            image_path = find_card_image(card_id)

            if image_path is None:
                self.send_json({"error": "card image not found", "card_id": card_id}, status=404)
                return

            self.send_file(image_path)
            return

        if path.startswith("/current/"):
            side_name = Path(path).stem
            side = normalize_side(side_name)
            side_state = get_state(side)
            card_id = side_state.get("card_id") or DEFAULT_CARD_BACK_IMAGE
            image_path = find_card_image(card_id)

            if image_path is None:
                image_path = find_card_image(DEFAULT_CARD_BACK_IMAGE)

            if image_path is None:
                self.send_json({"error": "current image not found"}, status=404)
                return

            self.send_file(image_path)
            return

        self.send_json({"error": "not found"}, status=404)


def start_overlay_server():
    global _server, _server_thread

    if _server is not None:
        return

    _server = ThreadingHTTPServer((OVERLAY_HOST, OVERLAY_PORT), OverlayRequestHandler)
    _server_thread = threading.Thread(
        target=_server.serve_forever,
        name="OverlayServer",
        daemon=True,
    )
    _server_thread.start()

    print(f"Overlay server running at http://{OVERLAY_HOST}:{OVERLAY_PORT}/")
    print(f"  left:  http://{OVERLAY_HOST}:{OVERLAY_PORT}/card-view-left.html")
    print(f"  right: http://{OVERLAY_HOST}:{OVERLAY_PORT}/card-view-right.html")


def stop_overlay_server():
    global _server, _server_thread

    if _server is None:
        return

    _server.shutdown()
    _server.server_close()
    _server = None
    _server_thread = None
