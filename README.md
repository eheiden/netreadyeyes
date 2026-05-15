# CollectorVision Netrunner Scanner

Local scanner for recognizing Netrunner cards from an OBS Virtual Camera feed.

## Run

```powershell
python .\live_scanner.py
```

## Build the catalog from NRDB Standard

```powershell
python .\build_netrunner_catalog.py
```

## Build from a local folder

```powershell
python .\build_netrunner_catalog.py --source folder --image-dir cards
```

## Append local cards

```powershell
python .\build_netrunner_catalog.py --source folder --image-dir extra_cards --append
```

## Alt arts

Put custom images in:

```text
alt_arts/
```

Example names:

```text
sure_gamble_alt_worlds_2024.jpg
hedge_fund_alt_stream_overlay.png
```

## Scanner notes

The scanner is organized as:

```text
camera.py          camera selection / capture setup
roi.py             saved playmat regions and mouse editing
detection.py       rough rectangle proposals
corner_refine.py   optional corner refinement and dewarping
recognition.py     crop analysis, matching, queue decisions
tracking.py        card tracking and stable-card reuse
obs_queue.py       paced FIFO output to OBS
obs_bridge.py      HTTP bridge call
drawing.py         debug overlays
```

Current recognition flow:

```text
ROI
→ rough rectangle proposals
→ optional corner refinement
→ card-back / low-detail check
→ embedding match
→ tracker
→ OBS FIFO queue
```

## Useful settings

Most tuning values are in:

```text
netrunner_scanner/config.py
```

Important ones:

```python
CONFIDENCE_THRESHOLD = 0.4

OBS_QUEUE_ENABLED = True
OBS_QUEUE_POP_INTERVAL_SECONDS = 8.0

TRACK_IOU_THRESHOLD = 0.35
STATIONARY_CENTER_THRESHOLD_PX = 18
STATIONARY_REFRESH_SECONDS = 20.0

USE_COLLECTORVISION_CORNER_REFINER = True
CORNER_REFINER_PADDING_RATIO = 0.10
CORNER_REFINER_MIN_SHARPNESS = 0.035

AMBIGUOUS_MATCH_SCORE_THRESHOLD = 0.55
AMBIGUOUS_MATCH_MIN_MARGIN = 0.08

USE_INTERNAL_CONTOUR_PROPOSALS = True
PROPOSAL_NMS_IOU_THRESHOLD = 0.55

REJECT_NESTED_CANDIDATES = True
NESTED_CONTAINMENT_THRESHOLD = 0.85
NESTED_AREA_RATIO_THRESHOLD = 0.25

COMPOSITE_PARENT_REJECTION_ENABLED = True
```

## Overlay labels

Examples:

```text
sure_gamble_alt_1 0.62 opencv m:0.23 s:0.010 stable
card_back 1.00 opencv edge:0.032 stable
```

Meaning:

```text
opencv = using the rough OpenCV crop
cv     = using the refined/dewarped crop
m      = top match score minus second-best score
s      = corner-refiner sharpness
stable = tracker has not seen meaningful movement recently
```

## Debug tip

After editing code, run:

```powershell
python -m compileall netrunner_scanner
```

This catches syntax and indentation problems before running the scanner.


## v14 notes

Added a secondary solid-card-back proposal pass. It looks for low-texture colored rectangles that stand out from the mat, which helps catch facedown cards that do not produce enough internal edges for the normal rectangle detector.

Added motion-gated scanning. The display loop still runs continuously, but recognition is skipped for a side if that playmat has not changed recently. This should reduce CPU use during stable board states.

Relevant settings:

```python
SOLID_BACK_PROPOSALS_ENABLED = True
MOTION_GATING_ENABLED = True
MOTION_SCAN_THRESHOLD = 0.006
MOTION_FORCE_SCAN_SECONDS = 8.0
```


## Built-in OBS overlay server

The scanner now starts its own local overlay server. The old Node/socket.io process is no longer needed.

OBS browser sources:

```text
http://127.0.0.1:8765/card-view-left.html
http://127.0.0.1:8765/card-view-right.html
```

Direct image URLs are also available:

```text
http://127.0.0.1:8765/current/left.jpg
http://127.0.0.1:8765/current/right.jpg
```

The recognition queue sends cards to the matching side based on ROI:

```text
left ROI  -> left overlay
right ROI -> right overlay
```

Card images are searched in this order:

```python
CARD_IMAGE_DIRS = [
    "public/cards",
    "cards",
    "alt_arts",
    "downloaded_cards",
]
```

The old `/cardmatch/<id>` route still works for compatibility and updates the left side. The new route is:

```text
/cardmatch/<side>/<id>
```


## v16 notes

The overlay update now preserves the old timing behavior:

```text
write obs_sync_file.txt
wait OBS_IMAGE_UPDATE_DELAY_SECONDS
update the browser overlay image
```

The scanner window now has a status sidebar showing:

```text
OBS FIFO queue
last card sent left/right
recognized/tracked cards on each playmat
queued/sent/stable states
```

Relevant settings:

```python
OBS_IMAGE_UPDATE_DELAY_SECONDS = 1.0
GUI_STATUS_SIDEBAR_ENABLED = True
GUI_STATUS_SIDEBAR_WIDTH = 430
```


## v17 notes

Left and right overlays now have independent FIFOs. If both sides have cards ready, both can update on the same queue tick.

OBS sync files are now side-specific:

```text
obs_sync_file_left.txt
obs_sync_file_right.txt
```

The sync files contain only:

```text
card_id.jpg
```

The scanner preview window is resizable. The OpenCV window uses `WINDOW_KEEPRATIO` so the preview should scale without intentional aspect distortion.

Stable known cards are no longer periodically re-recognized. They stay cached until the tracker sees meaningful movement.


## v18 notes

Removed cards should clear faster now. Tracks expire after about four seconds, while stable known cards still reuse their cached recognition result until they move.

The status sidebar now includes a legend for overlay colors and label fields:

```text
m = top match score minus second-best score
s = corner-refiner sharpness
opencv = rough crop used
cv = refined crop used
stable = cached/not moving
```

The resizable scanner window now manually letterboxes the preview into the window to avoid stretching the video/sidebar image.


## v19 notes

The detector is stricter about partial-card proposals. It now rejects edge-based candidate boxes that are much smaller than the normal card-sized boxes on the same playmat.

This is meant to remove detections around:

```text
text boxes
art boxes
covered top/bottom card regions
small inner rectangles
```

New settings:

```python
EDGE_PROPOSAL_MIN_AREA_RATIO = 0.006
WHOLE_CARD_SIZE_FILTER_ENABLED = True
WHOLE_CARD_MIN_AREA_FRACTION_OF_REFERENCE = 0.52
INNER_BOX_OVERLAP_REJECTION_ENABLED = True
```


## v20 notes

The tracker now keeps old card positions in memory after they disappear from the scanner overlay. If a hand passes over a card and the same card is detected again in roughly the same spot, the old recognition result is reused and it is not sent back into the OBS queue.

Removed cards still disappear from the scanner overlay after a few seconds:

```python
TRACK_VISIBLE_MISSING_SECONDS = 3.0
```

but their old positions remain available for reacquiring:

```python
TRACK_EXPIRE_SECONDS = 60.0
TRACK_REACQUIRE_CENTER_THRESHOLD_PX = 120
TRACK_REQUEUE_CENTER_THRESHOLD_PX = 140
```


## v21 notes

Cached/reacquired known cards now use the handled/yellow state instead of staying magenta forever.

The rough proposal box is refreshed whenever a track is matched, even when the card does not need to be re-recognized. This should prevent old gray proposal boxes from hanging around after a small proposal shift.

The legend is always drawn near the top of the status sidebar.


## v22 notes

Reacquired cards now use a lightweight visual signature check. If a card-shaped
proposal appears in the same spot as a known card, the scanner compares the new
crop to the saved signature before reusing the old ID. This should keep a hand
passing over a card from requeueing it, while still allowing a different card in
the same spot to be recognized.

Click a card in the scanner window to open the card action menu:

```text
C = clear identification and force a refresh
F = force that card to the front of its side's OBS queue
```

New settings:

```python
VISUAL_SIGNATURE_CHANGED_THRESHOLD = 0.32
VISUAL_SIGNATURE_FORCE_RECHECK_THRESHOLD = 0.42
CARD_MENU_ENABLED = True
```


## v23 notes

Same-position reuse is now limited to short hand/arm occlusions:

```python
TRACK_REUSE_AFTER_MISSING_SECONDS = 1.25
```

If a card has been missing longer than that, a new proposal in the same spot is
recognized again instead of inheriting the old ID.

The card action menu is now mouse-clickable:

```text
Clear ID + refresh
Force to front of OBS queue
```

The menu highlights actions on hover.

Clicking an unrecognized spot inside either ROI now runs a manual scan around
that click location. This is meant as a fallback for cards the automatic proposal
stage missed.


## v25 notes

Recognition now runs in a background scanner worker instead of the OpenCV GUI loop.
The GUI should keep updating while detection/recognition is busy.

The worker drops duplicate pending left/right scans so motion bursts do not build an ever-growing backlog.
Manual click scans are also submitted to the worker.

The status sidebar now includes basic timing diagnostics:

```text
gui frame ms
left/right worker scan ms
manual point scan ms
queued / dropped worker jobs
```

New settings:

```python
SCANNER_WORKER_ENABLED = True
SCANNER_WORKER_MAX_QUEUE_SIZE = 6
SCANNER_WORKER_DROP_DUPLICATE_SIDE_REQUESTS = True
```


## v26 notes

Performance-first update.

CPU reductions:
- full side scans are now throttled with `SIDE_SCAN_COOLDOWN_SECONDS`
- stable known cards no longer run the visual-signature crop/refine path
- each side scan recognizes at most `MAX_RECOGNITIONS_PER_SCAN` new/problem candidates
- preview drawing is downscaled with `GUI_DISPLAY_SCALE`
- motion gate downsample width was reduced
- accepted worker jobs copy frames only after passing duplicate/cooldown checks

Anti-flicker changes:
- a known card is not immediately changed to `unknown` after one bad read
- `UNKNOWN_CLEAR_FRAMES` controls how many consecutive bad reads are required
- when a card is likely physically replaced, the tracker marks it as a fresh ID case

Important knobs:

```python
AUTO_SCAN_INTERVAL_SECONDS = 1.5
SIDE_SCAN_COOLDOWN_SECONDS = 1.5
MAX_RECOGNITIONS_PER_SCAN = 2
GUI_DISPLAY_SCALE = 0.70
DISABLE_STABLE_SIGNATURE_RECHECK = True
UNKNOWN_CLEAR_FRAMES = 4
```

If CPU is still too high during stream tests, try:

```python
AUTO_SCAN_INTERVAL_SECONDS = 2.0
SIDE_SCAN_COOLDOWN_SECONDS = 2.0
GUI_DISPLAY_SCALE = 0.55
MAX_RECOGNITIONS_PER_SCAN = 1
```


## v27 notes

CPU diagnostics:
- added per-thread CPU sampling with `psutil`
- top thread CPU users now show in the Performance panel
- CPU spikes above `PERF_SPIKE_LOG_THRESHOLD_PERCENT` are appended to `scanner_perf_spikes.log`

Install psutil if the panel says it is unavailable:

```powershell
pip install psutil
```

CPU reductions:
- only one playmat side is scanned per auto-scan interval
- auto-scan interval is now 2 seconds
- side cooldown is now 2 seconds
- stable known cards are not rechecked with visual-signature crops
- preview scale defaults to 0.60

Recognition stability:
- known card IDs are held through transient bad reads
- `unknown` tracks are retried on a cooldown instead of every scan
- unprocessed `tracking` candidates are prioritized fairly so the same bad candidates do not monopolize the recognition budget


## v28 notes

This update fixes a mouse-coordinate bug caused by preview downscaling. Clicks are
now mapped from the resized/letterboxed window back through the GUI display scale
before they are compared to card boxes or ROIs.

CPU changes:
- OpenCV is forced to one native thread with `cv2.setNumThreads(1)`
- common native math thread env vars are set in `netrunner_scanner/__init__.py`
- camera FPS defaults to 15
- GUI draw loop is capped with `GUI_TARGET_FPS`
- thread monitor labels unknown native threads as `native-<tid>`

Recognition changes:
- CollectorVision corner refinement is enabled again for accuracy
- manual clicks no longer create card_back boxes on blank mat by default
- manual click scans require enough edge/detail before accepting a card-back-like result

Useful settings:

```python
REQUEST_FPS = 15
GUI_TARGET_FPS = 15
OPENCV_NUM_THREADS = 1
GUI_DISPLAY_SCALE = 0.55
USE_COLLECTORVISION_CORNER_REFINER = True
MANUAL_SCAN_ALLOW_CARDBACK = False
```


## v29 notes

This update backs out the part of the performance tuning that hurt recognition stability.

Important fixes:
- `TRACK_REUSE_AFTER_MISSING_SECONDS` is now 5.0 seconds. The previous short
  value could be shorter than the normal time between scans of the same side,
  so stable cards were being treated as possible replacements.
- automatic scanning is back to 1 second / 1 second side cooldown
- max recognitions per scan is back up to 4
- stable known cards are not overwritten by transient `unknown` reads
- the sidebar now shows per-track diagnostic state:
  - unknown streak
  - missing time
  - force-new status
  - last decision
- `scanner_stability_events.log` records scan summaries and label changes

Useful settings:

```python
AUTO_SCAN_INTERVAL_SECONDS = 1.0
SIDE_SCAN_COOLDOWN_SECONDS = 1.0
MAX_RECOGNITIONS_PER_SCAN = 4
TRACK_REUSE_AFTER_MISSING_SECONDS = 5.0
UNKNOWN_RETRY_COOLDOWN_SECONDS = 2.5
TRACKING_RETRY_COOLDOWN_SECONDS = 0.75
```

The key lesson: if `TRACK_REUSE_AFTER_MISSING_SECONDS` is shorter than the
ordinary scan cadence for a side, normal stable cards look "missing" and can be
reprocessed as if they were new cards.
