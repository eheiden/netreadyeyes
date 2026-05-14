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
