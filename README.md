# Net Ready Eyes v0.9

Net Ready Eyes is a webcam-based Netrunner card recognition tool for livestream overlays and OBS workflows.

## Current version

- Major: `0`
- Minor: `9`
- Version: `0.9`

Version numbers live in `netrunner_scanner/config.py` and `netrunner_scanner/version.py`.

## What it does

- Watches one or two playmat regions from a webcam or capture device
- Detects Netrunner cards on the table
- Identifies face-up cards from a local catalogue
- Detects facedown/card-back sleeves without sleeve-specific images
- Tracks cards across frames so the overlay does not flicker
- Sends recognized cards to OBS through left/right queues
- Supports manual click/drag recovery when automatic detection struggles
- Provides a manual card selector for ambiguous or incorrect IDs

## Controls

- Left-click a recognized card: force it to the front of the OBS queue
- Left-click an empty playmat area: manually scan for a missed card
- Left-click and drag: manually define a card search area
- Right-click a card: open the card action menu
- Right-click menu: open manual selector, force to OBS, or delete track
- Manual card selector: click a choice or press `1`–`5`
- `M`: open selector from the right-click menu
- `Q`: scan left side
- `E`: scan right side
- `S`: save ROIs
- `L`: load ROIs
- `R`: reset ROIs
- `ESC`: exit or cancel a selector/menu

## Setup on Windows

From PowerShell in the project folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Then run:

```powershell
.\.venv\Scripts\python.exe .\netreadyeyes.py
```

`netreadyeyes.py` is the app entry point.

## Manual setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\check_setup.py
.\.venv\Scripts\python.exe .\netreadyeyes.py
```

## Node / nvm notes

Net Ready Eyes is primarily Python. Node is only needed if you add or maintain Node-based overlay/helper tooling.

On Windows, install **nvm-windows**, then:

```powershell
nvm install 20
nvm use 20
node -v
npm -v
```

If `node -v` shows the wrong version after `nvm use`, close and reopen the terminal, then check your PATH.

## Catalogue

The runtime expects:

```text
netrunner-catalog.npz
```

Build or copy that file into the project root before running the scanner.

## Diagnostics

- Worker/fatal errors print in red in the terminal
- Stability logging is quiet by default
- Set `DEBUG_VERBOSE_STABILITY_LOGS = True` in `config.py` if a recognition issue comes back
- Performance spikes are written to `scanner_perf_spikes.log`
- Stability events are written to `scanner_stability_events.log`

## Troubleshooting

Run:

```powershell
.\.venv\Scripts\python.exe .\check_setup.py
```

Common things to verify:

- Python 3.10+ is installed
- The virtual environment is active or being called directly
- `netrunner-catalog.npz` exists in the project root
- The correct webcam/capture device is selected in `config.py`
- ROI settings match the camera frame

## Codebase cleanup

See `CODE_CLEANUP_AUDIT.md` for a short refactor map and dead-code cleanup notes.

## Runtime controls

The right sidebar includes a Controls panel.

- **Automatic mode**: recognized cards can be added to OBS automatically.
- **Manual mode**: cards are still recognized/tracked, but only left-clicked cards are sent to OBS.
- **Left-click OBS send**:
  - **Instant** sends a clicked card to OBS immediately.
  - **Use queue** puts a clicked card at the front of its side queue and respects the queue wait timer.
- **Queue wait** value control: 1 to 15 seconds; default is 5 seconds for new settings files.
- **Save settings** writes your choices to `netreadyeyes_settings.json`; they load automatically next time.

## GPU acceleration

The Controls panel includes a GPU On/Off setting and the Performance section shows the active ONNX provider when detectable.

Important: CollectorVision owns some model/session internals. Net Ready Eyes can show whether CUDA/TensorRT/DirectML providers are available and records the GPU preference before models are loaded, but if a provider change does not take effect immediately, restart the app after saving the setting.

## Manual selector text search

When the Manual card selector is open, type part of a card name. The selector will replace the image-recognition suggestions with the closest catalogue matches. Use Backspace to edit and `1`-`5` or click to choose a result.

## Measuring GPU impact

The live UI shows which ONNX provider is available/active, but the best way to compare GPU/CPU is to benchmark the same images both ways:

```powershell
.\.venv\Scripts\python.exe .\benchmark_gpu.py
```

By default, the benchmark:
- searches the configured card image folders from `CARD_IMAGE_DIRS`
- tests every catalogue card image it can find
- runs GPU ON and GPU OFF automatically
- prints timing results
- writes `benchmark_report.txt`
- lists the most ambiguous / difficult cards by incorrect result or low match margin

Useful options:

```powershell
.\.venv\Scripts\python.exe .\benchmark_gpu.py --limit 100
.\.venv\Scripts\python.exe .\benchmark_gpu.py --report my_report.txt
.\.venv\Scripts\python.exe .\benchmark_gpu.py --image-dirs public/cards cards downloaded_cards alt_arts
```

If GPU and CPU numbers are almost identical, CollectorVision or ONNX Runtime may still be using CPU internally, the model may be too small for GPU to help, or image preprocessing may dominate the runtime.

### Ambiguity report filtering

The benchmark report now excludes same-name printing conflicts from the difficult-card list by default. For example, `hedge_fund_20132` vs `hedge_fund_25146` is treated as the same practical card name, not a gameplay-relevant misrecognition.

To include those printing-level conflicts anyway:

```powershell
.\.venv\Scripts\python.exe .\benchmark_gpu.py --include-same-name-printings
```

### CPU-load comparison

`benchmark_gpu.py` now reports CPU load for GPU ON and OFF runs.

- `ms/image` is the average catalogue matching time for one card image.
- `core-equiv CPU %` is CPU usage measured as one core = 100%; this can exceed 100% on multicore systems.
- `whole-machine CPU %` divides that by your logical CPU count, which is closer to overall Task Manager-style headroom.

Install `psutil` for CPU-load measurements:

```powershell
.\.venv\Scripts\python.exe -m pip install psutil
```

## Local pre-push checks

Run the lightweight test suite before pushing changes:

```powershell
.\run_tests.ps1
```

or:

```powershell
python .\scripts\run_pre_push_tests.py
```

The checks compile the main Python files and run the unit tests in `tests/`. They intentionally avoid opening cameras, OBS, or GUI windows, so they should be safe to run quickly while developing.

To install an optional local Git pre-push hook:

```powershell
python .\scripts\install_git_hooks.py
```

