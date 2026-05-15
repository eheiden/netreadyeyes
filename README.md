# Net Ready Eyes v0.4

Net Ready Eyes is a webcam-based Netrunner card recognition tool for livestream overlays and OBS workflows.

## Current version

- Major: `0`
- Minor: `4`
- Version: `0.4`

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

`live_scanner.py` is kept as a small compatibility wrapper, but `netreadyeyes.py` is the main entry point.

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
- **Queue wait** slider: 4 to 10 seconds.
- **Save settings** writes your choices to `netreadyeyes_settings.json`; they load automatically next time.
