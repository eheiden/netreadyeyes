# CollectorVision Netrunner Scanner

Real-time Netrunner card recognition for webcam-based livestream overlays.

## Core features

- Live webcam scanning of physical Netrunner cards
- Separate left/right playmat regions
- Automatic card candidate detection
- CollectorVision-assisted corner refinement
- Rotation-aware catalogue matching
- Card-back detection
- Unknown/ambiguous result handling
- Stability tracking to reduce flicker

## Card tracking

- Persistent card tracks across frames
- Automatic re-identification when a card moves or flips
- Always-on cheap raw thumbnail comparison for stable cards
- Raw visual-difference checks for card-back/face-up changes
- Same-position reuse to avoid unnecessary expensive rescans
- Partial-card/art-box rejection
- Bounding-box smoothing
- Existing overlays stay visible while re-identification is pending

## Manual controls

- Left-click a recognized card: force it to the front of the OBS queue
- Left-click an empty playmat area: manually scan for a missed card there
- Left-click and drag: manually define a search area for a missed card
- Right-click a card: open the card action menu
- Manual scans preserve the rest of the visible tracks
- Manual scans use more permissive search behavior than automatic scans

## OBS integration

- Left and right OBS queues
- Force-to-front queue action
- Last-sent tracking
- Text sync-file support
- Overlay/server support for browser-source workflows

## Diagnostics

- Worker errors are printed in red in the terminal
- Live status sidebar
- Performance timings
- Per-thread CPU visibility
- Human-readable stability log
- Detailed machine-readable stability event log
- Raw visual-difference logs for flip debugging
- Performance spike log

## Keyboard controls

- `Q`: scan left side
- `E`: scan right side
- `S`: save ROIs
- `L`: load ROIs
- `R`: reset ROIs
- `ESC`: exit

Closing the OpenCV window with the X button exits the scanner.
