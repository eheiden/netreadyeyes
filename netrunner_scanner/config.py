from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

CAMERA_NAME = "OBS Virtual Camera"

TOP_K = 5
CONFIDENCE_THRESHOLD = 0.4

REQUEST_WIDTH = 1920
REQUEST_HEIGHT = 1080
REQUEST_FPS = 15

ROI_SETTINGS_FILE = Path("roi_settings.json")
WINDOW_NAME = "Net Ready Eyes"

HANDLE_SIZE = 12
MIN_ROI_SIZE = 40

AUTO_SCAN_ENABLED = True
AUTO_SCAN_INTERVAL_SECONDS = 1.0

# Direct send is disabled because OBS_QUEUE_ENABLED handles output timing.
AUTO_SEND_TO_OBS = False

DEBUG_SAVE_CROPS = False
DEBUG_CROPS_DIR = Path("debug_crops")
DEBUG_CROPS_DIR.mkdir(exist_ok=True)

# Tracking

TRACK_IOU_THRESHOLD = 0.35
STATIONARY_CENTER_THRESHOLD_PX = 18
STATIONARY_REFRESH_SECONDS = 9999.0
TRACK_EXPIRE_SECONDS = 12.0

# OBS output queue

OBS_QUEUE_ENABLED = True
OBS_QUEUE_POP_INTERVAL_SECONDS = 8.0
OBS_QUEUE_DEDUP_VISIBLE = True

# Corner refinement

# First pass finds rough card regions; the corner model tightens the crop before matching.
USE_COLLECTORVISION_CORNER_REFINER = True

# Add a small buffer around rough boxes before corner refinement.
CORNER_REFINER_PADDING_RATIO = 0.10

# Reject corner-refine results that look too weak or blurry.
CORNER_REFINER_MIN_SHARPNESS = 0.0355

# Draw the refined corner box when available.
DRAW_REFINED_BOX = True

# Prefer CUDA where the installed runtime supports it.
PREFER_CUDA_FOR_CORNER_DETECTOR = True

CORNER_REFINER_MIN_IOU_WITH_PROPOSAL = 0.25
CORNER_REFINER_MIN_AREA_RATIO = 0.45
CORNER_REFINER_MAX_AREA_RATIO = 2.20
CORNER_REFINER_MIN_EDGE_LENGTH = 25
CORNER_REFINER_MIN_ASPECT = 1.20
CORNER_REFINER_MAX_ASPECT = 1.70
CORNER_REFINER_FALLBACK_TO_OPENCV = True

# Match quality

AMBIGUOUS_MATCH_SCORE_THRESHOLD = 0.55
AMBIGUOUS_MATCH_MIN_MARGIN = 0.08

CARD_BACK_EDGE_RATIO_THRESHOLD = 0.030
CARD_BACK_SAT_STD_THRESHOLD = 45.0

REFINED_BOX_SMOOTHING_ALPHA = 0.75

CARD_BACK_VAL_STD_THRESHOLD = 60.0

# Overlapping-card proposals

USE_INTERNAL_CONTOUR_PROPOSALS = True
PROPOSAL_NMS_IOU_THRESHOLD = 0.55
MIN_RECTANGULAR_FILL_RATIO = 0.48
MAX_RECTANGULAR_FILL_RATIO = 1.25

# Proposal quality filters

REJECT_NESTED_CANDIDATES = True
NESTED_CONTAINMENT_THRESHOLD = 0.85
NESTED_AREA_RATIO_THRESHOLD = 0.25

MIN_CARD_SHORT_SIDE_PX = 82

RECTANGLE_BORDER_CHECK_ENABLED = True
MIN_BORDER_BAND_EDGE_RATIO = 0.018
MIN_BORDER_BANDS_PRESENT = 2

MAX_CANDIDATES_PER_SIDE = 8

# Reject large merged blobs when they contain better card-sized child boxes.
COMPOSITE_PARENT_REJECTION_ENABLED = True
COMPOSITE_CHILD_MIN_AREA_RATIO = 0.25
COMPOSITE_CHILD_MAX_AREA_RATIO = 0.85
COMPOSITE_CHILD_CONTAINMENT_THRESHOLD = 0.55
COMPOSITE_PARENT_MIN_CHILDREN = 1

# Detect facedown cards from low center texture instead of sleeve color.
CARD_BACK_CENTER_EDGE_RATIO_THRESHOLD = 0.020
CARD_BACK_CENTER_SAT_STD_THRESHOLD = 55.0
CARD_BACK_CENTER_VAL_STD_THRESHOLD = 55.0

# Solid facedown cards can be missed by edge-based proposals because they have
# very little internal texture. This secondary pass looks for smooth colored
# rectangles that contrast with the mat.
SOLID_BACK_PROPOSALS_ENABLED = True
SOLID_BACK_MIN_AREA_RATIO = 0.004
SOLID_BACK_MAX_AREA_RATIO = 0.08
SOLID_BACK_MIN_SHORT_SIDE_PX = 70
SOLID_BACK_MAX_TEXTURE_EDGE_RATIO = 0.012
SOLID_BACK_MIN_COLOR_DISTANCE = 32.0

# Skip expensive recognition scans when the playmat has not changed. The UI
# still updates every frame; this only gates the recognition pass.
MOTION_GATING_ENABLED = True
MOTION_DOWNSAMPLE_WIDTH = 180
MOTION_SCAN_THRESHOLD = 0.006
MOTION_FORCE_SCAN_SECONDS = 10.0

# Local OBS overlay server
OVERLAY_SERVER_ENABLED = True
OVERLAY_HOST = "127.0.0.1"
OVERLAY_PORT = 8765

# Folders checked, in order, when the overlay serves /cards/<id>.jpg.
CARD_IMAGE_DIRS = [
    "public/cards",
    "cards",
    "alt_arts",
    "downloaded_cards",
]

CARD_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

DEFAULT_CARD_BACK_IMAGE = "000_card_back_runner.png"

# Optional text file for OBS/Advanced Scene Switcher style triggers.
OBS_TEXT_SYNC_ENABLED = True
OBS_TEXT_SYNC_FILE = "obs_sync_file.txt"

# Delay the overlay image update after writing obs_sync_file.txt.
# This gives OBS movement/blur filters time to animate the old card off-screen
# before the browser source changes to the next card.
OBS_IMAGE_UPDATE_DELAY_SECONDS = 1.0

# Scanner preview sidebar
GUI_STATUS_SIDEBAR_ENABLED = True
GUI_STATUS_SIDEBAR_WIDTH = 430
GUI_STATUS_MAX_QUEUE_ITEMS = 8
GUI_STATUS_MAX_RECOGNIZED_ITEMS = 12

# Side-specific OBS trigger files. These replace the single shared sync file.
OBS_TEXT_SYNC_FILE_LEFT = "obs_sync_file_left.txt"
OBS_TEXT_SYNC_FILE_RIGHT = "obs_sync_file_right.txt"

# Write just the image filename to the sync file.
OBS_TEXT_SYNC_INCLUDE_LABEL = False

# One FIFO per side. Left and right overlays can update at the same time.
OBS_QUEUE_PER_SIDE = True

# More tolerant track matching keeps stationary cards from getting treated as new
# cards when proposal boxes wiggle slightly.
TRACK_CENTER_MATCH_THRESHOLD_PX = 70

# Make the preview window resizable without intentionally changing the source aspect.
GUI_WINDOW_RESIZABLE = True

# When the OpenCV window is resized, draw the preview into a black letterboxed
# canvas instead of letting the backend stretch it.
GUI_LETTERBOX_RESIZED_WINDOW = True

# Edge proposals from internal contours can pick up card text boxes or art boxes.
# These are smaller than actual cards, so use a stricter area floor and a
# per-playmat size sanity check.
EDGE_PROPOSAL_MIN_AREA_RATIO = 0.006
WHOLE_CARD_SIZE_FILTER_ENABLED = True
WHOLE_CARD_MIN_AREA_FRACTION_OF_REFERENCE = 0.52
WHOLE_CARD_REFERENCE_TOP_N = 6

# If an edge proposal is small but overlaps a larger candidate, keep only the
# larger one. This helps reject text/art boxes inside a card.
INNER_BOX_OVERLAP_REJECTION_ENABLED = True
INNER_BOX_OVERLAP_THRESHOLD = 0.35
INNER_BOX_MAX_AREA_RATIO = 0.55

# Track memory vs display clearing:
# - visible_missing controls how fast a picked-up/hidden card disappears
# - expire controls how long the tracker remembers its old position for reacquire
TRACK_VISIBLE_MISSING_SECONDS = 1.0

# If a known card disappears briefly and then reappears close to the same spot,
# reuse the old recognition result and do not requeue it.
TRACK_REACQUIRE_CENTER_THRESHOLD_PX = 120
TRACK_REQUEUE_CENTER_THRESHOLD_PX = 140

# Reacquire guard:
# If a card-shaped proposal appears in the same spot as a known card, compare
# the new crop to the saved lightweight visual signature before reusing the old
# ID. This keeps a hand-wave from requeueing the same card, but lets a different
# card in the same spot get recognized.
VISUAL_SIGNATURE_CHANGED_THRESHOLD = 0.32
VISUAL_SIGNATURE_FORCE_RECHECK_THRESHOLD = 0.42

# Click card controls
CARD_MENU_ENABLED = True

# Same-spot reuse is only for brief hand/arm occlusions. If a card has been
# missing longer than this, re-recognize it even if a new proposal appears in
# the same place.
TRACK_REUSE_AFTER_MISSING_SECONDS = 20.0

# Manual click-scan fallback size. Used when the normal detector missed a card
# and I click inside a playmat ROI.
MANUAL_CLICK_CARD_WIDTH_PX = 155
MANUAL_CLICK_CARD_HEIGHT_PX = 215



# Performance tuning
SCAN_EVERY_N_FRAMES = 4
MAX_RECOGNITIONS_PER_SCAN = 6
UNKNOWN_CLEAR_FRAMES = 4
ENABLE_TIMING_OVERLAY = True


# Scanner worker / responsiveness
SCANNER_WORKER_ENABLED = True
SCANNER_WORKER_MAX_QUEUE_SIZE = 6
SCANNER_WORKER_DROP_DUPLICATE_SIDE_REQUESTS = True
PERF_HISTORY_SIZE = 60


# Performance-first display tuning. Recognition still uses the original frame;
# only the preview image is reduced before drawing overlays/sidebar.
GUI_DISPLAY_SCALE = 0.65

# If True, stable known tracks are not re-run through the visual-signature check.
# This saves a lot of CPU because the signature check has to crop/refine again.
DISABLE_STABLE_SIGNATURE_RECHECK = True

# Minimum time between full side scans when motion is present. This keeps a hand
# or sleeve movement from scheduling recognition continuously.
SIDE_SCAN_COOLDOWN_SECONDS = 1.0


# CPU diagnostics
PER_THREAD_CPU_MONITOR_ENABLED = True
PER_THREAD_CPU_SAMPLE_SECONDS = 2.0
PER_THREAD_CPU_TOP_N = 6
PERF_SPIKE_LOG_THRESHOLD_PERCENT = 30.0
PERF_SPIKE_LOG_FILE = str(LOGS_DIR / "scanner_perf_spikes.log")

# Recognition stability / fairness
UNKNOWN_RETRY_COOLDOWN_SECONDS = 2.5
TRACKING_RETRY_COOLDOWN_SECONDS = 0.75
HIDE_TRACKING_BOXES = False

# CPU-heavy refinement control.
# False = use the rough OpenCV crop by default, which is much cheaper.
# Click/manual scans still work, and this can be turned back on if accuracy drops.
USE_COLLECTORVISION_CORNER_REFINER = True
GUI_TARGET_FPS = 15
OPENCV_NUM_THREADS = 1
MANUAL_SCAN_ALLOW_CARDBACK = False
MANUAL_CLICK_MIN_EDGE_RATIO = 0.030
MANUAL_CLICK_MIN_CENTER_EDGE_RATIO = 0.018
MANUAL_CLICK_REQUIRE_DETAIL = True
HIDE_STALE_MANUAL_MISSES = True
STABILITY_DEBUG_ENABLED = True
STABILITY_EVENT_LOG_FILE = str(LOGS_DIR / "scanner_stability_events.log")
STABILITY_EVENT_LOG_MAX_LINES = 5000
DISPLAY_TRACK_DIAGNOSTICS = True
SAME_SPOT_SIGNATURE_CHECK_ENABLED = True
SAME_SPOT_SIGNATURE_SAME_THRESHOLD = 0.30
SAME_SPOT_SIGNATURE_DIFFERENT_THRESHOLD = 0.42
HOLD_KNOWN_ON_CARDBACK_MISREAD = True
HOLD_KNOWN_ON_ANY_LOW_CONFIDENCE_MISREAD = True
HUMAN_READABLE_CONSOLE_LOGS = True
STABILITY_VERBOSE_RECOGNITION_LOGS = True
STATUS_TEXT_TRUNCATION_SUFFIX = "..."
RELATIVE_CARD_SIZE_FILTER_ENABLED = True
RELATIVE_CARD_SIZE_MIN_AREA_FRACTION = 0.62
RELATIVE_CARD_SIZE_MIN_SHORT_SIDE_FRACTION = 0.70
RELATIVE_CARD_SIZE_MIN_REFERENCE_CARDS = 3
MANUAL_DRAG_SCAN_ENABLED = True
MANUAL_DRAG_MIN_WIDTH_PX = 45
MANUAL_DRAG_MIN_HEIGHT_PX = 45
MANUAL_DRAG_EXPAND_RATIO = 0.10
LEFT_CLICK_FORCE_OBS = True
RIGHT_CLICK_CARD_MENU = True
DRAW_LABEL_SHADOWS = True
STATUS_PANEL_USE_COMPACT_LOG = True
STATUS_LOG_WRAP_WIDTH = 34
STATUS_MAX_STABILITY_EVENTS = 2
MOVED_SANITY_RESCAN_ENABLED = True
MOVED_SANITY_RESCAN_PX = 35
MOVED_SANITY_RESCAN_COOLDOWN_SECONDS = 3.0
MOVED_SANITY_VISUAL_CHECK_FIRST = False
PARTIAL_CARD_REJECT_ENABLED = True
PARTIAL_CARD_MIN_AREA_FRACTION = 0.72
PARTIAL_CARD_MIN_SHORT_SIDE_FRACTION = 0.82
PARTIAL_CARD_REFERENCE_TOP_N = 4
PARTIAL_CARD_MIN_REFERENCES = 2
MANUAL_DRAG_FORCE_FRONT = True
MANUAL_DRAG_PROCESS_SYNC = True
EXIT_ON_WINDOW_CLOSE = True

# Re-identify cards after meaningful movement.
MOVED_REIDENTIFY_ENABLED = False
MOVED_REIDENTIFY_MIN_PX = 35
MOVED_REIDENTIFY_COOLDOWN_SECONDS = 0.25

# Manual scans should aggressively try to find a card.
MANUAL_SCAN_RELAX_THRESHOLDS = True
MANUAL_SCAN_EXPAND_PX = 42
MANUAL_SCAN_MIN_AREA_FRACTION = 0.38

# Preserve visible tracks during manual scans.
MANUAL_SCAN_PRESERVE_TRACKS = True
STABLE_VISUAL_RECHECK_ENABLED = True
STABLE_VISUAL_RECHECK_INTERVAL_SECONDS = 2.0
STABLE_VISUAL_RECHECK_DIFFERENT_THRESHOLD = 0.34
STABLE_VISUAL_RECHECK_SAME_THRESHOLD = 0.16
STABLE_VISUAL_RECHECK_MAX_PER_SCAN = 2
MANUAL_BOX_FIND_CANDIDATES_FIRST = True
MANUAL_BOX_PRESERVE_VISIBLE_MATCHES = True
MANUAL_BOX_MIN_CANDIDATE_AREA = 1800
MANUAL_BOX_EXPAND_SEARCH_PX = 30
MANUAL_POINT_SCAN_ON_CLICK = True
ROI_LEFT_LABEL = "LEFT"
ROI_RIGHT_LABEL = "RIGHT"
VISUAL_RECHECK_USE_RAW_PROPOSAL = True
VISUAL_RECHECK_LOG_EVERY_CHECK = False
VISUAL_RECHECK_FORCE_PROCESS_THIS_SCAN = False
ALWAYS_RAW_VISUAL_DIFF_ENABLED = True
RAW_VISUAL_DIFF_SIZE = 16
RAW_VISUAL_DIFF_THRESHOLD = 0.22
RAW_VISUAL_DIFF_LOG_THRESHOLD = 0.09
RAW_VISUAL_DIFF_COOLDOWN_SECONDS = 1.0
RAW_VISUAL_DIFF_FORCE_PRIORITY = True
HOLD_KNOWN_ON_FORCED_UNKNOWN = True
HOLD_KNOWN_ON_FORCED_LOW_SCORE = True
ALLOW_CARDBACK_ON_FORCED_VISUAL_CHANGE = False
FORCED_VISUAL_CHANGE_RETRY_SECONDS = 0.75
MANUAL_POINT_CREATE_UNKNOWN_TRACK = False
MANUAL_POINT_FORCE_FRONT = True
MANUAL_POINT_EXPAND_SEARCH_PX = 60
MANUAL_POINT_MIN_CANDIDATE_AREA = 1000
CONSOLE_ERROR_RED = True
STATUS_PANEL_COMPACT_MODE = False
STATUS_PANEL_SHOW_RECOGNIZED_LISTS = False
DETECTION_NUMERIC_SANITIZE = True
DEBUG_VERBOSE_STABILITY_LOGS = False
LOG_SCAN_SUMMARIES = False
LOG_VISUAL_SAME_CHECKS = False
LOG_VISUAL_RECHECK_EVERY_CHECK = False
LAST_KNOWN_REACQUIRE_ENABLED = False
LAST_KNOWN_REACQUIRE_MISSING_SECONDS = 1.25
LAST_KNOWN_REACQUIRE_MAX_TRACKS_PER_SIDE = 4
LAST_KNOWN_REACQUIRE_EXPAND_RATIO = 0.0
LAST_KNOWN_REACQUIRE_MIN_IOU_WITH_CANDIDATES = 0.10
STABLE_REPLACEMENT_CONFIRM_ENABLED = True
STABLE_REPLACEMENT_CONFIRMATIONS = 2
STABLE_REPLACEMENT_CONFIRM_WINDOW_SECONDS = 4.0
STABLE_REPLACEMENT_BYPASS_ON_RAW_CHANGE = True
STATUS_PANEL_POLISHED = True
STATUS_PANEL_SHOW_LEGEND = True
STATUS_PANEL_SHOW_LAST_SENT = True
RAW_VISUAL_CHANGE_CONFIRMATIONS = 4
RAW_VISUAL_CHANGE_CONFIRM_WINDOW_SECONDS = 5.0
RAW_VISUAL_CHANGE_DO_NOT_CLEAR_OVERLAY = True
MANUAL_CHOICE_ENABLED = True
MANUAL_CHOICE_TOP_N = 5
MANUAL_CHOICE_MIN_SCORE = 0.30
MANUAL_CHOICE_SHOW_FOR_LOW_CONFIDENCE = True
MANUAL_CHOICE_LOW_CONFIDENCE_SCORE = 0.62
MANUAL_CHOICE_FORCE_FRONT = True
RIGHT_CLICK_CLEAR_DELETES_TRACK = True
APP_NAME = "Net Ready Eyes"
APP_VERSION_MAJOR = 0
APP_VERSION_MINOR = 902
APP_VERSION = "0.902"
MANUAL_CHOICE_TITLE = "Manual card selector"
MANUAL_CHOICE_POSITION_NEAR_CARD = True
MANUAL_CHOICE_WIDTH = 460
MANUAL_CHOICE_ROW_HEIGHT = 36
MANUAL_CHOICE_MARGIN = 12
CARD_BACK_REFINE_BOX_ENABLED = False
CARD_BACK_REFINE_MIN_AREA_RATIO = 0.18
CARD_BACK_REFINE_MAX_AREA_RATIO = 0.96
CARD_BACK_REFINE_BORDER_DIFF_THRESHOLD = 32.0
MANUAL_CHOICE_OPEN_FOR_IDENTIFIED = True
CARD_BACK_USE_PROPOSAL_BOX = True

# Card-back artifact suppression
# Card-back reads are useful, but they are the easiest class to hallucinate from
# smooth table/playmat regions. Keep them ephemeral and require confirmation.
CARD_BACK_CONFIRMATION_ENABLED = True
CARD_BACK_CONFIRMATIONS = 2
CARD_BACK_CONFIRM_WINDOW_SECONDS = 4.0
CARD_BACK_VISIBLE_MISSING_SECONDS = 0.75
CARD_BACK_EXPIRE_SECONDS = 1.25
CARD_BACK_LAST_KNOWN_REACQUIRE_ENABLED = False

# Automatic-scan ghost suppression
# Weak card-like rectangles left behind on the mat should not become visible or
# queue to OBS as real cards. Manual/direct scans still remain permissive.
AUTO_REJECT_WEAK_NEW_TRACKS = True
AUTO_ACCEPT_NEW_CARD_MIN_SCORE = 0.62
AUTO_ACCEPT_NEW_CARD_MIN_MARGIN = 0.05
AUTO_ACCEPT_BAD_GEOMETRY_MIN_SCORE = 0.70
AUTO_ACCEPT_BAD_GEOMETRY_MIN_MARGIN = 0.10
# Escape hatch: many real Netrunner cards in an angled/overhead webcam view
# are marked sharpness_low/aspect_bad/not_convex, but are still correct when
# the embedding score and top-vs-runner-up margin are convincing. These values
# let those cards appear automatically again without allowing low-margin ghosts.
AUTO_ACCEPT_WEAK_GEOMETRY_IF_SCORE_AT_LEAST = 0.60
AUTO_ACCEPT_WEAK_GEOMETRY_IF_MARGIN_AT_LEAST = 0.15
AUTO_BAD_GEOMETRY_REASONS = (
    "aspect_bad",
    "not_convex",
    "not_card_present",
    "area_ratio_low",
    "area_ratio_high",
    "sharpness_low",
    "low_detail",
)
HIDE_UNCONFIRMED_UNKNOWN_TRACKS = True
HIDE_CARD_BACK_TRACKS = True
CONTROL_SETTINGS_PATH = "netreadyeyes_settings.json"
CONTROL_MODE_DEFAULT = "automatic"
CONTROL_MANUAL_CLICK_RESPECT_QUEUE_DEFAULT = False
CONTROL_QUEUE_SECONDS_MIN = 1.0
CONTROL_QUEUE_SECONDS_MAX = 15.0
CONTROL_QUEUE_SECONDS_DEFAULT = 1.0
STATUS_PANEL_SHOW_CONTROLS = True
CONTROL_GPU_ENABLED_DEFAULT = True
VISUAL_RECHECK_FORCE_REQUEUE_SAME_LABEL = False
MANUAL_SELECTOR_TEXT_SEARCH_ENABLED = True
MANUAL_SELECTOR_SEARCH_MAX_RESULTS = 5
CARD_DIAGNOSTICS_ENABLED = False
CARD_DIAGNOSTICS_TARGET = ""
CARD_DIAGNOSTICS_DIR = Path("diagnostics")
CARD_DIAGNOSTICS_SAVE_ALL_MANUAL_SCANS = True
CARD_DIAGNOSTICS_TOP_N = 8
MANUAL_SCAN_BYPASS_GEOMETRY_REJECTION = True
MANUAL_SCAN_BEST_OF_N_ENABLED = True
MANUAL_SCAN_BEST_OF_N_FRAMES = 5
MANUAL_SCAN_BEST_OF_N_DELAY_SECONDS = 0.08
CARD_MARGIN_OVERRIDES = {"cezve": 0.015}
