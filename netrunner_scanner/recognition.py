
import cv2
import numpy as np
from PIL import Image

from .config import (
    CONFIDENCE_THRESHOLD,
    AUTO_SEND_TO_OBS,
    DEBUG_SAVE_CROPS,
    DEBUG_CROPS_DIR,
    TRACK_IOU_THRESHOLD,
    STATIONARY_CENTER_THRESHOLD_PX,
    STATIONARY_REFRESH_SECONDS,
    TRACK_EXPIRE_SECONDS,
    OBS_QUEUE_ENABLED,
    USE_COLLECTORVISION_CORNER_REFINER,
    AMBIGUOUS_MATCH_SCORE_THRESHOLD,
    AMBIGUOUS_MATCH_MIN_MARGIN,
    CARD_BACK_EDGE_RATIO_THRESHOLD,
    CARD_BACK_SAT_STD_THRESHOLD,
    CARD_BACK_VAL_STD_THRESHOLD,
    CARD_BACK_CENTER_EDGE_RATIO_THRESHOLD,
    CARD_BACK_CENTER_SAT_STD_THRESHOLD,
    CARD_BACK_CENTER_VAL_STD_THRESHOLD,
    REFINED_BOX_SMOOTHING_ALPHA,
    VISUAL_SIGNATURE_CHANGED_THRESHOLD,
    VISUAL_SIGNATURE_FORCE_RECHECK_THRESHOLD,
    MAX_RECOGNITIONS_PER_SCAN,
    UNKNOWN_CLEAR_FRAMES,
    DISABLE_STABLE_SIGNATURE_RECHECK,
    UNKNOWN_RETRY_COOLDOWN_SECONDS,
    TRACKING_RETRY_COOLDOWN_SECONDS,
    MANUAL_SCAN_ALLOW_CARDBACK,
    MANUAL_CLICK_MIN_EDGE_RATIO,
    MANUAL_CLICK_MIN_CENTER_EDGE_RATIO,
    MANUAL_CLICK_REQUIRE_DETAIL,
    HIDE_STALE_MANUAL_MISSES,
    MANUAL_DRAG_EXPAND_RATIO,
    STABLE_VISUAL_RECHECK_ENABLED,
    STABLE_VISUAL_RECHECK_INTERVAL_SECONDS,
    STABLE_VISUAL_RECHECK_DIFFERENT_THRESHOLD,
    STABLE_VISUAL_RECHECK_SAME_THRESHOLD,
    STABLE_VISUAL_RECHECK_MAX_PER_SCAN,
    VISUAL_RECHECK_USE_RAW_PROPOSAL,
    VISUAL_RECHECK_LOG_EVERY_CHECK,
    VISUAL_RECHECK_FORCE_PROCESS_THIS_SCAN,
    MANUAL_BOX_FIND_CANDIDATES_FIRST,
    MANUAL_BOX_PRESERVE_VISIBLE_MATCHES,
    MANUAL_BOX_MIN_CANDIDATE_AREA,
    MANUAL_BOX_EXPAND_SEARCH_PX,
    MANUAL_DRAG_FORCE_FRONT,
    MANUAL_SCAN_RELAX_THRESHOLDS,
    MANUAL_SCAN_EXPAND_PX,
    MANUAL_CLICK_CARD_WIDTH_PX,
    MANUAL_CLICK_CARD_HEIGHT_PX,
    SAME_SPOT_SIGNATURE_CHECK_ENABLED,
    SAME_SPOT_SIGNATURE_SAME_THRESHOLD,
    SAME_SPOT_SIGNATURE_DIFFERENT_THRESHOLD,
    HOLD_KNOWN_ON_CARDBACK_MISREAD,
    HOLD_KNOWN_ON_ANY_LOW_CONFIDENCE_MISREAD,
    STABILITY_VERBOSE_RECOGNITION_LOGS,
    ALWAYS_RAW_VISUAL_DIFF_ENABLED,
    RAW_VISUAL_DIFF_SIZE,
    RAW_VISUAL_DIFF_THRESHOLD,
    RAW_VISUAL_DIFF_LOG_THRESHOLD,
    RAW_VISUAL_DIFF_COOLDOWN_SECONDS,
    RAW_VISUAL_DIFF_FORCE_PRIORITY,
    HOLD_KNOWN_ON_FORCED_UNKNOWN,
    HOLD_KNOWN_ON_FORCED_LOW_SCORE,
    ALLOW_CARDBACK_ON_FORCED_VISUAL_CHANGE,
    FORCED_VISUAL_CHANGE_RETRY_SECONDS,
    MANUAL_POINT_CREATE_UNKNOWN_TRACK,
    MANUAL_POINT_FORCE_FRONT,
    MANUAL_POINT_EXPAND_SEARCH_PX,
    MANUAL_POINT_MIN_CANDIDATE_AREA,
)
from .crop import crop_candidate
from .detection import find_card_candidates
from .obs_bridge import send_match_to_obs
from .tracking import CardTracker
from .obs_queue import ObsFifoQueue
from .corner_refine import dewarp_candidate_with_collectorvision
from .perf import snapshot as perf_snapshot
from .stability import log_event, log_human_event, recent_events

latest_matches = {"left": [], "right": []}

tracker = CardTracker(
    iou_threshold=TRACK_IOU_THRESHOLD,
    stationary_center_threshold_px=STATIONARY_CENTER_THRESHOLD_PX,
    stationary_refresh_seconds=STATIONARY_REFRESH_SECONDS,
    expire_seconds=TRACK_EXPIRE_SECONDS,
    refined_box_smoothing_alpha=REFINED_BOX_SMOOTHING_ALPHA,
)

obs_queue = ObsFifoQueue()


def legacy_opencv_crop_to_pil(frame, candidate):
    crop = crop_candidate(frame, candidate)

    if crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def visual_signature(pil_image):
    img = np.array(pil_image.convert("RGB"))

    small = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_hist = cv2.calcHist([hsv.astype(np.uint8)], [0], None, [12], [0, 180]).ravel()
    hue_hist = hue_hist / max(1.0, float(hue_hist.sum()))

    sat_mean = float(hsv[:, :, 1].mean() / 255.0)
    val_mean = float(hsv[:, :, 2].mean() / 255.0)

    return {
        "gray": gray,
        "hue_hist": hue_hist.astype(np.float32),
        "sat_mean": sat_mean,
        "val_mean": val_mean,
    }


def visual_signature_distance(a, b):
    if a is None or b is None:
        return 1.0

    gray_dist = float(np.mean(np.abs(a["gray"] - b["gray"])))
    hue_dist = float(np.mean(np.abs(a["hue_hist"] - b["hue_hist"])))
    sat_dist = abs(float(a["sat_mean"]) - float(b["sat_mean"]))
    val_dist = abs(float(a["val_mean"]) - float(b["val_mean"]))

    return (gray_dist * 0.70) + (hue_dist * 0.80) + (sat_dist * 0.25) + (val_dist * 0.25)


def analyze_crop_detail(pil_image):
    img = np.array(pil_image.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = np.count_nonzero(edges) / edges.size

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, w = gray.shape[:2]
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    center_gray = gray[y1:y2, x1:x2]
    center_hsv = hsv[y1:y2, x1:x2]
    center_edges = cv2.Canny(center_gray, 50, 150)

    center_edge_ratio = np.count_nonzero(center_edges) / center_edges.size

    return {
        "edge_ratio": float(edge_ratio),
        "sat_std": float(hsv[:, :, 1].std()),
        "val_std": float(hsv[:, :, 2].std()),
        "center_edge_ratio": float(center_edge_ratio),
        "center_sat_std": float(center_hsv[:, :, 1].std()),
        "center_val_std": float(center_hsv[:, :, 2].std()),
    }


def manual_crop_has_enough_detail(detail):
    if not MANUAL_CLICK_REQUIRE_DETAIL:
        return True

    return (
        detail.get("edge_ratio", 0.0) >= MANUAL_CLICK_MIN_EDGE_RATIO
        or detail.get("center_edge_ratio", 0.0) >= MANUAL_CLICK_MIN_CENTER_EDGE_RATIO
    )


def looks_like_card_back(detail):
    center_low_detail = (
        detail["center_edge_ratio"] < CARD_BACK_CENTER_EDGE_RATIO_THRESHOLD
        and (
            detail["center_sat_std"] < CARD_BACK_CENTER_SAT_STD_THRESHOLD
            or detail["center_val_std"] < CARD_BACK_CENTER_VAL_STD_THRESHOLD
        )
    )

    overall_low_detail = (
        detail["edge_ratio"] < CARD_BACK_EDGE_RATIO_THRESHOLD
        and (
            detail["sat_std"] < CARD_BACK_SAT_STD_THRESHOLD
            or detail["val_std"] < CARD_BACK_VAL_STD_THRESHOLD
        )
    )

    return center_low_detail or overall_low_detail


def get_candidate_pil_image(frame, candidate):
    if USE_COLLECTORVISION_CORNER_REFINER:
        refined = dewarp_candidate_with_collectorvision(frame, candidate)

        if refined is not None and not refined.get("fallback_to_opencv"):
            refined["proposal_box"] = candidate["box"]
            return refined

        fallback_image = legacy_opencv_crop_to_pil(frame, candidate)

        if fallback_image is None:
            return None

        reason = refined.get("reason", "refiner_none") if refined else "refiner_none"

        return {
            "image": fallback_image,
            "refined_box": None,
            "proposal_box": candidate["box"],
            "sharpness": refined.get("sharpness") if refined else None,
            "confidence": refined.get("confidence") if refined else None,
            "fallback_to_opencv": True,
            "reason": reason,
        }

    fallback_image = legacy_opencv_crop_to_pil(frame, candidate)

    if fallback_image is None:
        return None

    return {
        "image": fallback_image,
        "refined_box": None,
        "proposal_box": candidate["box"],
        "sharpness": None,
        "confidence": None,
        "fallback_to_opencv": True,
        "reason": "refiner_disabled",
    }


def card_back_result(refined, detail, signature):
    return {
        "id": "card_back",
        "score": 1.0,
        "margin": None,
        "rotation": "?",
        "refined_box": refined.get("refined_box"),
        "proposal_box": refined.get("proposal_box"),
        "sharpness": refined.get("sharpness"),
        "confidence": refined.get("confidence"),
        "used_fallback": refined.get("fallback_to_opencv", False),
        "refine_reason": "low_detail_card_back",
        "detail_metrics": detail,
        "visual_signature": signature,
    }


def recognize_candidate_crop(frame, candidate, side, candidate_index, catalog):
    refined = get_candidate_pil_image(frame, candidate)

    if refined is None:
        return None

    base_image = refined["image"]
    detail = analyze_crop_detail(base_image)
    signature = visual_signature(base_image)

    if candidate.get("force_card_back"):
        return card_back_result(refined, detail, signature)

    if looks_like_card_back(detail):
        return card_back_result(refined, detail, signature)

    if DEBUG_SAVE_CROPS:
        suffix = "opencv" if refined.get("fallback_to_opencv") else "cvrefined"
        debug_path = DEBUG_CROPS_DIR / f"{side}_candidate_{candidate_index}_{suffix}.jpg"
        base_image.save(debug_path)

    rotation_tests = [
        ("0", base_image),
        ("90", base_image.rotate(90, expand=True)),
        ("180", base_image.rotate(180, expand=True)),
        ("270", base_image.rotate(270, expand=True)),
    ]

    all_results = []

    for rotation_name, pil_image in rotation_tests:
        for result in catalog.search_image(pil_image):
            result["rotation"] = rotation_name
            result["refined_box"] = refined.get("refined_box")
            result["proposal_box"] = refined.get("proposal_box")
            result["sharpness"] = refined.get("sharpness")
            result["confidence"] = refined.get("confidence")
            result["used_fallback"] = refined.get("fallback_to_opencv", False)
            result["refine_reason"] = refined.get("reason")
            result["detail_metrics"] = detail
            result["visual_signature"] = signature
            all_results.append(result)

    all_results.sort(key=lambda r: r["score"], reverse=True)

    if not all_results:
        return None

    best = all_results[0]

    if len(all_results) > 1:
        second = all_results[1]
        margin = best["score"] - second["score"]
        best["margin"] = margin

        if best["score"] < AMBIGUOUS_MATCH_SCORE_THRESHOLD and margin < AMBIGUOUS_MATCH_MIN_MARGIN:
            if looks_like_card_back(detail):
                best["id"] = "card_back"
                best["score"] = 1.0
                best["refine_reason"] = f"ambiguous_low_detail_card_back:{margin:.3f}"
            else:
                best["id"] = "unknown"
                best["refine_reason"] = f"ambiguous_margin:{margin:.3f}"
    else:
        best["margin"] = None

    return best


def candidate_visual_signature(frame, candidate):
    refined = get_candidate_pil_image(frame, candidate)

    if refined is None:
        return None

    return visual_signature(refined["image"])

def candidate_raw_visual_signature(frame, candidate):
    crop = crop_candidate(frame, candidate)

    if crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return visual_signature(Image.fromarray(crop_rgb))


def best_candidate_signature(frame, candidate):
    if VISUAL_RECHECK_USE_RAW_PROPOSAL:
        raw = candidate_raw_visual_signature(frame, candidate)
        if raw is not None:
            return raw

    return candidate_visual_signature(frame, candidate)


def raw_visual_signature_from_candidate(frame, candidate):
    crop = crop_candidate(frame, candidate)

    if crop is None or crop.size == 0:
        return None

    size = int(RAW_VISUAL_DIFF_SIZE)
    if size < 4:
        size = 16

    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Keep this deliberately tiny. It runs for stable known tracks before the
    # expensive recognition path and is meant only to answer: "same picture?"
    return {
        "gray": gray,
        "sat": hsv[:, :, 1].astype(np.float32) / 255.0,
        "val": hsv[:, :, 2].astype(np.float32) / 255.0,
    }


def raw_visual_signature_distance(a, b):
    if a is None or b is None:
        return 1.0

    gray_dist = float(np.mean(np.abs(a["gray"] - b["gray"])))
    sat_dist = float(np.mean(np.abs(a["sat"] - b["sat"])))
    val_dist = float(np.mean(np.abs(a["val"] - b["val"])))

    return gray_dist * 0.70 + sat_dist * 0.18 + val_dist * 0.12


def update_raw_visual_state(frame, candidate, track, side, label, now_time):
    if not ALWAYS_RAW_VISUAL_DIFF_ENABLED:
        return False

    current = raw_visual_signature_from_candidate(frame, candidate)

    if current is None:
        return False

    previous = track.get("last_raw_signature")

    if previous is None:
        track["last_raw_signature"] = current
        track["last_raw_visual_diff"] = 0.0
        return False

    distance = raw_visual_signature_distance(previous, current)
    track["last_raw_visual_diff"] = distance

    known_or_back = label not in (None, "", "unknown", "tracking")
    cooldown_ok = (
        now_time - float(track.get("last_raw_visual_diff_at", 0.0))
        >= RAW_VISUAL_DIFF_COOLDOWN_SECONDS
    )

    if known_or_back and distance >= RAW_VISUAL_DIFF_THRESHOLD and cooldown_ok:
        track["pending_raw_signature"] = current
        track["last_raw_visual_diff_at"] = now_time
        track["forced_visual_change_at"] = now_time
        track["raw_visual_change_pending"] = True
        track["force_new_identification"] = True
        track["displayed"] = False
        track["queued"] = False
        track["refined_box"] = None
        track["last_decision"] = f"raw_visual_change diff={distance:.3f}"

        log_event(
            side,
            track.get("track_id"),
            "raw_visual_change",
            label=label,
            diff=f"{distance:.3f}",
            threshold=f"{RAW_VISUAL_DIFF_THRESHOLD:.3f}",
            box=str(cv2.boundingRect(candidate["box"])),
        )
        log_human_event(
            side,
            track.get("track_id"),
            "raw_visual_change",
            label=label,
            reason=f"raw visual diff {distance:.3f}",
        )
        return True

    # If it is the same image, update the baseline slightly so lighting drift
    # doesn't accumulate forever.
    if distance < RAW_VISUAL_DIFF_LOG_THRESHOLD:
        track["last_raw_signature"] = current
        track["raw_visual_change_pending"] = False

    elif known_or_back and cooldown_ok:
        log_event(
            side,
            track.get("track_id"),
            "raw_visual_same",
            label=label,
            diff=f"{distance:.3f}",
            threshold=f"{RAW_VISUAL_DIFF_THRESHOLD:.3f}",
        )

    return False



def scan_side_for_matches(frame, roi, side, catalog):
    candidates = find_card_candidates(frame, roi)

    scan_items = []
    considered = 0
    skipped_known = 0
    skipped_cooldown = 0
    visual_rechecks_this_scan = 0

    for i, candidate in enumerate(candidates, start=1):
        considered += 1
        track, needs_processing = tracker.update_or_create_track(
            side=side,
            box=candidate["box"],
        )

        label = track.get("label")
        now_time = tracker.current_time()

        raw_visual_changed = update_raw_visual_state(
            frame=frame,
            candidate=candidate,
            track=track,
            side=side,
            label=label,
            now_time=now_time,
        )
        if raw_visual_changed:
            needs_processing = True

        # If this same physical location has not been scanned for a while, first
        # compare the new crop to the old lightweight signature. This avoids
        # the previous failure mode where a stable card was treated as a replacement
        # just because wall-clock time passed between side scans.
        if (
            SAME_SPOT_SIGNATURE_CHECK_ENABLED
            and track.get("needs_same_spot_signature_check", False)
            and track.get("visual_signature") is not None
            and label not in (None, "", "unknown", "tracking", "card_back")
        ):
            current_signature = best_candidate_signature(frame, candidate)
            distance = visual_signature_distance(track.get("visual_signature"), current_signature)
            track["last_signature_distance"] = distance

            if distance <= SAME_SPOT_SIGNATURE_SAME_THRESHOLD:
                track["needs_same_spot_signature_check"] = False
                track["force_new_identification"] = False
                track["displayed"] = True
                track["last_decision"] = f"same_spot_reuse sig={distance:.3f}"
                log_event(side, track.get("track_id"), "same_spot_reuse", label=label, sig=f"{distance:.3f}")
                log_human_event(side, track.get("track_id"), "same_spot_reuse", label=label, sig=f"{distance:.3f}")
                skipped_known += 1
                continue

            if distance >= SAME_SPOT_SIGNATURE_DIFFERENT_THRESHOLD:
                track["needs_same_spot_signature_check"] = False
                track["force_new_identification"] = True
                track["displayed"] = False
                track["queued"] = False
                track["refined_box"] = None
                track["last_decision"] = f"same_spot_changed sig={distance:.3f}"
                log_event(side, track.get("track_id"), "same_spot_changed", label=label, sig=f"{distance:.3f}")
                log_human_event(side, track.get("track_id"), "same_spot_changed", label=label, sig=f"{distance:.3f}")
            else:
                # Borderline case: keep the old ID for now and log it. This is
                # safer for production than bouncing to unknown.
                track["last_decision"] = f"same_spot_borderline sig={distance:.3f}"
                log_event(side, track.get("track_id"), "same_spot_borderline", label=label, sig=f"{distance:.3f}")
                skipped_known += 1
                continue

        # Stable known cards are mostly left alone, but a card flip can keep the
        # same box with very little movement. Compare a lightweight raw proposal
        # signature so card_back <-> face flips actually refresh.
        if (
            STABLE_VISUAL_RECHECK_ENABLED
            and label not in (None, "", "unknown", "tracking")
            and not track.get("force_new_identification", False)
            and track.get("visual_signature") is not None
            and now_time - float(track.get("last_visual_recheck_at", 0.0)) >= STABLE_VISUAL_RECHECK_INTERVAL_SECONDS
            and visual_rechecks_this_scan < STABLE_VISUAL_RECHECK_MAX_PER_SCAN
        ):
            visual_rechecks_this_scan += 1
            current_signature = best_candidate_signature(frame, candidate)
            distance = visual_signature_distance(track.get("visual_signature"), current_signature)
            track["last_signature_distance"] = distance
            track["last_visual_recheck_at"] = now_time

            if VISUAL_RECHECK_LOG_EVERY_CHECK:
                log_event(
                    side,
                    track.get("track_id"),
                    "visual_recheck",
                    label=label,
                    sig=f"{distance:.3f}",
                    threshold=f"{STABLE_VISUAL_RECHECK_DIFFERENT_THRESHOLD:.3f}",
                    box=str(cv2.boundingRect(candidate["box"])),
                )

            if distance >= STABLE_VISUAL_RECHECK_DIFFERENT_THRESHOLD:
                track["force_new_identification"] = True
                track["displayed"] = False
                track["queued"] = False
                track["refined_box"] = None
                track["last_decision"] = f"visual_flip_recheck sig={distance:.3f}"
                log_event(
                    side,
                    track.get("track_id"),
                    "visual_flip_recheck",
                    label=label,
                    sig=f"{distance:.3f}",
                    threshold=f"{STABLE_VISUAL_RECHECK_DIFFERENT_THRESHOLD:.3f}",
                )
                log_human_event(
                    side,
                    track.get("track_id"),
                    "visual_flip_recheck",
                    label=label,
                    sig=f"{distance:.3f}",
                    reason=f"visual signature changed by {distance:.3f}",
                )

                if VISUAL_RECHECK_FORCE_PROCESS_THIS_SCAN:
                    needs_processing = True
            else:
                track["last_decision"] = f"visual_same sig={distance:.3f}"
                skipped_known += 1
                continue

        # Stable known cards are left alone unless the tracker explicitly marked
        # this proposal as a possible replacement. This is the main anti-flicker rule.
        if (
            label not in (None, "", "unknown", "tracking", "card_back")
            and not track.get("force_new_identification", False)
        ):
            skipped_known += 1
            continue

        if label == "unknown":
            last_attempt = float(track.get("last_processed_at", 0.0))
            if now_time - last_attempt < UNKNOWN_RETRY_COOLDOWN_SECONDS:
                skipped_cooldown += 1
                continue

        if track.get("raw_visual_change_pending", False):
            last_attempt = float(track.get("last_processed_at", 0.0))
            if last_attempt > 0 and now_time - last_attempt < FORCED_VISUAL_CHANGE_RETRY_SECONDS:
                skipped_cooldown += 1
                continue

        if label in (None, "", "tracking"):
            last_attempt = float(track.get("last_processed_at", 0.0))
            if last_attempt > 0 and now_time - last_attempt < TRACKING_RETRY_COOLDOWN_SECONDS:
                skipped_cooldown += 1
                continue

        if (
            not needs_processing
            and not DISABLE_STABLE_SIGNATURE_RECHECK
            and track.get("visual_signature") is not None
        ):
            current_signature = best_candidate_signature(frame, candidate)
            distance = visual_signature_distance(track.get("visual_signature"), current_signature)

            if distance >= VISUAL_SIGNATURE_FORCE_RECHECK_THRESHOLD:
                needs_processing = True
                track["displayed"] = False
                track["queued"] = False
                track["refined_box"] = None
                track["refine_reason"] = f"visual_changed:{distance:.3f}"

        if not needs_processing and label not in (None, "", "unknown", "tracking"):
            skipped_known += 1
            continue

        if track.get("raw_visual_change_pending", False) and RAW_VISUAL_DIFF_FORCE_PRIORITY:
            priority = -1
        elif track.get("force_new_identification", False):
            priority = 0
        elif label in (None, "", "tracking"):
            priority = 1
        elif label == "unknown":
            priority = 2
        else:
            priority = 3

        scan_items.append((priority, float(track.get("last_processed_at", 0.0)), i, candidate, track))

    scan_items.sort(key=lambda item: (item[0], item[1]))

    processed = 0
    held_unknown = 0
    label_changes = 0

    for priority, last_processed, i, candidate, track in scan_items[:MAX_RECOGNITIONS_PER_SCAN]:
        processed += 1

        best = recognize_candidate_crop(
            frame=frame,
            candidate=candidate,
            side=side,
            candidate_index=i,
            catalog=catalog,
        )

        previous_label = track.get("label")
        previous_known = previous_label not in (None, "", "unknown", "tracking", "card_back")
        forced_new = bool(track.get("force_new_identification", False))

        # If this is a stable known card that had one failed / weak recognition
        # read, do not overwrite it immediately. The old value is almost always
        # more useful for production than a flickering unknown/card_back box.
        weak_replacement_read = False

        raw_forced = bool(track.get("raw_visual_change_pending", False))

        if best is not None and previous_known:
            best_id = best.get("id")
            best_score = float(best.get("score", 0.0))

            if best_id == "unknown" and (not forced_new or HOLD_KNOWN_ON_FORCED_UNKNOWN):
                weak_replacement_read = True
            elif (
                HOLD_KNOWN_ON_CARDBACK_MISREAD
                and best_id == "card_back"
                and previous_label != "card_back"
                and not (raw_forced and ALLOW_CARDBACK_ON_FORCED_VISUAL_CHANGE)
            ):
                weak_replacement_read = True
            elif (
                HOLD_KNOWN_ON_ANY_LOW_CONFIDENCE_MISREAD
                and best_score < CONFIDENCE_THRESHOLD
                and (not forced_new or HOLD_KNOWN_ON_FORCED_LOW_SCORE)
                and not (raw_forced and best_id not in ("unknown", None, ""))
            ):
                weak_replacement_read = True

        if weak_replacement_read:
            streak = int(track.get("unknown_streak", 0)) + 1
            track["unknown_streak"] = streak
            track["last_processed_at"] = tracker.current_time()
            track["refine_reason"] = f"held_known_after_weak_read:{streak}"
            track["last_decision"] = f"held_known_after_weak_read:{streak}"
            held_unknown += 1
            log_event(
                side,
                track.get("track_id"),
                "held_unknown",
                label=previous_label,
                streak=streak,
                best_id=best.get("id"),
                best_score=f"{float(best.get('score', 0.0)):.2f}",
                best_reason=best.get("refine_reason"),
            )
            log_human_event(
                side,
                track.get("track_id"),
                "held_unknown",
                label=previous_label,
                streak=streak,
                best_reason=best.get("refine_reason"),
            )
            continue

        if best is not None and best.get("id") != "unknown":
            track["unknown_streak"] = 0

        tracker.apply_recognition_result(track, best)
        track["force_new_identification"] = False

        if best is not None and best.get("id") not in (None, "", "unknown"):
            pending_raw = track.get("pending_raw_signature")
            if pending_raw is not None:
                track["last_raw_signature"] = pending_raw
                track["pending_raw_signature"] = None
            track["raw_visual_change_pending"] = False

        current_label = track.get("label")
        current_score = float(track.get("score", 0.0))

        if current_label != previous_label:
            label_changes += 1

        if (
            current_label
            and current_label not in ("unknown", "card_back")
            and current_score >= CONFIDENCE_THRESHOLD
            and (
                current_label != previous_label
                or not track.get("displayed", False)
            )
            and not track.get("queued", False)
        ):
            if OBS_QUEUE_ENABLED:
                obs_queue.enqueue_match(
                    side=side,
                    card_id=current_label,
                    score=current_score,
                    track=track,
                )
            elif AUTO_SEND_TO_OBS:
                send_match_to_obs(side, current_label)

    log_event(
        side,
        "side",
        "scan_summary",
        candidates=considered,
        queued=len(scan_items),
        processed=processed,
        skipped_known=skipped_known,
        skipped_cooldown=skipped_cooldown,
        held_unknown=held_unknown,
        label_changes=label_changes,
    )
    log_human_event(
        side,
        "side",
        "scan_summary",
        candidates=considered,
        queued=len(scan_items),
        processed=processed,
        skipped_known=skipped_known,
        skipped_cooldown=skipped_cooldown,
        held_unknown=held_unknown,
        label_changes=label_changes,
    )

    latest_matches[side] = tracker.get_visible_matches(side)

    if OBS_QUEUE_ENABLED:
        obs_queue.tick()


def tick_obs_queue():
    if OBS_QUEUE_ENABLED:
        obs_queue.tick()


def get_scanner_status():
    return {
        "matches": {
            "left": list(latest_matches["left"]),
            "right": list(latest_matches["right"]),
        },
        "obs_queue": obs_queue.snapshot(),
        "perf": perf_snapshot(),
        "stability_events": recent_events(6),
    }


def refresh_latest_matches_preserving(side, before_matches):
    refreshed = tracker.get_visible_matches(side)

    if not before_matches:
        latest_matches[side] = refreshed
        return

    by_id = {match.get("track_id"): match for match in refreshed}
    merged = []

    for old_match in before_matches:
        track_id = old_match.get("track_id")
        merged.append(by_id.pop(track_id, old_match))

    merged.extend(by_id.values())
    latest_matches[side] = merged



def manual_candidate_at_point(x, y, width=None, height=None):
    from .config import MANUAL_CLICK_CARD_WIDTH_PX, MANUAL_CLICK_CARD_HEIGHT_PX
    import numpy as _np

    w = int(width or MANUAL_CLICK_CARD_WIDTH_PX)
    h = int(height or MANUAL_CLICK_CARD_HEIGHT_PX)

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    box = _np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=_np.intp,
    )

    return {
        "box": box,
        "area": float(w * h),
        "source": "manual_click",
    }

def best_manual_candidate_in_box(frame, x1, y1, x2, y2, side):
    from .roi import rois

    left = max(0, int(min(x1, x2)) - MANUAL_BOX_EXPAND_SEARCH_PX)
    right = min(frame.shape[1] - 1, int(max(x1, x2)) + MANUAL_BOX_EXPAND_SEARCH_PX)
    top = max(0, int(min(y1, y2)) - MANUAL_BOX_EXPAND_SEARCH_PX)
    bottom = min(frame.shape[0] - 1, int(max(y1, y2)) + MANUAL_BOX_EXPAND_SEARCH_PX)

    if right <= left or bottom <= top:
        return None

    # Use a temporary ROI clipped to the user's drawn/search area. This means the
    # user's big green rectangle is a search hint, not the final card box.
    roi = [left, top, right - left, bottom - top]
    candidates = find_card_candidates(frame, roi)

    if not candidates:
        return None

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    scored = []
    for candidate in candidates:
        area = float(candidate.get("area", 0.0))
        if area < MANUAL_BOX_MIN_CANDIDATE_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(candidate["box"])
        ccx = bx + bw / 2.0
        ccy = by + bh / 2.0
        distance = ((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5
        scored.append((distance, -area, candidate))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1]))
    candidate = scored[0][2]
    candidate["source"] = "manual_box_candidate"
    return candidate



def manual_candidate_from_rect(x1, y1, x2, y2):
    import numpy as _np

    left = int(min(x1, x2))
    right = int(max(x1, x2))
    top = int(min(y1, y2))
    bottom = int(max(y1, y2))

    w = max(1, right - left)
    h = max(1, bottom - top)

    expand_x = int(w * MANUAL_DRAG_EXPAND_RATIO)
    expand_y = int(h * MANUAL_DRAG_EXPAND_RATIO)

    left -= expand_x
    right += expand_x
    top -= expand_y
    bottom += expand_y

    box = _np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=_np.intp,
    )

    return {
        "box": box,
        "area": float(max(1, right - left) * max(1, bottom - top)),
        "source": "manual_drag",
    }


def scan_box_for_card(frame, x1, y1, x2, y2, side, catalog):
    before_matches = list(latest_matches.get(side, []))

    search_x1, search_y1, search_x2, search_y2 = x1, y1, x2, y2

    if MANUAL_SCAN_RELAX_THRESHOLDS:
        search_x1 = max(0, x1 - MANUAL_SCAN_EXPAND_PX)
        search_y1 = max(0, y1 - MANUAL_SCAN_EXPAND_PX)
        search_x2 = min(frame.shape[1] - 1, x2 + MANUAL_SCAN_EXPAND_PX)
        search_y2 = min(frame.shape[0] - 1, y2 + MANUAL_SCAN_EXPAND_PX)

    candidate = None

    if MANUAL_BOX_FIND_CANDIDATES_FIRST:
        candidate = best_manual_candidate_in_box(
            frame,
            search_x1,
            search_y1,
            search_x2,
            search_y2,
            side,
        )

    if candidate is None:
        candidate = manual_candidate_from_rect(search_x1, search_y1, search_x2, search_y2)

    track, _needs_processing = tracker.update_or_create_track(side=side, box=candidate["box"])
    track["force_new_identification"] = True
    track["last_decision"] = f"manual_box_scan source={candidate.get('source')}"
    best = recognize_candidate_crop(frame, candidate, side, 998, catalog)

    if best is None and MANUAL_SCAN_RELAX_THRESHOLDS:
        candidate["source"] = "manual_drag_relaxed"
        best = recognize_candidate_crop(frame, candidate, side, 999, catalog)

    previous_label = track.get("label")
    tracker.apply_recognition_result(track, best)

    current_label = track.get("label")
    current_score = float(track.get("score", 0.0))

    if (
        current_label
        and current_label not in ("unknown", "card_back")
        and current_score >= CONFIDENCE_THRESHOLD
        and not track.get("queued", False)
    ):
        if OBS_QUEUE_ENABLED:
            if MANUAL_DRAG_FORCE_FRONT:
                obs_queue.enqueue_front(
                    side=side,
                    card_id=current_label,
                    score=current_score,
                    track=track,
                )
            else:
                obs_queue.enqueue_match(
                    side=side,
                    card_id=current_label,
                    score=current_score,
                    track=track,
                )
        elif AUTO_SEND_TO_OBS:
            send_match_to_obs(side, current_label)

    if MANUAL_BOX_PRESERVE_VISIBLE_MATCHES:
        refresh_latest_matches_preserving(side, before_matches)
    else:
        latest_matches[side] = tracker.get_visible_matches(side)

    log_event(
        side,
        track.get("track_id"),
        "manual_box_scan",
        source=candidate.get("source"),
        old=previous_label,
        new=current_label,
        score=f"{current_score:.2f}",
    )
    return track


def scan_point_for_card(frame, x, y, side, catalog):
    from .roi import rois
    import cv2 as _cv2

    before_matches = list(latest_matches.get(side, []))
    roi = rois.get(side)

    if roi is None:
        return None

    candidates = find_card_candidates(frame, roi)

    containing = []
    for candidate in candidates:
        box = candidate.get("box")
        if box is not None and _cv2.pointPolygonTest(box, (float(x), float(y)), False) >= 0:
            containing.append(candidate)

    if containing:
        containing.sort(key=lambda c: c.get("area", 0.0), reverse=True)
        candidate = containing[0]
    else:
        # User clicked a missed card. Search a larger manual box around that point
        # before falling back to a plain centered crop.
        half_w = (MANUAL_CLICK_CARD_WIDTH_PX // 2) + MANUAL_POINT_EXPAND_SEARCH_PX
        half_h = (MANUAL_CLICK_CARD_HEIGHT_PX // 2) + MANUAL_POINT_EXPAND_SEARCH_PX
        candidate = best_manual_candidate_in_box(
            frame,
            x - half_w,
            y - half_h,
            x + half_w,
            y + half_h,
            side,
        )
        if candidate is None:
            candidate = manual_candidate_at_point(x, y)

    # Recognize first. If this explicit click still produces no useful answer,
    # preserve the existing overlays and do not create a temporary unknown box.
    best = recognize_candidate_crop(frame, candidate, side, 999, catalog)

    if best is None:
        refresh_latest_matches_preserving(side, before_matches)
        log_event(side, "manual", "manual_point_no_result", x=x, y=y, source=candidate.get("source"))
        return None

    best_id = best.get("id")
    best_score = float(best.get("score", 0.0))

    if not MANUAL_POINT_CREATE_UNKNOWN_TRACK and best_id in ("unknown", None, ""):
        refresh_latest_matches_preserving(side, before_matches)
        log_event(
            side,
            "manual",
            "manual_point_no_useful_id",
            x=x,
            y=y,
            best=best_id,
            score=f"{best_score:.2f}",
            reason=best.get("refine_reason"),
            source=candidate.get("source"),
        )
        return None

    if (
        candidate.get("source") == "manual_click"
        and best_id == "card_back"
        and not MANUAL_SCAN_ALLOW_CARDBACK
    ):
        refresh_latest_matches_preserving(side, before_matches)
        log_event(side, "manual", "manual_point_rejected_cardback", x=x, y=y, source=candidate.get("source"))
        return None

    track, _needs_processing = tracker.update_or_create_track(side=side, box=candidate["box"])
    track["force_new_identification"] = True
    track["last_decision"] = f"manual_point_scan source={candidate.get('source')}"

    previous_label = track.get("label")
    tracker.apply_recognition_result(track, best)

    current_label = track.get("label")
    current_score = float(track.get("score", 0.0))

    if (
        current_label
        and current_label not in ("unknown", "card_back")
        and current_score >= CONFIDENCE_THRESHOLD
        and not track.get("queued", False)
    ):
        if OBS_QUEUE_ENABLED:
            if MANUAL_POINT_FORCE_FRONT:
                obs_queue.enqueue_front(
                    side=side,
                    card_id=current_label,
                    score=current_score,
                    track=track,
                )
            else:
                obs_queue.enqueue_match(
                    side=side,
                    card_id=current_label,
                    score=current_score,
                    track=track,
                )
        elif AUTO_SEND_TO_OBS:
            send_match_to_obs(side, current_label)

    refresh_latest_matches_preserving(side, before_matches)
    log_event(
        side,
        track.get("track_id"),
        "manual_point_scan",
        source=candidate.get("source"),
        old=previous_label,
        new=current_label,
        score=f"{current_score:.2f}",
    )
    return track

