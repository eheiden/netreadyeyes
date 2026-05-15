
import cv2
import numpy as np

from .config import (
    USE_INTERNAL_CONTOUR_PROPOSALS,
    PROPOSAL_NMS_IOU_THRESHOLD,
    MIN_RECTANGULAR_FILL_RATIO,
    MAX_RECTANGULAR_FILL_RATIO,
    REJECT_NESTED_CANDIDATES,
    NESTED_CONTAINMENT_THRESHOLD,
    NESTED_AREA_RATIO_THRESHOLD,
    MIN_CARD_SHORT_SIDE_PX,
    RECTANGLE_BORDER_CHECK_ENABLED,
    MIN_BORDER_BAND_EDGE_RATIO,
    MIN_BORDER_BANDS_PRESENT,
    MAX_CANDIDATES_PER_SIDE,
    COMPOSITE_PARENT_REJECTION_ENABLED,
    COMPOSITE_CHILD_MIN_AREA_RATIO,
    COMPOSITE_CHILD_MAX_AREA_RATIO,
    COMPOSITE_CHILD_CONTAINMENT_THRESHOLD,
    COMPOSITE_PARENT_MIN_CHILDREN,
    SOLID_BACK_PROPOSALS_ENABLED,
    SOLID_BACK_MIN_AREA_RATIO,
    SOLID_BACK_MAX_AREA_RATIO,
    SOLID_BACK_MIN_SHORT_SIDE_PX,
    SOLID_BACK_MAX_TEXTURE_EDGE_RATIO,
    SOLID_BACK_MIN_COLOR_DISTANCE,
    EDGE_PROPOSAL_MIN_AREA_RATIO,
    WHOLE_CARD_SIZE_FILTER_ENABLED,
    WHOLE_CARD_MIN_AREA_FRACTION_OF_REFERENCE,
    WHOLE_CARD_REFERENCE_TOP_N,
    INNER_BOX_OVERLAP_REJECTION_ENABLED,
    INNER_BOX_OVERLAP_THRESHOLD,
    INNER_BOX_MAX_AREA_RATIO,
    RELATIVE_CARD_SIZE_FILTER_ENABLED,
    RELATIVE_CARD_SIZE_MIN_AREA_FRACTION,
    RELATIVE_CARD_SIZE_MIN_SHORT_SIDE_FRACTION,
    RELATIVE_CARD_SIZE_MIN_REFERENCE_CARDS,
    PARTIAL_CARD_REJECT_ENABLED,
    PARTIAL_CARD_MIN_AREA_FRACTION,
    PARTIAL_CARD_MIN_SHORT_SIDE_FRACTION,
    PARTIAL_CARD_REFERENCE_TOP_N,
    PARTIAL_CARD_MIN_REFERENCES,
)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def candidate_iou(a, b):
    ax, ay, aw, ah = cv2.boundingRect(a["box"])
    bx, by, bw, bh = cv2.boundingRect(b["box"])

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter

    return inter / union if union > 0 else 0.0


def rect_intersection_area(a_rect, b_rect):
    ax, ay, aw, ah = a_rect
    bx, by, bw, bh = b_rect

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def candidate_area_rect(candidate):
    x, y, w, h = cv2.boundingRect(candidate["box"])
    return x, y, w, h, w * h


def suppress_nested_and_composite_candidates(candidates):
    if not REJECT_NESTED_CANDIDATES and not COMPOSITE_PARENT_REJECTION_ENABLED:
        return candidates

    rejected_indexes = set()
    rects = [candidate_area_rect(candidate) for candidate in candidates]

    for i, _candidate_i in enumerate(candidates):
        xi, yi, wi, hi, area_i = rects[i]

        if area_i <= 0:
            rejected_indexes.add(i)
            continue

        child_count = 0

        for j, _candidate_j in enumerate(candidates):
            if i == j:
                continue

            xj, yj, wj, hj, area_j = rects[j]

            if area_j <= 0:
                continue

            if area_i < area_j:
                inter = rect_intersection_area(
                    (xi, yi, wi, hi),
                    (xj, yj, wj, hj),
                )
                containment = inter / area_i
                area_ratio = area_i / area_j

                if (
                    REJECT_NESTED_CANDIDATES
                    and containment >= NESTED_CONTAINMENT_THRESHOLD
                    and area_ratio <= NESTED_AREA_RATIO_THRESHOLD
                    and candidates[i].get("source") != "solid_back"
                ):
                    rejected_indexes.add(i)
                    break

            if area_j < area_i:
                inter = rect_intersection_area(
                    (xj, yj, wj, hj),
                    (xi, yi, wi, hi),
                )
                containment = inter / area_j
                area_ratio = area_j / area_i

                if (
                    COMPOSITE_PARENT_REJECTION_ENABLED
                    and containment >= COMPOSITE_CHILD_CONTAINMENT_THRESHOLD
                    and COMPOSITE_CHILD_MIN_AREA_RATIO <= area_ratio <= COMPOSITE_CHILD_MAX_AREA_RATIO
                ):
                    child_count += 1

        if (
            COMPOSITE_PARENT_REJECTION_ENABLED
            and child_count >= COMPOSITE_PARENT_MIN_CHILDREN
            and candidates[i].get("source") != "solid_back"
        ):
            rejected_indexes.add(i)

    return [
        candidate
        for index, candidate in enumerate(candidates)
        if index not in rejected_indexes
    ]


def suppress_inner_overlap_boxes(candidates):
    if not INNER_BOX_OVERLAP_REJECTION_ENABLED:
        return candidates

    rejected = set()
    rects = [candidate_area_rect(candidate) for candidate in candidates]

    for i, candidate_i in enumerate(candidates):
        if candidate_i.get("source") == "solid_back":
            continue

        xi, yi, wi, hi, area_i = rects[i]

        if area_i <= 0:
            rejected.add(i)
            continue

        for j, candidate_j in enumerate(candidates):
            if i == j:
                continue

            xj, yj, wj, hj, area_j = rects[j]

            if area_j <= area_i:
                continue

            area_ratio = area_i / area_j

            if area_ratio > INNER_BOX_MAX_AREA_RATIO:
                continue

            inter = rect_intersection_area(
                (xi, yi, wi, hi),
                (xj, yj, wj, hj),
            )

            overlap_fraction = inter / area_i

            if overlap_fraction >= INNER_BOX_OVERLAP_THRESHOLD:
                rejected.add(i)
                break

    return [
        candidate
        for idx, candidate in enumerate(candidates)
        if idx not in rejected
    ]


def filter_by_whole_card_size(candidates):
    if not WHOLE_CARD_SIZE_FILTER_ENABLED:
        return candidates

    edge_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("source") == "edge"
    ]

    if len(edge_candidates) < 2:
        return candidates

    edge_areas = sorted(
        [cv2.contourArea(candidate["box"].astype(np.float32)) for candidate in edge_candidates],
        reverse=True,
    )

    reference_pool = edge_areas[:WHOLE_CARD_REFERENCE_TOP_N]

    if not reference_pool:
        return candidates

    reference_area = float(np.median(reference_pool))

    if reference_area <= 0:
        return candidates

    min_area = reference_area * WHOLE_CARD_MIN_AREA_FRACTION_OF_REFERENCE

    filtered = []

    for candidate in candidates:
        if candidate.get("source") != "edge":
            filtered.append(candidate)
            continue

        area = cv2.contourArea(candidate["box"].astype(np.float32))

        if area >= min_area:
            filtered.append(candidate)

    return filtered


def filter_relative_card_sizes(candidates):
    if not RELATIVE_CARD_SIZE_FILTER_ENABLED:
        return candidates

    normal = [
        candidate
        for candidate in candidates
        if candidate.get("source") not in ("solid_back", "manual_click", "manual_drag")
    ]

    if len(normal) < RELATIVE_CARD_SIZE_MIN_REFERENCE_CARDS:
        return candidates

    areas = []
    short_sides = []

    for candidate in normal:
        _x, _y, w, h, rect_area = candidate_area_rect(candidate)
        if rect_area <= 0:
            continue
        areas.append(float(rect_area))
        short_sides.append(float(min(w, h)))

    if len(areas) < RELATIVE_CARD_SIZE_MIN_REFERENCE_CARDS:
        return candidates

    # Use the top half so a partial art/text box cannot drag the reference down.
    areas_sorted = sorted(areas, reverse=True)
    shorts_sorted = sorted(short_sides, reverse=True)
    take = max(RELATIVE_CARD_SIZE_MIN_REFERENCE_CARDS, len(areas_sorted) // 2)

    reference_area = float(np.median(areas_sorted[:take]))
    reference_short = float(np.median(shorts_sorted[:take]))

    min_area = reference_area * RELATIVE_CARD_SIZE_MIN_AREA_FRACTION
    min_short = reference_short * RELATIVE_CARD_SIZE_MIN_SHORT_SIDE_FRACTION

    filtered = []

    for candidate in candidates:
        if candidate.get("source") in ("solid_back", "manual_click", "manual_drag"):
            filtered.append(candidate)
            continue

        _x, _y, w, h, rect_area = candidate_area_rect(candidate)
        short_side = min(w, h)

        if rect_area >= min_area and short_side >= min_short:
            filtered.append(candidate)
        else:
            candidate["rejected_reason"] = (
                f"relative_size area={rect_area:.0f}/{min_area:.0f} "
                f"short={short_side:.0f}/{min_short:.0f}"
            )

    return filtered


def reject_partial_card_boxes(candidates):
    if not PARTIAL_CARD_REJECT_ENABLED:
        return candidates

    normal = [
        candidate
        for candidate in candidates
        if candidate.get("source") not in ("solid_back", "manual_click", "manual_drag")
    ]

    if len(normal) < PARTIAL_CARD_MIN_REFERENCES:
        return candidates

    measures = []

    for candidate in normal:
        _x, _y, w, h, rect_area = candidate_area_rect(candidate)
        if rect_area <= 0:
            continue
        measures.append((float(rect_area), float(min(w, h))))

    if len(measures) < PARTIAL_CARD_MIN_REFERENCES:
        return candidates

    measures.sort(key=lambda item: item[0], reverse=True)
    reference = measures[:max(PARTIAL_CARD_MIN_REFERENCES, min(PARTIAL_CARD_REFERENCE_TOP_N, len(measures)))]

    ref_area = float(np.median([item[0] for item in reference]))
    ref_short = float(np.median([item[1] for item in reference]))

    min_area = ref_area * PARTIAL_CARD_MIN_AREA_FRACTION
    min_short = ref_short * PARTIAL_CARD_MIN_SHORT_SIDE_FRACTION

    filtered = []

    for candidate in candidates:
        if candidate.get("source") in ("solid_back", "manual_click", "manual_drag"):
            filtered.append(candidate)
            continue

        _x, _y, w, h, rect_area = candidate_area_rect(candidate)
        short_side = float(min(w, h))

        if rect_area >= min_area and short_side >= min_short:
            filtered.append(candidate)
        else:
            candidate["rejected_reason"] = (
                f"partial_card area={rect_area:.0f}/{min_area:.0f} "
                f"short={short_side:.0f}/{min_short:.0f}"
            )

    return filtered


def non_max_suppression(candidates):
    kept = []

    candidates = sorted(
        candidates,
        key=lambda c: (c.get("score", 0.0), c.get("area", 0.0)),
        reverse=True,
    )

    for candidate in candidates:
        if all(candidate_iou(candidate, existing) < PROPOSAL_NMS_IOU_THRESHOLD for existing in kept):
            kept.append(candidate)

    return kept


def border_band_score(gray_roi, local_box):
    ordered = order_points(local_box.astype(np.float32))

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    if max_width <= 20 or max_height <= 20:
        return 0, 0.0

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(gray_roi, matrix, (max_width, max_height))

    warped = cv2.resize(warped, (120, 168), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(warped, 50, 150)

    band = 10
    bands = [
        edges[:band, :],
        edges[-band:, :],
        edges[:, :band],
        edges[:, -band:],
    ]

    ratios = [
        float(np.count_nonzero(b) / b.size)
        for b in bands
    ]

    bands_present = sum(
        1
        for ratio in ratios
        if ratio >= MIN_BORDER_BAND_EDGE_RATIO
    )

    return bands_present, float(sum(ratios) / len(ratios))


def contour_to_candidate(contour, gray_roi, roi_offset, roi_area, roi_bounds):
    offset_x, offset_y = roi_offset
    roi_x, roi_y, roi_w, roi_h = roi_bounds

    area = cv2.contourArea(contour)

    if area < roi_area * EDGE_PROPOSAL_MIN_AREA_RATIO:
        return None

    if area > roi_area * 0.24:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect

    if rw <= 0 or rh <= 0:
        return None

    long_side = max(rw, rh)
    short_side = min(rw, rh)

    if short_side <= 0:
        return None

    if short_side < MIN_CARD_SHORT_SIDE_PX:
        return None

    aspect = long_side / short_side

    if not (1.18 <= aspect <= 1.85):
        return None

    rect_area = float(rw * rh)

    if rect_area <= 0:
        return None

    fill_ratio = float(area / rect_area)

    if fill_ratio < MIN_RECTANGULAR_FILL_RATIO:
        return None

    if fill_ratio > MAX_RECTANGULAR_FILL_RATIO:
        return None

    local_box = cv2.boxPoints(rect)
    local_box = np.intp(local_box)

    if RECTANGLE_BORDER_CHECK_ENABLED:
        bands_present, border_score = border_band_score(gray_roi, local_box)

        if bands_present < MIN_BORDER_BANDS_PRESENT:
            return None
    else:
        bands_present = 0
        border_score = 0.0

    box = local_box.copy()
    box[:, 0] += offset_x
    box[:, 1] += offset_y

    margin = 6
    box_x, box_y, box_w, box_h = cv2.boundingRect(box)

    if (
        box_x <= roi_x + margin
        or box_y <= roi_y + margin
        or box_x + box_w >= roi_x + roi_w - margin
        or box_y + box_h >= roi_y + roi_h - margin
    ):
        return None

    aspect_score = 1.0 - min(abs(aspect - 1.40) / 0.55, 1.0)
    fill_score = 1.0 - min(abs(fill_ratio - 0.80) / 0.55, 1.0)

    score = (
        (aspect_score * 0.50)
        + (fill_score * 0.25)
        + (min(border_score / 0.08, 1.0) * 0.25)
    )

    return {
        "box": box,
        "area": area,
        "aspect": aspect,
        "fill_ratio": fill_ratio,
        "border_bands": bands_present,
        "border_score": border_score,
        "score": score,
        "source": "edge",
    }


def local_texture_edge_ratio(gray_roi, local_box):
    ordered = order_points(local_box.astype(np.float32))

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    if max_width <= 20 or max_height <= 20:
        return 1.0

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(gray_roi, matrix, (max_width, max_height))
    warped = cv2.resize(warped, (120, 168), interpolation=cv2.INTER_AREA)

    inner = warped[20:-20, 20:-20]

    if inner.size == 0:
        return 1.0

    edges = cv2.Canny(inner, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def solid_back_candidates(frame, roi, gray_roi):
    if not SOLID_BACK_PROPOSALS_ENABLED:
        return []

    x, y, w, h = roi
    roi_img = frame[y:y + h, x:x + w]
    roi_area = w * h

    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab, (41, 41), 0)

    diff = lab.astype(np.int16) - blurred.astype(np.int16)
    sq = np.sum(diff.astype(np.float32) * diff.astype(np.float32), axis=2)
    sq = np.nan_to_num(sq, nan=0.0, posinf=0.0, neginf=0.0)
    sq = np.maximum(sq, 0.0)
    dist = np.sqrt(sq).astype(np.float32)

    mask = (dist > SOLID_BACK_MIN_COLOR_DISTANCE).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < roi_area * SOLID_BACK_MIN_AREA_RATIO:
            continue

        if area > roi_area * SOLID_BACK_MAX_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect

        if rw <= 0 or rh <= 0:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)

        if short_side < SOLID_BACK_MIN_SHORT_SIDE_PX:
            continue

        aspect = long_side / short_side

        if not (1.18 <= aspect <= 1.85):
            continue

        local_box = cv2.boxPoints(rect)
        local_box = np.intp(local_box)

        texture = local_texture_edge_ratio(gray_roi, local_box)

        if texture > SOLID_BACK_MAX_TEXTURE_EDGE_RATIO:
            continue

        box = local_box.copy()
        box[:, 0] += x
        box[:, 1] += y

        candidates.append({
            "box": box,
            "area": area,
            "aspect": aspect,
            "fill_ratio": 1.0,
            "border_bands": 0,
            "border_score": 0.0,
            "score": 1.1,
            "source": "solid_back",
            "force_card_back": True,
        })

    return candidates


def find_card_candidates(frame, roi):
    x, y, w, h = roi
    roi_img = frame[y:y + h, x:x + w]
    roi_area = w * h

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    retrieval_mode = cv2.RETR_LIST if USE_INTERNAL_CONTOUR_PROPOSALS else cv2.RETR_EXTERNAL

    contours, _ = cv2.findContours(
        edges,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        candidate = contour_to_candidate(
            contour=contour,
            gray_roi=gray,
            roi_offset=(x, y),
            roi_area=roi_area,
            roi_bounds=(x, y, w, h),
        )

        if candidate is not None:
            candidates.append(candidate)

    candidates.extend(solid_back_candidates(frame, roi, gray))

    candidates = suppress_nested_and_composite_candidates(candidates)
    candidates = suppress_inner_overlap_boxes(candidates)
    candidates = filter_by_whole_card_size(candidates)
    candidates = filter_relative_card_sizes(candidates)
    candidates = reject_partial_card_boxes(candidates)
    candidates = non_max_suppression(candidates)

    candidates.sort(key=lambda c: c.get("area", 0.0), reverse=True)

    return candidates[:MAX_CANDIDATES_PER_SIDE]
