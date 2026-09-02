"""Fail-closed, no-retrain popup footprint repair.

The frozen model often predicts only a rim on a live picture-in-picture window.  When the
raw RGB window is a separate non-black connected component, the predicted class-3 rim can
identify that component without knowing its layout, video, timestamp, or annotation note.
The component's convex hull is then filled as class 3.

If the window touches the main image, it belongs to the largest raw component and this
repair deliberately does nothing.  That is the fail-closed boundary: attached windows
need either an already-solid model prediction or a different, validated method.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

POPUP_CLASS = 3
MIN_SOLID_BBOX_FILL = 0.80
BLACK_ABOVE = 18
MIN_RAW_COMPONENT_PIXELS = 5000
MIN_RIM_PIXELS = 50


@dataclass(frozen=True)
class ComponentRepair:
    label: int
    raw_area: int
    rim_pixels: int
    hull_area: int
    added_pixels: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class RepairResult:
    mask: np.ndarray
    footprint: np.ndarray
    repairs: tuple[ComponentRepair, ...]
    main_component_area: int

    @property
    def applied(self) -> bool:
        return bool(self.repairs)

    @property
    def added_pixels(self) -> int:
        return sum(item.added_pixels for item in self.repairs)


def postprocess_popup_masks(
        frames: list[np.ndarray], raw_masks: list[np.ndarray]) -> tuple[list[RepairResult],
                                                                        list[np.ndarray]]:
    """Apply the unchanged repair frame-by-frame and return the final mask stream."""
    if len(frames) != len(raw_masks):
        raise ValueError("frames and raw_masks must have the same length")
    repairs = [rim_anchored_component_fill(frame, mask)
               for frame, mask in zip(frames, raw_masks)]
    return repairs, [result.mask for result in repairs]


def _filled_hull(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(component)
    points = np.vstack(contours)
    hull = cv2.convexHull(points)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [hull], -1, 1, cv2.FILLED)
    return filled


def rim_anchored_component_fill(
        frame: np.ndarray,
        prediction: np.ndarray,
        *,
        popup_class: int = POPUP_CLASS,
        black_above: int = BLACK_ABOVE,
        min_raw_component_pixels: int = MIN_RAW_COMPONENT_PIXELS,
        min_rim_pixels: int = MIN_RIM_PIXELS) -> RepairResult:
    """Fill detached raw-image components that contain enough predicted popup rim pixels.

    No location, aspect ratio, video id, or human note is used.  The largest non-black raw
    component is always treated as the main image and is never filled by this function.
    """
    if frame.ndim != 3 or frame.shape[:2] != prediction.shape:
        raise ValueError("frame and prediction must share HxW; frame must be colour")

    content = (frame.max(axis=2) > black_above).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    if count <= 1:
        return RepairResult(prediction.copy(), np.zeros_like(prediction, dtype=bool), (), 0)

    main_label = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    main_area = int(stats[main_label, cv2.CC_STAT_AREA])
    popup_rim = prediction == popup_class
    repaired = prediction.copy()
    footprint = np.zeros_like(prediction, dtype=bool)
    repairs = []

    for label in range(1, count):
        if label == main_label:
            continue
        raw_area = int(stats[label, cv2.CC_STAT_AREA])
        if raw_area < min_raw_component_pixels:
            continue
        component = (labels == label).astype(np.uint8)
        rim_pixels = int((popup_rim & component.astype(bool)).sum())
        if rim_pixels < min_rim_pixels:
            continue

        hull = _filled_hull(component)
        hull_area = int(hull.sum())
        footprint |= hull.astype(bool)
        before = repaired == popup_class
        repaired[hull.astype(bool)] = popup_class
        added = int(((repaired == popup_class) & ~before).sum())
        repairs.append(ComponentRepair(
            label=label,
            raw_area=raw_area,
            rim_pixels=rim_pixels,
            hull_area=hull_area,
            added_pixels=added,
            bbox=(int(stats[label, cv2.CC_STAT_LEFT]),
                  int(stats[label, cv2.CC_STAT_TOP]),
                  int(stats[label, cv2.CC_STAT_WIDTH]),
                  int(stats[label, cv2.CC_STAT_HEIGHT])),
        ))

    return RepairResult(repaired, footprint, tuple(repairs), main_area)


def popup_crop_box(
        popup_repair: RepairResult,
        final_mask: np.ndarray,
        *,
        allow_raw_fallback: bool = True,
        min_popup_pixel_fraction: float = 1e-4,
        min_fallback_bbox_fill: float = MIN_SOLID_BBOX_FILL,
) -> tuple[int, int, int, int] | None:
    """Return a frame-local PiP crop from a repair footprint or solid final class 3.

    The fallback accepts only a component whose pixels fill at least
    ``min_fallback_bbox_fill`` of its own bbox.  This lets an already-solid attached PiP
    produce a crop while preventing a hollow/C-shaped prediction from being promoted to
    a crop unless the rim-anchored repair established its footprint first.
    """
    if not 0 <= min_popup_pixel_fraction <= 1:
        raise ValueError("min_popup_pixel_fraction must be in [0, 1]")
    if not 0 <= min_fallback_bbox_fill <= 1:
        raise ValueError("min_fallback_bbox_fill must be in [0, 1]")
    target = popup_repair.footprint.astype(np.uint8)
    if target.any():
        ys, xs = np.nonzero(target)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    if not allow_raw_fallback:
        return None

    popup = (final_mask == POPUP_CLASS).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(popup, 8)
    min_pixels = max(1, int(np.ceil(min_popup_pixel_fraction * popup.size)))
    eligible = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_fill = area / float(width * height)
        if area >= min_pixels and bbox_fill >= min_fallback_bbox_fill:
            eligible.append(label)
    if not eligible:
        return None
    label = max(eligible, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    x, y, width, height = map(int, stats[label, :4])
    return x, y, x + width, y + height
