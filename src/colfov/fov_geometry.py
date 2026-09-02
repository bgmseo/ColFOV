"""Derived FOV geometry: the inner-FOV and full-FOV boxes.

NOTHING here is a network output. The network emits a four-class mask; every box in
this module is ordinary geometry computed from a set of those masks.

The session contract:

  1. take 24 evenly-spaced calibration frames from the recording;
  2. run the model-free admissibility guard on exactly that sample;
  3. if the admissible FRACTION falls below ``Bounds.min_valid_frame_frac`` (0.50),
     abstain -- both boxes are ``None`` with a reason;
  4. otherwise derive the inner box from stable class-1 tissue and the full box from
     stable class-1-or-2 FOV support.

Step 3 is a FRACTION, not a count. At 24 frames it happens to mean 12, but hardcoding
12 would silently diverge from the frozen rule if the sample size ever changed.

``tests/test_session_geometry.py`` pins the max-inscribed-rectangle geometry against
a set of deterministic reference geometry cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
ENDODAC_ASPECT = 320 / 256  # 1.25 - EndoDAC / this repo's C3VD-SimCol training input (width/height)


class _Checker:
    """O(1) 'is every pixel of this rectangle inside the mask?' via an integral
    image of the INVALID pixels: a rectangle is fully valid iff its invalid
    count is zero.

    Replaces a 4-corner test. Corner-testing is only correct on a CONVEX mask.
    fov_mask() convex-fills, so it was sound for the video path here — but
    max_inscribed_rect is also called on masks that are not hull-filled (the
    per-pixel valid-tissue agreement map in
    scripts/segmentation/final_test_common.py), and there a corner-only test
    happily returns a rectangle whose middle overlaps black corner. Measured on
    a mask with a notch through the centre it returned a box that was 5.4%
    invalid. The integral image is exact for arbitrary masks at the same cost.
    """

    __slots__ = ("integral", "height", "width")

    def __init__(self, mask: np.ndarray) -> None:
        self.integral = cv2.integral((mask == 0).astype(np.uint8)).astype(np.int64)
        self.height, self.width = mask.shape[:2]

    def fits(self, cx: float, cy: float, half_w: float, half_h: float) -> bool:
        x1, y1 = int(round(cx - half_w)), int(round(cy - half_h))
        x2, y2 = int(round(cx + half_w)), int(round(cy + half_h))
        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height or x2 <= x1 or y2 <= y1:
            return False
        integral = self.integral
        return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]) == 0


def _max_half_height(checker: "_Checker", cx: float, cy: float, aspect: float) -> int:
    """Binary search the largest half-height that still fits. Valid because fit
    is monotone in scale: a rectangle that does not fit cannot fit once
    enlarged about the same centre."""
    upper = int(min(cy, checker.height - cy, cx / aspect, (checker.width - cx) / aspect))
    if upper <= 0:
        return 0
    lo, hi, best = 1, upper, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if checker.fits(cx, cy, aspect * mid, mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _search_seeds(mask: np.ndarray) -> list[tuple[float, float]]:
    """Starting centres for the compass search, best first.

    Multi-start is a correctness fix, not a refinement. Compass search only
    moves to a strictly better neighbour, and every rectangle centred inside a
    hole has zero area — so a single search seeded at the centroid gets stuck
    permanently whenever the centroid lands in one (an annular FOV, or a
    picture-in-picture overlay near the middle of the view), and the function
    returns None. The distance transform's peak is the point farthest from any
    invalid pixel, so it is always inside the valid region.
    """
    seeds: list[tuple[float, float]] = []
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    _, _, _, peak = cv2.minMaxLoc(distance)
    seeds.append((float(peak[0]), float(peak[1])))
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        seeds.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    unique: list[tuple[float, float]] = []
    for sx, sy in seeds:
        if all(abs(sx - ux) > 1.0 or abs(sy - uy) > 1.0 for ux, uy in unique):
            unique.append((sx, sy))
    return unique


def max_inscribed_rect(mask: np.ndarray, aspect: float) -> tuple[int, int, int, int] | None:
    """Largest axis-aligned rectangle of the given aspect ratio inside `mask`,
    searching over center position rather than anchoring at the mask's colour
    centroid.

    crop_window_modify.inscribed_rect() always centers the rectangle at the
    mask's centroid and grows it outward — simple and fine when the FOV is
    roughly symmetric, but asymmetrically cropped clinical frames are not: one side
    may be a straight cut from an on-screen panel while the others follow the scope's
    own circular/octagonal boundary, so the centroid isn't equidistant from every
    wall at the target aspect ratio. Centroid-anchoring then locks in a much
    smaller box than the FOV actually has room for. This does compass-search
    (pattern search) over the center position, re-solving the same
    binary-search-on-scale at each candidate center.

    Containment is checked exactly, with an integral image of the invalid
    pixels, so the mask does NOT have to be convex — see _Checker. The search is
    seeded from several starting centres so it cannot stall inside a hole — see
    _search_seeds. The returned box is (x1, y1, x2, y2) with x2/y2 EXCLUSIVE,
    matching how every caller slices it: img[y1:y2, x1:x2]."""
    height, width = mask.shape[:2]
    if not np.any(mask):
        return None
    checker = _Checker(mask)

    def area_and_box(cx: float, cy: float) -> tuple[float, tuple[int, int, int, int] | None]:
        s = _max_half_height(checker, cx, cy, aspect)
        if s <= 0:
            return -1.0, None
        hw = aspect * s
        box = (int(round(cx - hw)), int(round(cy - s)), int(round(cx + hw)), int(round(cy + s)))
        return 4.0 * hw * s, box

    def compass_search(seed_x: float, seed_y: float) -> tuple[float, tuple[int, int, int, int] | None]:
        cx, cy = seed_x, seed_y
        best_area, best_box = area_and_box(cx, cy)
        step = max(width, height) / 8.0
        while step >= 1.0:
            improved = True
            while improved:
                improved = False
                for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                    ncx, ncy = cx + dx, cy + dy
                    if not (0 <= ncx < width and 0 <= ncy < height):
                        continue
                    area, box = area_and_box(ncx, ncy)
                    if area > best_area:
                        best_area, best_box, cx, cy = area, box, ncx, ncy
                        improved = True
            step /= 2.0
        return best_area, best_box

    best_area, best_box = -1.0, None
    for seed_x, seed_y in _search_seeds(mask):
        area, box = compass_search(seed_x, seed_y)
        if area > best_area:
            best_area, best_box = area, box
    return best_box


def clean_agreement_mask(mask: np.ndarray, max_speckle_frac: float = 0.002) -> np.ndarray:
    """Largest connected component, plus enclosed holes below `max_speckle_frac`
    of the frame filled. Returns a 0/1 uint8 mask."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == biggest).astype(np.uint8)
    if max_speckle_frac > 0:
        height, width = binary.shape
        limit = max_speckle_frac * height * width
        count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=4)
        # A background component touching the frame edge is the outside world
        # (black corners, UI bars), not an enclosed hole -- never fill it.
        touches_border = (
            (stats[:, cv2.CC_STAT_LEFT] == 0)
            | (stats[:, cv2.CC_STAT_TOP] == 0)
            | (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] == width)
            | (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] == height)
        )
        fillable = (~touches_border) & (stats[:, cv2.CC_STAT_AREA] <= limit)
        fillable[0] = False  # label 0 is the foreground here, since we inverted
        if fillable.any():
            binary[fillable[labels]] = 1
    return binary


@dataclass(frozen=True)
class Bounds:
    """Calibrated on healthy frames; see --calibrate. Deliberately loose -- the job is
    to reject no-information input, not to judge image quality."""

    min_dynamic_range: float = 30.0
    max_crushed_frac: float = 0.90
    max_saturated_frac: float = 0.50
    min_gradient_energy: float = 1.0
    min_valid_frame_frac: float = 0.50


def frame_features(frame_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    p01, p99 = np.percentile(gray, [1, 99])
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "dynamic_range": float(p99 - p01),
        "crushed_frac": float((gray <= 2).mean()),
        "saturated_frac": float((gray >= 253).mean()),
        "gradient_energy": float(np.mean(np.hypot(gx, gy))),
    }


def frame_admissible(feat: dict[str, float], b: Bounds) -> tuple[bool, str]:
    if feat["dynamic_range"] < b.min_dynamic_range:
        return False, f"dynamic_range {feat['dynamic_range']:.1f} < {b.min_dynamic_range}"
    if feat["crushed_frac"] > b.max_crushed_frac:
        return False, f"crushed_frac {feat['crushed_frac']:.3f} > {b.max_crushed_frac}"
    if feat["saturated_frac"] > b.max_saturated_frac:
        return False, f"saturated_frac {feat['saturated_frac']:.3f} > {b.max_saturated_frac}"
    if feat["gradient_energy"] < b.min_gradient_energy:
        return False, f"gradient_energy {feat['gradient_energy']:.2f} < {b.min_gradient_energy}"
    return True, ""


@dataclass(frozen=True)
class FrameSelection:
    """What the calibration set is, and what was left out of it.

    Contract, frozen 2026-08-09 after the 23 dropped frames were reviewed and approved
    (`outputs/seg_4class/qc/stability/71_calibration_frame_review`):

      * only admitted frames feed the session crop calibration;
      * dropped frames are recorded as `frame_abstained`, never silently discarded;
      * dropped frames are not passed to the classifier or the depth model either --
        a frame too degenerate to calibrate from is too degenerate to score;
      * if the retained fraction falls below `min_valid_frame_frac` the WHOLE session
        abstains, rather than calibrating from the handful of survivors;
      * production, the SS5 crop evaluation and the SS7 degradation probe all call this
        one function, so a stability number can never again come from a frame set the
        shipped pipeline would not have used.
    """

    kept_indices: list[int]
    dropped: list[dict]
    n_input: int
    retained_fraction: float
    session_admitted: bool
    abstain_reason: str

    def as_manifest(self) -> dict:
        return {
            "frames_input": self.n_input,
            "frames_used": len(self.kept_indices),
            "frames_abstained": [d["index"] for d in self.dropped],
            "frame_abstain_reasons": {str(d["index"]): d["reason"] for d in self.dropped},
            "retained_fraction": self.retained_fraction,
            "session_admitted": self.session_admitted,
            "abstain_reason": self.abstain_reason,
        }


def select_calibration_frames(frames: list[np.ndarray],
                              b: Bounds = Bounds()) -> FrameSelection:
    """Split a session's frames into the calibration set and the abstained ones."""
    if not frames:
        return FrameSelection([], [], 0, 0.0, False, "no frames supplied")
    kept, dropped = [], []
    for index, frame in enumerate(frames):
        features = frame_features(frame)
        ok, why = frame_admissible(features, b)
        if ok:
            kept.append(index)
        else:
            dropped.append({"index": index, "reason": why,
                            **{k: round(v, 4) for k, v in features.items()}})
    retained = len(kept) / len(frames)
    admitted = retained >= b.min_valid_frame_frac
    reason = "" if admitted else (
        f"only {100 * retained:.0f}% of frames admissible "
        f"(need {100 * b.min_valid_frame_frac:.0f}%); "
        f"e.g. {dropped[0]['reason'] if dropped else 'unspecified'}")
    return FrameSelection(kept, dropped, len(frames), retained, admitted, reason)


@dataclass(frozen=True)
class SessionCropChoices:
    """Two named crop choices derived from one session-level segmentation.

    ``inner_crop_box`` is the fixed-aspect rectangle fully contained in stable
    class-1 tissue. ``full_crop_box`` is the free-aspect bounding rectangle of
    stable class-1 OR class-2 FOV support. Class 3 is deliberately not support,
    but a popup enclosed by the full rectangle remains visible in the RGB crop.

    Two preservation denominators, because the two boxes answer two questions:

    * ``*_preservation``     -- fraction of stable TISSUE inside the box. This is
      the deployment number for classification/depth input, and the one that
      collapses when a popup punches a hole in the inner rectangle.
    * ``*_fov_preservation`` -- fraction of stable FOV SUPPORT (tissue OR corner)
      inside the box. This is what ``full_crop`` exists to protect, and it is the
      only place the corner retention difference between the two modes shows up.

    ``full_preservation`` and ``full_fov_preservation`` are ~1.0 by construction
    (the full box is the bounding rectangle of the support). They are recorded as
    sanity constants, NOT as evidence that ``full_crop`` preserves more -- the
    informative comparison is ``inner_fov_preservation`` against them.

    ``popup_detection_available`` is False whenever the segmentation model has no
    class-3 channel. A 3-class model cannot distinguish "no popup" from "cannot
    see popups", so ``popup_present=False`` from such a model is not evidence of
    absence and must not be read as one.
    """

    inner_crop_box: tuple[int, int, int, int]
    full_crop_box: tuple[int, int, int, int]
    masks: list[np.ndarray]
    popup_detection_available: bool
    popup_present: bool
    popup_frame_fraction: float
    popup_frames: int
    # `popup_present` is a DETECTION signal and nothing more. It fires when any frame
    # clears the pixel floor, so it is deliberately sensitive and usable as a proposal.
    # The FOOTPRINT is a separate question: on a 2026-08-10 challenge video the model
    # recovered only the PiP border and called ~50% of the window interior tissue, while
    # `popup_present` was True the whole time. Any logic that consumes popup AREA, SHAPE
    # or COVERAGE must gate on `popup_mask_reliable`. Raw/unscoped callers default False;
    # the approved postprocessed 4-class deployment stream explicitly sets it True.
    popup_mask_reliable: bool
    popup_mask_reliable_reason: str
    inner_preservation: float
    full_preservation: float
    inner_fov_preservation: float
    full_fov_preservation: float
    review_required: bool
    review_reasons: tuple[str, ...]


def _box_fraction_of_mask(box: tuple[int, int, int, int], mask: np.ndarray) -> float:
    denominator = int(mask.sum())
    if denominator == 0:
        return 0.0
    x1, y1, x2, y2 = box
    return float(mask[y1:y2, x1:x2].sum() / denominator)


def _full_fov_box(
        stable_fov: np.ndarray,
        *,
        minimum_bottom: int,
) -> tuple[int, int, int, int]:
    """Tight bbox of the stable semantic FOV, with no synthetic bottom inset.

    A genuine lower margin remains excluded because it is absent from ``stable_fov``. If the
    semantic FOV reaches the stored frame edge, the box reaches that edge too. ``minimum_bottom``
    is retained in the signature for compatibility and as a defensive postcondition.
    """
    ys, xs = np.where(stable_fov > 0)
    bottom = int(ys.max()) + 1
    bottom = max(minimum_bottom, bottom)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, bottom


def crop_choices_from_masks(
    masks: list[np.ndarray], agreement_threshold: float = 0.4,
    min_inner_preservation: float | None = None,
    popup_detection_available: bool = True,
    min_popup_pixel_fraction: float = 1e-4,
    min_popup_frame_fraction: float = 0.0,
    popup_mask_reliable: bool = False,
    popup_mask_reliable_reason: str | None = None,
) -> tuple[SessionCropChoices | None, str | None]:
    """Pure mask-domain implementation used by inference and synthetic tests.

    ``min_popup_pixel_fraction`` is the share of the frame a mask must call class
    3 before that frame counts as showing a popup. The default 0.01% is ~150x
    below the measured popup share of the reviewed VAL positives (1.54% of the
    native frame), so a real popup cannot be filtered out, while a handful of
    isolated misclassified pixels no longer flips an entire session.

    ``min_popup_frame_fraction`` is the share of a session's frames that must
    qualify before ``popup_present`` is set. It defaults to 0.0 -- ANY qualifying
    frame marks the session -- because popups are intermittent and no calibrated
    value has been frozen from approved session references yet. Raise it only
    from such a calibration, never to quieten a noisy session.
    """
    if not masks:
        return None, "no masks were supplied to compute crop choices from"
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        return None, "mask resolution mismatch within session"
    if not 0 < agreement_threshold <= 1:
        return None, "agreement_threshold must be in (0, 1]"
    if not 0 <= min_popup_pixel_fraction <= 1:
        return None, "min_popup_pixel_fraction must be in [0, 1]"
    if not 0 <= min_popup_frame_fraction <= 1:
        return None, "min_popup_frame_fraction must be in [0, 1]"
    if popup_mask_reliable and not popup_detection_available:
        return None, "popup_mask_reliable requires a model with a class-3 channel"

    # Floor of 1: a fraction of 0 means "no area filter", not "every frame counts".
    min_popup_pixels = max(1.0, min_popup_pixel_fraction * float(shape[0] * shape[1]))
    tissue_counts = np.zeros(shape, dtype=np.int32)
    full_counts = np.zeros(shape, dtype=np.int32)
    popup_frames = 0
    for mask in masks:
        tissue_counts += mask == 1
        full_counts += (mask == 1) | (mask == 2)
        popup_frames += int(int((mask == 3).sum()) >= min_popup_pixels)

    denominator = float(len(masks))
    stable_tissue = clean_agreement_mask(
        ((tissue_counts / denominator) >= agreement_threshold).astype(np.uint8))
    stable_full = clean_agreement_mask(
        ((full_counts / denominator) >= agreement_threshold).astype(np.uint8))
    if not np.any(stable_tissue):
        return None, (f"no pixels reached the {agreement_threshold:.0%} stable-tissue "
                      "agreement threshold")
    if not np.any(stable_full):
        return None, (f"no pixels reached the {agreement_threshold:.0%} full-FOV "
                      "agreement threshold")

    inner_box = max_inscribed_rect(stable_tissue, ENDODAC_ASPECT)
    if inner_box is None:
        return None, "max_inscribed_rect returned None for inner_crop"
    full_box = _full_fov_box(stable_full, minimum_bottom=inner_box[3])

    popup_fraction = popup_frames / denominator
    popup_present = bool(popup_detection_available and popup_frames > 0
                         and popup_fraction >= min_popup_frame_fraction)
    inner_preservation = _box_fraction_of_mask(inner_box, stable_tissue)
    full_preservation = _box_fraction_of_mask(full_box, stable_tissue)
    inner_fov_preservation = _box_fraction_of_mask(inner_box, stable_full)
    full_fov_preservation = _box_fraction_of_mask(full_box, stable_full)
    reasons: list[str] = []
    # This threshold is intentionally configurable, not silently guessed.  A
    # calibrated value must be frozen from approved session references before
    # it becomes a deployment default.
    if (popup_present and min_inner_preservation is not None
            and inner_preservation < min_inner_preservation):
        reasons.append(
            f"popup present and inner preservation {inner_preservation:.4f} "
            f"< configured minimum {min_inner_preservation:.4f}")
    # Fail closed rather than fail silent: an operator who configured a
    # popup-conditioned guard gets a review flag when the model cannot evaluate
    # it, instead of a clean pass that only means "class 3 was never predicted".
    if min_inner_preservation is not None and not popup_detection_available:
        reasons.append(
            "min_inner_preservation is configured but the segmentation model has no "
            "class-3 channel, so popup presence could not be evaluated")

    return SessionCropChoices(
        inner_crop_box=inner_box,
        full_crop_box=full_box,
        masks=masks,
        popup_detection_available=popup_detection_available,
        popup_present=popup_present,
        popup_frame_fraction=popup_fraction,
        popup_frames=popup_frames,
        popup_mask_reliable=popup_mask_reliable,
        popup_mask_reliable_reason=(popup_mask_reliable_reason or (
            "raw/unscoped mask path; reliable popup footprint was not asserted")),
        inner_preservation=inner_preservation,
        full_preservation=full_preservation,
        inner_fov_preservation=inner_fov_preservation,
        full_fov_preservation=full_fov_preservation,
        review_required=bool(reasons),
        review_reasons=tuple(reasons),
    ), None
