"""Fail-closed routing gate for a detected class-3 overlay.

Detection and routing are deliberately separate.  The 4-class model may correctly mark a
settings menu, notification banner, or device window as class 3, but those regions must not
be emitted as a secondary endoscopic view.  This gate looks for endoscopic-content support
inside the proposed crop and otherwise abstains.

The 0.30 development threshold is recorded in eval/95_pip_routability_gate.  It is not a
confirmatory threshold: wider multi-recording validation is still required before a public
performance claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


MIN_ENDOSCOPIC_CONTENT_SUPPORT = 0.30


@dataclass(frozen=True)
class PiPRoutingDecision:
    route: bool
    reason_code: str
    support: float
    semantic_tissue_interior_fraction: float
    warm_colour_fraction: float

    def as_manifest(self) -> dict:
        return asdict(self)


def pip_routing_decision(
        frame_bgr: np.ndarray,
        raw_segmentation: np.ndarray,
        candidate_box: tuple[int, int, int, int] | None,
        *,
        min_support: float = MIN_ENDOSCOPIC_CONTENT_SUPPORT,
) -> PiPRoutingDecision:
    """Return whether a candidate may be emitted as ``pip_crop``.

    Support is the larger of (a) learned tissue probability after hard segmentation inside
    the crop interior and (b) a broad red/orange/pink tissue-colour fraction.  The second
    term is necessary because a strong class-3 rim can occupy the secondary view even when
    the raw mask contains little class 1.  Requiring support from either cue rejects flat
    menus/banners in the development material while retaining endoscopic sub-views.
    """
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be in [0, 1]")
    if candidate_box is None:
        return PiPRoutingDecision(False, "no_reliable_popup_footprint", 0.0, 0.0, 0.0)
    if frame_bgr.ndim != 3 or frame_bgr.shape[:2] != raw_segmentation.shape:
        raise ValueError("frame and raw_segmentation must share HxW")

    x1, y1, x2, y2 = candidate_box
    h, w = raw_segmentation.shape
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return PiPRoutingDecision(False, "invalid_popup_box", 0.0, 0.0, 0.0)

    roi_mask = raw_segmentation[y1:y2, x1:x2]
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    tissue = (roi_mask == 1) | (roi_mask == 2)
    by = max(1, int(round(0.12 * roi_mask.shape[0])))
    bx = max(1, int(round(0.12 * roi_mask.shape[1])))
    interior = tissue[by:max(by + 1, tissue.shape[0] - by),
                      bx:max(bx + 1, tissue.shape[1] - bx)]
    tissue_support = float(interior.mean()) if interior.size else 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (((hue <= 30) | (hue >= 165)) & (sat >= 35) & (val >= 25))
    warm_support = float(warm.mean())
    support = max(tissue_support, warm_support)
    if support < min_support:
        return PiPRoutingDecision(
            False, "overlay_without_endoscopic_content_support", support,
            tissue_support, warm_support)
    return PiPRoutingDecision(
        True, "secondary_endoscopic_view_supported", support,
        tissue_support, warm_support)

