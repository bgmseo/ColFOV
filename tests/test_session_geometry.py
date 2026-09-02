"""The 24-frame FOV contract: selection, the 50% fraction rule, and abstention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from colfov.api import N_CALIBRATION_FRAMES, _evenly_spaced
from colfov.fov_geometry import (
    ENDODAC_ASPECT, Bounds, crop_choices_from_masks, max_inscribed_rect,
    select_calibration_frames,
)

FIX = Path(__file__).resolve().parent / "fixtures"


def test_the_calibration_sample_is_exactly_24_evenly_spaced_frames():
    assert N_CALIBRATION_FRAMES == 24
    idx = _evenly_spaced(56_272, N_CALIBRATION_FRAMES)
    assert len(idx) == 24 and idx[0] == 0 and idx[-1] == 56_271
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1, "sample must be evenly spaced"


def test_the_admissibility_rule_is_a_fraction_not_a_count_of_12():
    """Hardcoding 12 would silently diverge if the sample size ever changed."""
    assert Bounds().min_valid_frame_frac == 0.50


def test_twelve_of_24_is_admitted_and_eleven_is_not(synthetic_frame):
    good, dead = synthetic_frame(), np.zeros((720, 1280, 3), np.uint8)
    twelve = select_calibration_frames([good] * 12 + [dead] * 12)
    eleven = select_calibration_frames([good] * 11 + [dead] * 13)
    assert twelve.session_admitted and twelve.retained_fraction == 0.5
    assert not eleven.session_admitted
    assert eleven.abstain_reason


def test_an_all_degenerate_session_abstains_before_any_model_runs():
    dead = [np.zeros((720, 1280, 3), np.uint8)] * 24
    sel = select_calibration_frames(dead)
    assert not sel.session_admitted and sel.retained_fraction == 0.0


def test_both_fov_boxes_are_derived_from_stable_masks():
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), np.uint8)
    mask[((yy - 120) / 100) ** 2 + ((xx - 160) / 140) ** 2 <= 1] = 1     # tissue
    mask[((yy - 120) / 115) ** 2 + ((xx - 160) / 155) ** 2 <= 1] |= 0
    choices, err = crop_choices_from_masks([mask] * 12)
    assert err is None and choices is not None
    for box in (choices.inner_crop_box, choices.full_crop_box):
        x1, y1, x2, y2 = box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    ix1, iy1, ix2, iy2 = choices.inner_crop_box
    fx1, fy1, fx2, fy2 = choices.full_crop_box
    assert (fx2 - fx1) >= (ix2 - ix1) and (fy2 - fy1) >= (iy2 - iy1)


def test_no_fallback_box_is_invented_when_there_is_no_support():
    empty = np.zeros((240, 320), np.uint8)
    choices, err = crop_choices_from_masks([empty] * 12)
    assert choices is None and err


def test_max_inscribed_rect_matches_the_reference_geometry_cases():
    """Reference geometry cases: deterministic synthetic ellipses with expected boxes."""
    fx = json.loads((FIX / "inscribed_rect_cases.json").read_text(encoding="utf-8"))
    assert fx["aspect"] == ENDODAC_ASPECT
    for case in fx["cases"]:
        h, w = case["shape"]
        cy, cx, ry, rx = case["ellipse"]
        yy, xx = np.mgrid[0:h, 0:w]
        m = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.uint8)
        got = max_inscribed_rect(m, ENDODAC_ASPECT)
        assert (list(got) if got else None) == case["expected_box"], case["index"]
