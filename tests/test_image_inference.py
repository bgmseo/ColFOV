"""Four-class output on a synthetic frame: shape, dtype, class range, determinism."""

from __future__ import annotations

import numpy as np

from colfov import CLASS_NAMES, analyze_frame, class_pixel_counts, segment_frame


def test_class_names_are_the_four_documented_classes():
    assert CLASS_NAMES == ("background_ui", "valid_fov_tissue", "black_corner",
                           "popup_overlay")


def test_mask_is_restored_to_the_input_resolution(loaded, synthetic_frame):
    model, cfg = loaded
    frame = synthetic_frame(h=720, w=1280)
    mask = segment_frame(model, cfg, frame)
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8


def test_mask_values_stay_inside_the_four_classes(loaded, synthetic_frame):
    model, cfg = loaded
    mask = segment_frame(model, cfg, synthetic_frame())
    assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3}


def test_class_pixel_counts_sum_to_the_frame(loaded, synthetic_frame):
    model, cfg = loaded
    mask = segment_frame(model, cfg, synthetic_frame())
    assert sum(class_pixel_counts(mask).values()) == mask.size


def test_inference_is_deterministic(loaded, synthetic_frame):
    model, cfg = loaded
    frame = synthetic_frame()
    a = segment_frame(model, cfg, frame)
    b = segment_frame(model, cfg, frame)
    assert np.array_equal(a, b)


def test_a_non_square_resolution_round_trips(loaded, synthetic_frame):
    model, cfg = loaded
    frame = synthetic_frame(h=1080, w=1350)
    assert segment_frame(model, cfg, frame).shape == (1080, 1350)


def test_frame_analysis_reports_no_session_boxes(loaded, synthetic_frame):
    """A single frame cannot produce inner/full FOV or a session PiP box."""
    model, cfg = loaded
    r = analyze_frame(model, cfg, synthetic_frame())
    assert not hasattr(r, "inner_fov_box")
    assert not hasattr(r, "active_session_pip_box")
    assert isinstance(r.routed, bool)


def test_any_frame_local_box_lies_inside_the_frame(loaded, synthetic_frame):
    model, cfg = loaded
    frame = synthetic_frame(inset=(900, 460, 200, 150))
    r = analyze_frame(model, cfg, frame)
    if r.frame_local_pip_box:
        x1, y1, x2, y2 = r.frame_local_pip_box
        assert 0 <= x1 < x2 <= frame.shape[1]
        assert 0 <= y1 < y2 <= frame.shape[0]
