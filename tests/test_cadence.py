"""Fractional-phase monitor cadence, not a rounded integer stride."""

from __future__ import annotations

import pytest

from colfov.cadence import PhaseAccumulator, effective_hz, scheduled_count


def test_effective_rate_is_min_of_source_and_target():
    assert effective_hz(59.0, 5.0) == 5.0
    assert effective_hz(1.0, 5.0) == 1.0      # cannot sample faster than the source
    assert effective_hz(25.0, 30.0) == 25.0


@pytest.mark.parametrize("fps", [59.0, 29.97, 25.0, 1.0])
def test_long_run_rate_equals_the_effective_rate(fps):
    seconds = 1000
    n = int(round(fps * seconds))
    want = min(fps, 5.0)
    got = scheduled_count(n, fps, 5.0)
    assert abs(got / (n / fps) - want) < 1e-3, f"{fps} fps -> {got} observations"


@pytest.mark.parametrize("fps,expected", [(59.0, 5000), (29.97, 5000),
                                          (25.0, 5000), (1.0, 1000)])
def test_exact_counts_over_1000_seconds(fps, expected):
    assert scheduled_count(int(round(fps * 1000)), fps, 5.0) == expected


def test_the_established_59fps_contract_is_reproduced():
    """393,297 frames at 59 fps monitored at 5 Hz gave 33,331 observations."""
    assert scheduled_count(393_297, 59.0, 5.0) == 33_331


def test_a_rounded_stride_would_miss_that_contract():
    """Guards the reason this module exists."""
    stride = round(59.0 / 5.0)                 # 12, not 11.8
    assert len(range(0, 393_297, stride)) != 33_331


def test_a_gap_catches_up_in_whole_intervals_without_a_burst():
    """A gap must not fire once per missed interval.

    The catch-up lands `_next_due_t` on a whole multiple of the interval, which
    floating-point rounding can place exactly at the current timestamp; the following
    frame is then due immediately. That is the verified implementation's behaviour and
    is reproduced here rather than "corrected" -- at most one extra observation
    follows a gap, and the long-run rate is unaffected.
    """
    phase = PhaseAccumulator(effective_hz=5.0)
    assert phase.due(0.0)
    assert not phase.due(0.1)
    assert phase.due(0.2)
    assert phase.due(10.0)                       # the gap fires exactly once
    fired = sum(1 for k in range(1, 40) if phase.due(10.0 + k * 0.05))
    assert fired <= 11, "a gap must not fire once per missed interval"


def test_no_drift_accumulates_over_a_long_run():
    n = int(round(59.0 * 3600))                  # one hour at 59 fps
    got = scheduled_count(n, 59.0, 5.0)
    assert abs(got - 3600 * 5) <= 1


def test_a_non_positive_rate_is_refused():
    with pytest.raises(ValueError):
        effective_hz(0.0, 5.0)
    with pytest.raises(ValueError):
        PhaseAccumulator(effective_hz=0.0)
