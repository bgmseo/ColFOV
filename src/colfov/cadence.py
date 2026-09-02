"""Fixed-cadence monitor scheduling, by fractional phase rather than integer stride.

`round(source_fps / monitor_hz)` is wrong whenever the ratio is not an integer. At
29.97 fps and 5 Hz it rounds to a stride of 6, giving 4.995 Hz; at 25 fps and 5 Hz it
happens to be exact; at 59 fps it rounds 11.8 to 12 and yields 4.917 Hz. The error is
small per frame and unbounded over a recording: at 59 fps a 393,297-frame recording
monitored at 5 Hz should yield 33,331 observations, and a rounded stride yields 32,775.

`PhaseAccumulator` carries the fractional remainder forward, so the long-run rate is
exactly `min(source_fps, monitor_hz)`. It advances the next due instant by whole
multiples of the interval from the FIRST timestamp rather than from the last accepted
frame, so one late frame cannot drag the whole schedule, and a gap catches up in whole
intervals instead of firing a burst.

`tests/test_cadence.py` pins this against a deterministic cadence contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def effective_hz(source_fps: float, target_hz: float) -> float:
    """`min(target, source)`. A source cannot be sampled faster than it exists."""
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz}")
    return float(min(float(target_hz), float(source_fps)))


@dataclass
class PhaseAccumulator:
    """Decides, per frame, whether the monitor should run -- without integer stride."""

    effective_hz: float
    _next_due_t: float | None = field(default=None, init=False, repr=False)
    n_due: int = field(default=0, init=False)
    n_skipped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.effective_hz <= 0:
            raise ValueError("effective_hz must be positive")

    @property
    def interval_sec(self) -> float:
        return 1.0 / self.effective_hz

    def due(self, t: float) -> bool:
        if self._next_due_t is None:
            self._next_due_t = t + self.interval_sec
            self.n_due += 1
            return True
        if t + 1e-9 >= self._next_due_t:
            missed = int((t - self._next_due_t) / self.interval_sec) + 1
            self._next_due_t += missed * self.interval_sec
            self.n_due += 1
            return True
        self.n_skipped += 1
        return False


def scheduled_indices(n_frames: int, source_fps: float,
                      target_hz: float) -> list[int]:
    """The frame indices a monitor at `target_hz` would observe.

    Timestamps are `index / source_fps`, the same derived clock the verified run used.
    """
    hz = effective_hz(source_fps, target_hz)
    phase = PhaseAccumulator(effective_hz=hz)
    return [i for i in range(int(n_frames)) if phase.due(i / float(source_fps))]


def scheduled_count(n_frames: int, source_fps: float, target_hz: float) -> int:
    hz = effective_hz(source_fps, target_hz)
    phase = PhaseAccumulator(effective_hz=hz)
    n = 0
    for i in range(int(n_frames)):
        if phase.due(i / float(source_fps)):
            n += 1
    return n
