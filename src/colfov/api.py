"""The public ColFOV contract: one four-class mask, three derived geometry products.

    network output      mask [H, W] in {0, 1, 2, 3}
    derived geometry    inner_fov_box, full_fov_box, active_session_pip_box

`active_session_pip_box` is CAUSAL and TIME-DEPENDENT. It is not a property of a
recording; it is the box that was live at one instant, given only the evidence that had
arrived by then. A recording whose picture-in-picture window moves has more than one
correct answer over its length, and asking for "the" session box is asking the wrong
question. `SessionAnalysis.final_active_session_pip_box` is the LAST state only and
must never be applied retrospectively to earlier timestamps.

All three box fields always exist. `None` is a fail-closed abstention carrying a reason
code -- never an error, and never replaced by a fallback box.

Memory is bounded and independent of video length: masks are not retained, per-frame
records are optionally streamed to JSONL, and no per-frame image is written by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .cadence import PhaseAccumulator, effective_hz
from .checkpoint import load_model  # noqa: F401  (re-exported)
from .fov_geometry import Bounds, crop_choices_from_masks, select_calibration_frames
from .popup import popup_crop_box, postprocess_popup_masks
from .routing import pip_routing_decision
from .segmentation import CLASS_NAMES, class_pixel_counts, segment_frame
from .session_pip import (
    METHOD_MEDIAN,
    PipQualificationState,
    PipQualifier,
    RoutedObservation,
    active_session_box,
    apply_observation,
)

BBox = tuple[int, int, int, int]

# The verified run monitored at this cadence. It is NOT the native frame rate: a 59 fps
# recording is sampled at 5 Hz, so 393,297 frames become 33,331 observations.
DEFAULT_MONITOR_HZ = 5.0
N_CALIBRATION_FRAMES = 24


@dataclass(frozen=True)
class FrameAnalysis:
    """One frame. Produces NO session-level FOV box -- that needs a calibration set."""

    mask: np.ndarray
    class_pixel_counts: dict
    popup_detected: bool
    frame_local_pip_box: BBox | None      # diagnostics, not a headline product
    routed: bool
    routing_reason: str
    routing_support: float


@dataclass(frozen=True)
class MonitorRecord:
    """One monitored observation, as it happened."""

    frame_index: int
    timestamp_sec: float
    active_session_pip_box: BBox | None
    pip_state: str                        # "unlocked" | "locked"
    pip_epoch: int
    pip_event: str | None                 # "lock" | "unlock" | "relock" | None
    pip_reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["active_session_pip_box"] = (list(self.active_session_pip_box)
                                       if self.active_session_pip_box else None)
        return d


@dataclass(frozen=True)
class CalibrationSummary:
    n_requested: int
    n_admissible: int
    retained_fraction: float
    min_valid_frame_frac: float
    session_admitted: bool
    abstain_reason: str


@dataclass(frozen=True)
class SessionAnalysis:
    inner_fov_box: BBox | None
    full_fov_box: BBox | None
    fov_reason: str
    # LAST state only. Do not apply to earlier timestamps -- see the module docstring.
    final_active_session_pip_box: BBox | None
    final_session_pip_reason: str
    # Epoch bookkeeping, stated unambiguously. `final_pip_epoch` is the zero-based
    # identifier of the LAST epoch; `n_pip_epoch_rotations` is how many confirmed
    # relocations occurred. They are equal only because epoch 0 is the first.
    final_pip_epoch: int
    n_pip_epoch_rotations: int
    n_pip_lock_events: int
    n_pip_unlock_events: int
    monitor_hz: float
    effective_monitor_hz: float
    calibration: CalibrationSummary
    n_frames_in_source: int
    n_frames_monitored: int
    monitor_records_path: Path | None
    # False when the caller asked for FOV boxes only. The PiP fields above then carry
    # their "never observed" values rather than a measurement, and no monitoring pass ran.
    pip_enabled: bool = True


def analyze_frame(model, model_config: dict, frame_bgr: np.ndarray,
                  device="cpu") -> FrameAnalysis:
    mask = segment_frame(model, model_config, frame_bgr, device)
    repairs, finals = postprocess_popup_masks([frame_bgr], [mask])
    box = popup_crop_box(repairs[0], finals[0])
    decision = pip_routing_decision(frame_bgr, mask, box)
    return FrameAnalysis(
        mask=mask,
        class_pixel_counts=class_pixel_counts(mask),
        popup_detected=bool((finals[0] == 3).any()),
        frame_local_pip_box=tuple(int(v) for v in box) if box else None,
        routed=bool(decision.route),
        routing_reason=decision.reason_code,
        routing_support=float(decision.support),
    )


def _evenly_spaced(n_total: int, k: int) -> list[int]:
    return np.linspace(0, max(n_total - 1, 0), k).round().astype(int).tolist()


def analyze_session(video_path, model, model_config: dict, *, device="cpu",
                    monitor_hz: float = DEFAULT_MONITOR_HZ,
                    method: str = METHOD_MEDIAN,
                    monitor_records_path=None,
                    pip_enabled: bool = True) -> SessionAnalysis:
    """Fixed-cadence causal monitoring of one recording.

    Two passes over the file, deliberately: the FOV boxes come from a 24-frame
    calibration sample, and the PiP box comes from monitoring the whole recording in
    time order. They are not the same population and must not be conflated.

    `pip_enabled=False` runs calibration only. The monitoring pass, the routing content
    gate and the PiP coordinate updates are all skipped, so nothing walks the recording
    a second time and no `monitor_records.jsonl` is written. The FOV boxes come from the
    same four-class segmentation as always, and the PiP fields report `pip_disabled`
    rather than a measured absence -- a caller must not read them as "no PiP was there".
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    if n_total <= 0 or src_fps <= 0:
        raise RuntimeError(f"video reports {n_total} frames at {src_fps} fps")

    # --- pass 1: calibration -> inner_fov_box, full_fov_box -----------------------
    wanted = set(_evenly_spaced(n_total, N_CALIBRATION_FRAMES))
    calib: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    i, hi = 0, max(wanted)
    while i <= hi:
        if i in wanted:
            ok, frame = cap.read()
            if ok:
                calib.append(frame)
        elif not cap.grab():
            break
        i += 1
    cap.release()

    selection = select_calibration_frames(calib)
    bounds = Bounds()
    summary = CalibrationSummary(
        n_requested=N_CALIBRATION_FRAMES, n_admissible=len(selection.kept_indices),
        retained_fraction=selection.retained_fraction,
        min_valid_frame_frac=bounds.min_valid_frame_frac,
        session_admitted=selection.session_admitted,
        abstain_reason=selection.abstain_reason)

    inner_box = full_box = None
    fov_reason = selection.abstain_reason or "ok"
    if selection.session_admitted:
        masks = [segment_frame(model, model_config, calib[i], device)
                 for i in selection.kept_indices]
        choices, err = crop_choices_from_masks(
            masks, popup_detection_available=int(model_config["num_classes"]) >= 4)
        if choices is None:
            fov_reason = err or "crop choices unavailable"
        else:
            inner_box = tuple(int(v) for v in choices.inner_crop_box)
            full_box = tuple(int(v) for v in choices.full_crop_box)
            fov_reason = "ok"
    del calib

    # --- pass 2: causal fixed-cadence monitoring -> active_session_pip_box ---------
    # Fractional phase, not a rounded stride: see colfov.cadence for why.
    hz = effective_hz(src_fps, monitor_hz)
    if not pip_enabled:
        # No second decode, no content gate, no coordinate updates. The effective
        # cadence is still reported so the two modes stay comparable on paper.
        return SessionAnalysis(
            inner_fov_box=inner_box, full_fov_box=full_box, fov_reason=fov_reason,
            final_active_session_pip_box=None,
            final_session_pip_reason="pip_disabled",
            final_pip_epoch=0, n_pip_epoch_rotations=0,
            n_pip_lock_events=0, n_pip_unlock_events=0,
            monitor_hz=monitor_hz, effective_monitor_hz=hz, calibration=summary,
            n_frames_in_source=n_total, n_frames_monitored=0,
            monitor_records_path=None, pip_enabled=False)
    phase = PhaseAccumulator(effective_hz=hz)
    h, w = None, None
    qualifier = state = None
    locks = unlocks = 0
    last_box, last_reason = None, "pip_session_box_unavailable"
    sink = open(monitor_records_path, "w", encoding="utf-8") if monitor_records_path else None
    n_monitored = 0
    cap = cv2.VideoCapture(str(video_path))
    completed = False
    try:
        for i in range(n_total):
            t = i / src_fps
            if not phase.due(t):
                if not cap.grab():
                    raise RuntimeError(
                        f"decode failed at frame index {i} of {n_total} "
                        f"({video_path}); the container reported more frames than it "
                        "could deliver. A partial session result is not returned.")
                continue
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"decode failed at frame index {i} of {n_total} ({video_path}); "
                    "a partial session result is not returned.")
            if qualifier is None:
                h, w = frame.shape[:2]
                qualifier = PipQualifier(frame_width=w, frame_height=h, method=method)
                state = PipQualificationState(monitor_interval_sec=1.0 / hz)
            analysis = analyze_frame(model, model_config, frame, device)
            obs = (RoutedObservation(frame_id=f"f{i:06d}", t=t,
                                     box=analysis.frame_local_pip_box)
                   if analysis.routed and analysis.frame_local_pip_box else None)
            outcome = apply_observation(obs, t, pip_state=state, qualifier=qualifier,
                                        frame_width=w, frame_height=h)
            box, reason = active_session_box(state, t, w, h)
            event = None
            if outcome.locked_now:
                locks += 1
                event = "relock" if unlocks else "lock"
            elif outcome.epoch_rotated:
                unlocks += 1
                event = "unlock"
            record = MonitorRecord(
                frame_index=i, timestamp_sec=round(t, 6),
                active_session_pip_box=tuple(int(v) for v in box) if box else None,
                pip_state="locked" if state.locked else "unlocked",
                pip_epoch=qualifier.epoch, pip_event=event, pip_reason=reason)
            if sink:
                sink.write(json.dumps(record.as_dict()) + "\n")
            last_box, last_reason = record.active_session_pip_box, reason
            n_monitored += 1
            del frame, analysis
        completed = True
    finally:
        cap.release()
        if sink:
            sink.close()
        # Never leave a monitor_records.jsonl that reads as a successful run.
        if not completed and monitor_records_path:
            Path(monitor_records_path).unlink(missing_ok=True)

    return SessionAnalysis(
        inner_fov_box=inner_box, full_fov_box=full_box, fov_reason=fov_reason,
        final_active_session_pip_box=last_box, final_session_pip_reason=last_reason,
        final_pip_epoch=(qualifier.epoch if qualifier else 0),
        n_pip_epoch_rotations=unlocks,
        n_pip_lock_events=locks, n_pip_unlock_events=unlocks,
        monitor_hz=monitor_hz, effective_monitor_hz=hz, calibration=summary,
        n_frames_in_source=n_total, n_frames_monitored=n_monitored,
        monitor_records_path=Path(monitor_records_path) if monitor_records_path else None,
        pip_enabled=True)
