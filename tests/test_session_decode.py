"""analyze_session must fail closed on a decode failure, and leave no artifact."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from colfov import analyze_session


def _write_video(path, n_frames, fps=30.0, h=240, w=320):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((xx - w / 2) / (w * 0.36)) ** 2 + ((yy - h / 2) / (h * 0.46)) ** 2 <= 1
    rng = np.random.default_rng(3)
    for _ in range(n_frames):
        f = np.zeros((h, w, 3), np.uint8)
        f[ell] = np.clip(np.array([60, 90, 180], np.float32)
                         * (0.85 + 0.3 * rng.random((int(ell.sum()), 1))), 0, 255)
        vw.write(f)
    vw.release()


class _FailingCapture:
    """Reports `n_declared` frames but stops delivering after `n_real`."""

    def __init__(self, path, n_declared=300, n_real=90, fps=30.0, size=(240, 320)):
        self._i, self._n_real, self._n_declared = 0, n_real, n_declared
        self._fps, self._h, self._w = fps, *size

    def isOpened(self):
        return True

    def get(self, prop):
        return (float(self._n_declared) if prop == cv2.CAP_PROP_FRAME_COUNT
                else self._fps)

    def _frame(self):
        yy, xx = np.mgrid[0:self._h, 0:self._w]
        f = np.zeros((self._h, self._w, 3), np.uint8)
        ell = (((xx - self._w / 2) / (self._w * 0.36)) ** 2
               + ((yy - self._h / 2) / (self._h * 0.46)) ** 2 <= 1)
        f[ell] = (60, 90, 180)
        return f

    def read(self):
        if self._i >= self._n_real:
            return False, None
        self._i += 1
        return True, self._frame()

    def grab(self):
        if self._i >= self._n_real:
            return False
        self._i += 1
        return True

    def release(self):
        pass


def test_a_short_stream_raises_with_the_frame_index_and_leaves_no_records(
        tmp_path, loaded, monkeypatch):
    """The container declares more frames than it can deliver."""
    model, cfg = loaded
    records = tmp_path / "monitor_records.jsonl"
    monkeypatch.setattr("colfov.api.cv2.VideoCapture", _FailingCapture)
    with pytest.raises(RuntimeError) as err:
        analyze_session(tmp_path / "declared_300_real_90.mp4", model, cfg,
                        monitor_records_path=records)
    msg = str(err.value)
    assert "frame index" in msg and "90" in msg
    assert not records.exists(), "a failed run must not leave a records artifact"


def test_a_truncated_file_is_also_refused(tmp_path, loaded):
    model, cfg = loaded
    good = tmp_path / "good.mp4"
    _write_video(good, 120)
    truncated = tmp_path / "truncated.mp4"
    data = good.read_bytes()
    truncated.write_bytes(data[: int(len(data) * 0.55)])
    records = tmp_path / "monitor_records.jsonl"
    with pytest.raises(RuntimeError):
        analyze_session(truncated, model, cfg, monitor_records_path=records)
    assert not records.exists()


def test_an_unreadable_path_raises_before_anything_is_written(tmp_path, loaded):
    model, cfg = loaded
    records = tmp_path / "monitor_records.jsonl"
    with pytest.raises(RuntimeError):
        analyze_session(tmp_path / "missing.mp4", model, cfg,
                        monitor_records_path=records)
    assert not records.exists()


def test_a_complete_video_reports_the_fractional_cadence(tmp_path, loaded):
    model, cfg = loaded
    video = tmp_path / "ok.mp4"
    _write_video(video, 300, fps=30.0)            # 10 s
    records = tmp_path / "monitor_records.jsonl"
    r = analyze_session(video, model, cfg, monitor_hz=5.0,
                        monitor_records_path=records)
    assert r.effective_monitor_hz == 5.0
    assert r.n_frames_monitored == 50             # 10 s at 5 Hz
    assert records.is_file()
    assert sum(1 for _ in records.open(encoding="utf-8")) == r.n_frames_monitored
    # Epoch bookkeeping is unambiguous and internally consistent.
    assert r.final_pip_epoch >= 0
    assert r.n_pip_epoch_rotations == r.n_pip_unlock_events
    assert not hasattr(r, "n_pip_epochs")
