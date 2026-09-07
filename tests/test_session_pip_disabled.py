"""`pip_enabled=False` runs calibration only, and says so rather than implying absence.

The FOV boxes must be identical to a normal run, because they come from the same
24-frame calibration and the same four-class segmentation. What must NOT happen is a
second decode pass, a routing content gate, a coordinate update, or a
`monitor_records.jsonl` that a reader could mistake for "no PiP was present".
"""

from __future__ import annotations

import cv2
import numpy as np

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


def test_disabling_pip_keeps_the_fov_boxes_and_skips_the_monitoring_pass(
        tmp_path, loaded):
    model, cfg = loaded
    video = tmp_path / "ok.mp4"
    _write_video(video, 300, fps=30.0)                      # 10 s

    on = analyze_session(video, model, cfg, monitor_hz=5.0,
                         monitor_records_path=tmp_path / "on.jsonl")
    off = analyze_session(video, model, cfg, monitor_hz=5.0,
                          monitor_records_path=tmp_path / "off.jsonl",
                          pip_enabled=False)

    # The geometry that does not depend on PiP is unchanged.
    assert off.inner_fov_box == on.inner_fov_box
    assert off.full_fov_box == on.full_fov_box
    assert off.fov_reason == on.fov_reason
    assert off.calibration == on.calibration

    # Nothing walked the recording a second time.
    assert on.n_frames_monitored == 50
    assert off.n_frames_monitored == 0
    assert not (tmp_path / "off.jsonl").exists()
    assert off.monitor_records_path is None

    # The PiP fields report the disabled state, not a measured absence.
    assert off.pip_enabled is False and on.pip_enabled is True
    assert off.final_session_pip_reason == "pip_disabled"
    assert off.final_active_session_pip_box is None
    assert (off.final_pip_epoch, off.n_pip_epoch_rotations,
            off.n_pip_lock_events, off.n_pip_unlock_events) == (0, 0, 0, 0)

    # The declared cadence is still reported, so the two modes stay comparable.
    assert off.effective_monitor_hz == on.effective_monitor_hz


def test_the_cli_exposes_the_same_switch(tmp_path, loaded):
    """--no-pip must reach analyze_session, and the payload must record the mode."""
    import json
    import runpy
    import sys

    video = tmp_path / "ok.mp4"
    _write_video(video, 90, fps=30.0)
    out = tmp_path / "session_no_pip"
    argv = ["infer_session.py", "--video", str(video), "--out", str(out),
            "--weights", "weights/colfov_b8s3.pt", "--no-pip"]
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path("examples/infer_session.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    finally:
        sys.argv = old

    payload = json.loads((out / "session_result.json").read_text(encoding="utf-8"))
    assert payload["pip_enabled"] is False
    assert payload["final_session_pip_reason"] == "pip_disabled"
    assert payload["monitor_records"] is None
    assert not (out / "monitor_records.jsonl").exists()
