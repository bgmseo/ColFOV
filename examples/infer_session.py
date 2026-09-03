"""One recording -> four-class segmentation plus all three geometry products.

Offline, fixed-cadence, causal. The monitor samples at `--monitor-hz` (5 Hz by
default, matching the verified run); it does not evaluate every native frame. The
per-observation record in `monitor_records.jsonl` is the authoritative PiP output:
`active_session_pip_box` is the box that was live at THAT timestamp. The summary's
`final_active_session_pip_box` is the last state only and must not be applied to
earlier timestamps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from colfov import CLASS_NAMES, analyze_session, load_model


def _resolve_device(name: str) -> str:
    """Fail with an explanation rather than a bare torch assertion."""
    import torch
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {name} was requested but this PyTorch build has no CUDA "
            "support. Install a CUDA build of torch, or run with --device cpu.")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="weights/colfov_b8s3.pt")
    ap.add_argument("--out", default="outputs/session_example")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--monitor-hz", type=float, default=5.0, dest="monitor_hz")
    args = ap.parse_args()
    args.device = _resolve_device(args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.weights, device=args.device)
    r = analyze_session(args.video, model, cfg, device=args.device,
                        monitor_hz=args.monitor_hz,
                        monitor_records_path=out / "monitor_records.jsonl")

    calib = asdict(r.calibration)
    (out / "calibration_summary.json").write_text(
        json.dumps(calib, indent=2), encoding="utf-8")

    payload = {
        "segmentation_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "inner_fov_box": list(r.inner_fov_box) if r.inner_fov_box else None,
        "full_fov_box": list(r.full_fov_box) if r.full_fov_box else None,
        "fov_reason": r.fov_reason,
        "final_active_session_pip_box": (list(r.final_active_session_pip_box)
                                         if r.final_active_session_pip_box else None),
        "final_session_pip_reason": r.final_session_pip_reason,
        "final_state_caveat": (
            "final_active_session_pip_box is the LAST state of a causal, "
            "time-dependent quantity. It must not be applied retrospectively to "
            "earlier timestamps. Use monitor_records.jsonl for per-observation "
            "boxes."),
        "final_pip_epoch": r.final_pip_epoch,
        "n_pip_epoch_rotations": r.n_pip_epoch_rotations,
        "n_pip_lock_events": r.n_pip_lock_events,
        "n_pip_unlock_events": r.n_pip_unlock_events,
        "monitor_hz": r.monitor_hz,
        "effective_monitor_hz": r.effective_monitor_hz,
        "n_calibration_frames_requested": calib["n_requested"],
        "n_calibration_frames_admissible": calib["n_admissible"],
        "min_valid_frame_frac": calib["min_valid_frame_frac"],
        "n_frames_in_source": r.n_frames_in_source,
        "n_frames_monitored": r.n_frames_monitored,
        "monitor_records": "monitor_records.jsonl",
        "status": "ok" if r.inner_fov_box else f"fov_abstained: {r.fov_reason}",
    }
    (out / "session_result.json").write_text(json.dumps(payload, indent=2),
                                             encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
