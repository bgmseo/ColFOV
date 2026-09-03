"""One image -> four-class mask + frame-local diagnostics.

A single frame CANNOT produce the session-level FOV boxes: those need a calibration
sample. Nor can it produce a session PiP box, which is causal and needs a timeline.
This example therefore emits the mask and the frame-local diagnostics only, and says
so in its own output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from colfov import CLASS_NAMES, analyze_frame, load_model

OVERLAY_BGR = {0: (140, 60, 15), 2: (30, 30, 220), 3: (40, 200, 255)}


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
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="weights/colfov_b8s3.pt")
    ap.add_argument("--out", default="outputs/image_example")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.device = _resolve_device(args.device)

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    model, cfg = load_model(args.weights, device=args.device)
    result = analyze_frame(model, cfg, frame, device=args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "mask.png"), result.mask)

    overlay = frame.copy()
    for cls, colour in OVERLAY_BGR.items():
        idx = result.mask == cls
        if idx.any():
            overlay[idx] = np.clip(overlay[idx] * 0.4 + np.array(colour) * 0.6,
                                   0, 255).astype(np.uint8)
    if result.frame_local_pip_box:
        x1, y1, x2, y2 = result.frame_local_pip_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(out / "overlay.png"), overlay)

    payload = {
        "segmentation_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "class_pixel_counts": result.class_pixel_counts,
        "popup_detected": result.popup_detected,
        "routed": result.routed,
        "routing_reason": result.routing_reason,
        "routing_support": round(result.routing_support, 6),
        "diagnostics": {"frame_local_pip_box": list(result.frame_local_pip_box)
                        if result.frame_local_pip_box else None},
        "note": ("A single frame yields no inner_fov_box, no full_fov_box and no "
                 "active_session_pip_box. The FOV boxes need a 24-frame calibration "
                 "sample and the PiP box is causal over a timeline; use "
                 "examples/infer_session.py."),
    }
    (out / "frame_result.json").write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
