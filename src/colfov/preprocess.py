"""Letterbox preprocessing and mask reconstruction, verbatim from the frozen pipeline.

The order in `logits_to_mask` is load-bearing and must not be rearranged:

    softmax -> crop the letterbox padding -> bilinear upsample -> argmax

Upsampling the continuous probabilities BEFORE argmax is what keeps a diagonal corner
boundary smooth. Argmax-ing first and then nearest-neighbour upsampling a 512x384 label
map to a 1920x1080 frame produces a blocky staircase along exactly the boundaries the
FOV geometry is measured from.

Normalisation is `/255` only. There is no ImageNet mean/std step; adding one changes
what the frozen checkpoint sees.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def letterbox_params(src_w: int, src_h: int, target_w: int,
                     target_h: int) -> tuple[float, int, int, int, int]:
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    pad_w, pad_h = target_w - new_w, target_h - new_h
    top, left = pad_h // 2, pad_w // 2
    return scale, new_w, new_h, top, left


def preprocess(frame_bgr: np.ndarray, target_w: int,
               target_h: int) -> tuple[torch.Tensor, tuple]:
    """BGR frame -> letterboxed, /255 normalised NCHW tensor plus the geometry needed
    to put the mask back on the original frame."""
    h, w = frame_bgr.shape[:2]
    scale, new_w, new_h, top, left = letterbox_params(w, h, target_w, target_h)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor, (scale, new_w, new_h, top, left, w, h)


def logits_to_mask(logits: torch.Tensor, params: tuple) -> np.ndarray:
    """Logits -> uint8 label map at the ORIGINAL frame resolution."""
    _scale, new_w, new_h, top, left, orig_w, orig_h = params
    probs = torch.softmax(logits, dim=1)
    cropped = probs[:, :, top:top + new_h, left:left + new_w]
    resized = torch.nn.functional.interpolate(
        cropped, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return resized.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
