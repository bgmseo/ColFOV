"""One forward pass: BGR frame in, four-class label map out.

This is the ONLY network output in the package. Everything else -- the inner-FOV box,
the full-FOV box, the session PiP box -- is geometry derived from these masks.
"""

from __future__ import annotations

import numpy as np
import torch

from .preprocess import logits_to_mask, preprocess

CLASS_NAMES: tuple[str, ...] = (
    "background_ui",        # 0
    "valid_fov_tissue",     # 1
    "black_corner",         # 2
    "popup_overlay",        # 3
)


def segment_frame(model, model_config: dict, frame_bgr: np.ndarray,
                  device="cpu") -> np.ndarray:
    """Return a `(H, W)` uint8 mask at the frame's ORIGINAL resolution.

    Values are class indices into `CLASS_NAMES`. The softmax/crop/upsample/argmax
    order lives in `preprocess.logits_to_mask` and must not be rearranged.
    """
    tensor, params = preprocess(frame_bgr, int(model_config["input_width"]),
                                int(model_config["input_height"]))
    with torch.no_grad():
        logits = model(tensor.to(torch.device(device)))
    return logits_to_mask(logits, params)


def class_pixel_counts(mask: np.ndarray) -> dict[str, int]:
    return {name: int((mask == i).sum()) for i, name in enumerate(CLASS_NAMES)}
