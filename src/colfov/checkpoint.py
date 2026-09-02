"""Loading the frozen ColFOV checkpoint, with no way to load it wrongly.

Every failure mode this module guards against has a specific cost:

  * a silent architecture default -- `TinyUNet` used to default to 3 classes and 16
    base channels, which is a DIFFERENT model. Constructing that and then loading a
    4-class/base-8 state dict non-strictly gives a model that runs and is wrong.
    `colfov.model.TinyUNet` therefore has no architecture defaults at all, and this
    loader reads the shape from the checkpoint's own embedded config.
  * a swapped or truncated checkpoint -- `expected_sha256` is checked against the file
    before anything is loaded.
  * arbitrary code execution -- `torch.load` is called with `weights_only=True`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .model import TinyUNet

# The frozen ColFOV checkpoint. Any other file is a different model.
EXPECTED_SHA256 = "df4054b8d2a22413ae232b7ea2ce01cbe70c838031167b5afb72c71ce9670713"
EXPECTED_NUM_CLASSES = 4
EXPECTED_BASE_CHANNELS = 8
EXPECTED_STATE_TENSORS = 136


class CheckpointError(RuntimeError):
    """The checkpoint is missing, altered, or not the architecture it claims."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path, *, device: str = "cpu",
               expected_sha256: str | None = EXPECTED_SHA256):
    """Return `(model, model_config)` for the frozen checkpoint.

    Fails loudly rather than degrading: a hash mismatch, a missing key, a wrong class
    count or a state dict that does not fit the constructed model all raise.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    if expected_sha256 is not None:
        got = sha256_file(path)
        if got != expected_sha256:
            raise CheckpointError(
                f"checkpoint sha256 {got} != expected {expected_sha256}. This is not "
                "the frozen ColFOV checkpoint.")

    ck = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state", "config"):
        if key not in ck:
            raise CheckpointError(f"checkpoint has no {key!r}; keys are {sorted(ck)}")

    cfg = ck["config"]["model"]
    n_classes, base = int(cfg["num_classes"]), int(cfg["base_channels"])
    if n_classes != EXPECTED_NUM_CLASSES or base != EXPECTED_BASE_CHANNELS:
        raise CheckpointError(
            f"embedded config says {n_classes}-class/base-{base}; ColFOV is "
            f"{EXPECTED_NUM_CLASSES}-class/base-{EXPECTED_BASE_CHANNELS}")

    state = ck["model_state"]
    n_tensors = sum(1 for v in state.values() if torch.is_tensor(v))
    if n_tensors != EXPECTED_STATE_TENSORS or n_tensors != len(state):
        raise CheckpointError(
            f"model_state has {len(state)} entries, {n_tensors} tensors; expected "
            f"{EXPECTED_STATE_TENSORS} tensors and nothing else")

    model = TinyUNet(num_classes=n_classes, base_channels=base)
    model.load_state_dict(state, strict=True)          # strict: no silent mismatch
    model.eval().to(torch.device(device))
    return model, dict(cfg)
