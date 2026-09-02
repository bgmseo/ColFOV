"""The frozen checkpoint gate. Nothing here may skip.

A weights-first release whose checkpoint test skips when the weights are absent tests
the one thing the release exists to ship. Missing weights and a hash mismatch are
FAILURES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from colfov import EXPECTED_SHA256, CheckpointError, load_model, sha256_file
from colfov.checkpoint import EXPECTED_STATE_TENSORS
from colfov.model import TinyUNet

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "colfov_b8s3.pt"
CONFIG = ROOT / "configs" / "colfov_b8s3.yaml"


def test_the_checkpoint_ships_with_the_repository():
    assert WEIGHTS.is_file(), f"missing {WEIGHTS}; this release is weights-first"
    assert CONFIG.is_file(), f"missing {CONFIG}"


def test_checkpoint_hash_matches_the_frozen_value():
    assert sha256_file(WEIGHTS) == EXPECTED_SHA256


def test_sha256sums_file_agrees_with_the_checkpoint():
    line = (ROOT / "weights" / "SHA256SUMS").read_text(encoding="utf-8").split()
    assert line[0] == EXPECTED_SHA256
    assert line[1].lstrip("*") == "colfov_b8s3.pt"


def test_checkpoint_loads_under_weights_only():
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    assert sorted(ck) == ["config", "epoch", "model_state"]


def test_model_state_is_exactly_136_tensors_and_nothing_else():
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = ck["model_state"]
    tensors = sum(1 for v in state.values() if torch.is_tensor(v))
    assert tensors == EXPECTED_STATE_TENSORS
    assert len(state) == tensors, "model_state carries a non-tensor entry"


def test_embedded_config_matches_the_shipped_yaml():
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    text = CONFIG.read_text(encoding="utf-8")
    cfg = ck["config"]["model"]
    assert f"num_classes: {cfg['num_classes']}" in text
    assert f"base_channels: {cfg['base_channels']}" in text
    assert f"input_width: {cfg['input_width']}" in text
    assert f"input_height: {cfg['input_height']}" in text


def test_load_model_returns_the_four_class_base_eight_architecture():
    model, cfg = load_model(WEIGHTS)
    assert cfg["num_classes"] == 4 and cfg["base_channels"] == 8
    assert cfg["class_names"] == ["background_ui", "valid_fov_tissue",
                                  "black_corner", "popup_overlay"]
    assert sum(p.numel() for p in model.parameters()) == 487_316


def test_logits_have_the_documented_shape():
    model, cfg = load_model(WEIGHTS)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, cfg["input_height"], cfg["input_width"]))
    assert tuple(out.shape) == (1, 4, 384, 512)


def test_a_wrong_hash_is_refused():
    with pytest.raises(CheckpointError, match="sha256"):
        load_model(WEIGHTS, expected_sha256="0" * 64)


def test_a_missing_checkpoint_is_refused():
    with pytest.raises(CheckpointError, match="not found"):
        load_model(ROOT / "weights" / "does_not_exist.pt")


def test_the_architecture_has_no_defaults():
    """A default 3-class/base-16 TinyUNet is a different model that would load."""
    with pytest.raises(TypeError):
        TinyUNet()
