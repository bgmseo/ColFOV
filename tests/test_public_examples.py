"""Keep the public CLI examples aligned with the exported result dataclasses."""

import ast
from dataclasses import fields
from pathlib import Path

from colfov.api import FrameAnalysis, SessionAnalysis


ROOT = Path(__file__).resolve().parents[1]


def _referenced_attributes(path: Path, variable: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == variable
    }


def test_image_example_uses_current_frame_analysis_fields():
    public_fields = {field.name for field in fields(FrameAnalysis)}
    referenced = _referenced_attributes(ROOT / "examples" / "infer_image.py", "result")
    assert referenced <= public_fields


def test_session_example_uses_current_session_analysis_fields():
    public_fields = {field.name for field in fields(SessionAnalysis)}
    referenced = _referenced_attributes(ROOT / "examples" / "infer_session.py", "r")
    assert referenced <= public_fields
