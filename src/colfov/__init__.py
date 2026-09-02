"""ColFOV: semantic-guided four-class segmentation of endoscopic frames, and the
three geometry products derived from it.

The network outputs a segmentation mask and nothing else. The inner-FOV box, the
full-FOV box and the session PiP box are all derived geometry.
"""

from .api import (
    DEFAULT_MONITOR_HZ,
    N_CALIBRATION_FRAMES,
    CalibrationSummary,
    FrameAnalysis,
    MonitorRecord,
    SessionAnalysis,
    analyze_frame,
    analyze_session,
)
from .checkpoint import EXPECTED_SHA256, CheckpointError, load_model, sha256_file
from .segmentation import CLASS_NAMES, class_pixel_counts, segment_frame

__all__ = [
    "CLASS_NAMES", "CalibrationSummary", "CheckpointError", "DEFAULT_MONITOR_HZ",
    "EXPECTED_SHA256", "FrameAnalysis", "MonitorRecord", "N_CALIBRATION_FRAMES",
    "SessionAnalysis", "analyze_frame", "analyze_session", "class_pixel_counts",
    "load_model", "segment_frame", "sha256_file",
]
__version__ = "0.1.0"
