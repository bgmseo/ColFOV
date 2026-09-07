# ColFOV

[한국어](README.ko.md)

ColFOV is a lightweight semantic segmentation model with post-processing for
task-specific field-of-view selection in colonoscopy videos. It distinguishes
endoscopic tissue, optical corners, recorder interfaces and secondary displays,
then derives **video-specific bounding boxes** from multiple frames.

![ColFOV workflow](assets/fig1_workflow.png)

## Model and outputs

The four-class U-Net has 487,316 parameters and uses a 512 × 384 letterboxed RGB
input. Its direct output is a semantic mask. Bounding boxes are derived from
these masks, not predicted by a separate object detector.

| Label | Image content |
|---|---|
| 0 — Background/UI | Recorder interfaces and background outside the primary field |
| 1 — Tissue field | Primary endoscopic image, including tissue, lumen and surgical tools |
| 2 — Optical corner | Non-tissue corners within the rectangular optical field |
| 3 — PiP | Secondary displays, including endoscopic views and non-endoscopic overlays |

ColFOV provides two complementary main-field outputs.

- **Full-FOV bbox** preserves the predicted primary endoscopic field, including its
  optical corners. It is useful when peripheral tissue or findings must remain visible.
- **Inner-FOV bbox** fits a 1.25-aspect rectangle within stable predicted tissue,
  excluding optical corners and recorder interfaces. It retains less peripheral
  tissue, so its suitability depends on the task and the model's training inputs.

Both are video-specific and remain fixed while the capture layout is unchanged.
The study evaluates public and in-house hospital videos and separately tests field
localization on unseen processor families. Depth estimation and annotation-retention
experiments examine the consequences of choosing different input regions.
These findings do not imply that one crop improves every downstream model.

**Optional PiP processing** checks detected secondary displays for endoscopic content
and temporal stability before activating a bbox. A detected display is not necessarily
an endoscopic input, and an active PiP bbox can change over time.

## Installation

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

## Usage

Obtain your image or video separately. No clinical media is included.

```bash
# Semantic mask for one image
python examples/infer_image.py --image /path/to/frame.png --out outputs/image

# Video-specific FOV boxes without PiP monitoring
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session --no-pip

# FOV boxes with optional PiP processing enabled
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session
```

Use `--device cuda` with a CUDA-enabled PyTorch installation. The default checkpoint
is `weights/colfov_b8s3.pt`. Loading verifies its architecture and checksum.

The video example writes `session_result.json` and `calibration_summary.json`.
Main-field coordinates are `full_fov_box` and `inner_fov_box`, in original-frame
pixels as `(x1, y1, x2, y2)`, with exclusive upper bounds.

```python
import json
import cv2

with open("outputs/session/session_result.json") as handle:
    result = json.load(handle)
frame = cv2.imread("/path/to/frame.png")
box = result["inner_fov_box"]
if box is None:
    raise ValueError(result["fov_reason"])
x1, y1, x2, y2 = box
crop = frame[y1:y2, x1:x2]
```

A `null` bbox means the required checks did not establish an output. Consult the
reason code rather than treating an arbitrary fallback as a ColFOV result.
Single-image inference returns a mask and frame-local PiP candidate, not calibrated
video-specific FOV boxes.

With PiP enabled, `monitor_records.jsonl` provides the time-indexed
`active_session_pip_box`. The summary's `final_active_session_pip_box` is only the
last state and must not be applied to all earlier frames.

`--no-pip` (API: `pip_enabled=False`) skips PiP monitoring, content checks and
coordinate updates without changing main-field calibration. It produces no monitor
file. The reason `pip_disabled` means processing was disabled, not that the video
contained no secondary display.

## Data and citation

Public datasets are available from their maintainers under their respective terms.

- [REAL-Colon](https://doi.org/10.1038/s41597-024-03359-0)
- [C3VDv2](https://doi.org/10.1038/s41597-026-07471-1)
- [CAS-Colon](https://doi.org/10.1038/s41597-025-05588-3)

Hospital data are not redistributed. Manuscript citation details and licensing terms
will be added when finalized. This package is for research, not clinical decisions.

## Tests

```bash
pytest -q
```

Tests use synthetic fixtures and cover model loading, mask outputs, bbox derivation
and PiP control.
