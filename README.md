# ColFOV

[한국어](README.ko.md)

Crop colonoscopy frames down to the part that is actually endoscopic image, and find the
picture-in-picture window when one is present.

![ColFOV workflow](assets/fig1_workflow.png)

A colonoscopy frame arrives wrapped in things that are not tissue: a processor UI border,
black corners where the optics do not reach, letterbox padding, and sometimes a
picture-in-picture window showing a second view. ColFOV segments the frame into those four
classes and turns the result into three rectangles you can crop with — a Full-FOV bbox
around the whole usable field, an Inner-FOV bbox that fits inside stable tissue at a fixed
1.25 aspect, and a PiP bbox for the sub-window. The first two are computed once per
recording and reused; the PiP bbox is tracked over time, because the window moves.

Code and model weights are published here for scientific transparency. Licensing terms are
not settled yet and will be added when the manuscript is submitted.

## Install

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

## Use it

No image or video data ships with this repository, so point it at your own files.

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session
```

That writes `session_result.json` with the three rectangles, `calibration_summary.json`,
and `monitor_records.jsonl` with one row per monitored instant. The boxes are integer
`(x1, y1, x2, y2)` with exclusive `x2`/`y2`, in your frame's own resolution, so they slice
straight into a crop:

```python
import cv2, json

result = json.load(open("outputs/session/session_result.json"))
frame  = cv2.imread("/path/to/frame.png")

box = result["inner_fov_box"]
crop = frame[box[1]:box[3], box[0]:box[2]] if box else frame
```

A single frame works too, though it can only give you the mask and a frame-local PiP
candidate — the FOV boxes need a calibration sample and the PiP box needs a timeline:

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image
```

Add `--device cuda` to either command if your PyTorch build has CUDA; a plain
`pip install torch` often does not, and the examples will tell you so rather than dying on
an assertion.

## When a box comes back `null`

Any of the three can be `null`, and that is the intended answer rather than a failure. It
means the evidence for that rectangle was not there — too few usable calibration frames,
or a PiP window that never showed enough stable evidence to lock onto — and a reason code
alongside it says which condition was not met. Substituting a fallback rectangle is the one
thing you should not do, because a wrong crop corrupts everything downstream of it without
ever raising an error.

## Reading the PiP box

The two FOV boxes hold for the whole recording, so you can read them once from
`session_result.json` and forget about them. The PiP box is different. If the window
relocates mid-procedure there is no single rectangle that is correct for the recording, so
`final_active_session_pip_box` in the summary is only the last state and applying it to
earlier timestamps gives you the wrong box for everything before the move. Read it per
instant instead:

```python
live = {}
for line in open("outputs/session/monitor_records.jsonl"):
    r = json.loads(line)
    live[r["frame_index"]] = r["active_session_pip_box"]   # box, or None
```

Each row also carries `timestamp_sec`, `pip_state`, `pip_epoch`, `pip_reason`, and a
`pip_event` of `lock`, `unlock`, `relock` or `null`, so you can see exactly when the window
appeared, moved or went away.

<details>
<summary><b>Output schema</b></summary>

`session_result.json`, with placeholder values — these are the keys and shapes, not
results:

```json
{
  "inner_fov_box": [null, null, null, null],
  "full_fov_box": [null, null, null, null],
  "fov_reason": "<ok | abstain reason>",
  "final_active_session_pip_box": null,
  "final_session_pip_reason": "<reason code>",
  "final_pip_epoch": null,
  "n_pip_epoch_rotations": null,
  "n_pip_lock_events": null,
  "n_pip_unlock_events": null,
  "monitor_hz": null,
  "effective_monitor_hz": null,
  "n_calibration_frames_requested": 24,
  "n_calibration_frames_admissible": null,
  "min_valid_frame_frac": 0.5,
  "n_frames_in_source": null,
  "n_frames_monitored": null,
  "status": "<ok | fov_abstained: ...>"
}
```

The reason codes you can see:

```
ok                             a box is live
pip_session_box_unavailable    never locked, or unlocked and not re-locked yet
pip_ttl_expired                locked earlier, no fresh evidence since
insufficient_temporal_evidence too few routed frames or time bins
insufficient_split_evidence    too few bins to run the stability check
session_box_failed_stability   the split halves disagreed
no_reliable_popup_footprint    a popup is visible but has no usable box
model_has_no_class3_channel    a 3-class model: popup-blind, not popup-free
```

`calibration_summary.json` records how many of the 24 calibration frames were usable and
why the FOV geometry abstained if it did. `monitor_records.jsonl` is the authoritative PiP
output.

</details>

<details>
<summary><b>The four classes, and what the network actually outputs</b></summary>

The network produces a per-pixel class map and nothing else — every bounding box in this
repository is geometry computed from those masks afterwards.

```
mask[y, x] ∈ {0, 1, 2, 3}
```

| value | class | in the figure | what it is |
|---:|---|---|---|
| 0 | `background_ui` | UI / background | processor UI, letterbox, anything outside the optics |
| 1 | `valid_fov_tissue` | Tissue | usable endoscopic tissue |
| 2 | `black_corner` | Corner | dark peripheral region inside the frame |
| 3 | `popup_overlay` | PiP | an overlaid sub-window |

![Label examples](assets/sup_fig3_label_pairs.png)

</details>

<details>
<summary><b>What adapts to your input, what you choose, what is fixed</b></summary>

Three things follow your input without being asked. Box coordinates come back in your
frame's own resolution, so nothing needs rescaling. The monitor rate is
`min(source_fps, --monitor-hz)`, because a source cannot be sampled faster than it exists —
a 3 fps clip is monitored at 3 Hz no matter what you pass. And popup detection is active
because this checkpoint has a class-3 channel; a 3-class model would be reported as
popup-blind rather than silently popup-free.

Four things are yours to set:

| flag | default | effect |
|---|---|---|
| `--device` | `cpu` | `cuda` is roughly 5–10× faster |
| `--monitor-hz` | `5.0` | how often the PiP monitor looks; lower is faster and coarser |
| `--weights` | `weights/colfov_b8s3.pt` | |
| `--out` | `outputs/...` | output directory |

Everything else is frozen and deliberately not exposed on the command line: 24 calibration
frames, the 0.50 admissible fraction, the 1.25 inner-box aspect, the 0.4 agreement
threshold, the stability gates at IoU 0.95 and area ratio 0.97, the 0.50 drift threshold,
and the minimum routed frames and time bins. Those are the values the model was evaluated
under, and changing them would produce boxes that no longer correspond to any reported
behaviour.

</details>

<details>
<summary><b>Speed</b></summary>

Single frame, no batching, on an RTX 5060 Ti and an Intel CPU:

| | 1920×1080 | 1350×1080 | 1280×720 |
|---|---:|---:|---:|
| mask only, GPU | 4.0 ms | 3.5 ms | 3.3 ms |
| mask + popup + routing, GPU | 48 ms | 36 ms | 21 ms |
| mask only, CPU | 220 ms | 165 ms | 109 ms |
| mask + popup + routing, CPU | 267 ms | 199 ms | 131 ms |

The network is small at 487,316 parameters, so most of the per-frame cost after the mask is
the popup post-processing and routing gate, which run on CPU through OpenCV.

Over a whole recording on GPU at 1280×720 with 5 Hz monitoring and decoding included, that
works out to roughly 8.5 seconds of compute per minute of video, about seven times faster
than realtime. Only the monitored frames are segmented, so a 59 fps recording at 5 Hz means
about one frame in twelve rather than all of them.

</details>

<details>
<summary><b>How a recording is processed</b></summary>

Calibration comes first: 24 evenly spaced frames are taken from the recording and screened
by a guard that runs no model at all, looking only at dynamic range, crushed and saturated
pixel fractions, and gradient energy. If fewer than half survive, both FOV boxes abstain.
The rule is a fraction rather than a count, which at 24 frames happens to work out to 12.

The recording is then replayed in time order at a fixed cadence. The scheduler carries a
fractional phase rather than rounding to an integer stride, so the long-run rate is exactly
`min(source_fps, monitor_hz)` instead of drifting over an hour of video.

The PiP box locks only once enough temporally spread evidence passes a stability check, and
a confirmed relocation unlocks it and rotates the epoch rather than immediately re-locking
from the evidence that justified the old position. Throughout, what is reported for an
instant uses only evidence that had arrived by then. This is causal offline replay; it is
not validated for capture-card timing, asynchronous decoding, or live-stream deadlines.

</details>

<details>
<summary><b>Model and data</b></summary>

TinyUNet with 4 classes and 8 base channels, 487,316 parameters, input 512 × 384 letterbox
with `/255` normalisation and nothing else. The loader reads the architecture from the
checkpoint's own config, requires 4 classes and 8 base channels, loads strictly, and checks
the file against `weights/SHA256SUMS`. `TinyUNet` has no architecture defaults, so a
checkpoint cannot quietly load into a plausible-looking wrong model.

No data is included. The datasets used in the study are distributed by their maintainers
under their own terms; obtain them from the official sources and comply with those terms.

| dataset | dataset paper — follow its data-availability statement |
|---|---|
| REAL-Colon | https://doi.org/10.1038/s41597-024-03359-0 |
| C3VDv2 | https://doi.org/10.1038/s41597-026-07471-1 |
| CAS-Colon | https://doi.org/10.1038/s41597-025-05588-3 |

Part of the training material is a hospital dataset that is not publicly redistributable
and is not linked here.

</details>

<details>
<summary><b>Tests</b></summary>

```bash
pytest -q
```

Covers the checkpoint gate, the four-class output, the 24-frame geometry contract, the
cadence contract, the causal state-transition contract, fail-closed decoding, and that the
CLI examples stay aligned with the exported dataclasses. Synthetic fixtures only — no
clinical media is needed or shipped. This is software correctness, not evidence of clinical
performance.

</details>

## Limitations

Research use only: this is not a medical device and is not for diagnosis or clinical
decisions. The evaluation behind the work is a development evaluation over a limited number
of recordings rather than an independent confirmatory validation, and the reported results
live in the paper rather than here. Behaviour on equipment, resolutions or overlay styles
outside the development material is simply unknown. Abstention is by design, so a `null`
box is the intended fail-closed outcome and not something to route around.

## Citation

The accompanying manuscript is not yet submitted. Citation details will be added here and
in `CITATION.cff` on publication.
