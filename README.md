# ColFOV

Semantic-guided four-class segmentation of endoscopic frames, and the three geometry
products derived from it: an inner-FOV box, a full-FOV box, and a conditional
session-level picture-in-picture (PiP) box.

Code and model weights are publicly available for scientific transparency. This is
**not** open-source software — see [License](#license).

![ColFOV workflow](assets/fig1_workflow.png)

## What the model outputs, and what it does not

The network outputs **only a segmentation mask**:

```
logits  [B, 4, H, W]
mask    [H, W], values in {0, 1, 2, 3}
```

| value | class | meaning |
|---:|---|---|
| 0 | `background_ui` | on-screen interface, letterbox, anything outside the optical field |
| 1 | `valid_fov_tissue` | usable endoscopic tissue |
| 2 | `black_corner` | dark peripheral region inside the frame |
| 3 | `popup_overlay` | an overlaid sub-window |

![Label examples](assets/sup_fig3_label_pairs.png)

Both figures are flattened reproductions of figures from the accompanying publication.
They are illustrations, not data: see `assets/RIGHTS.md` and `assets/PROVENANCE.md`.

**All three bounding boxes are derived geometry, not network outputs.** They are
computed from masks by ordinary geometry, and each can be unavailable:

| field | derived from | `null` when |
|---|---|---|
| `inner_fov_box` | 24 calibration frames, stable class-1 tissue | fewer than 50% of the sampled frames are admissible |
| `full_fov_box` | the same 24 frames, stable class-1-or-2 support | same condition |
| `active_session_pip_box` | causal monitoring of the whole recording | no lock, evidence insufficient, TTL expired, or stability gate not passed |

All three keys always exist. `null` is a **fail-closed abstention** carrying a reason
code — never an error, and never replaced by a fallback box.

### `active_session_pip_box` is time-dependent

It is not a property of a recording. It is the box that was live **at one instant**,
given only the evidence that had arrived by then. A recording whose PiP window moves
has more than one correct answer over its length.

`session_result.json` reports `final_active_session_pip_box`, which is the **last state
only**. Do not apply it retrospectively to earlier timestamps — use the per-observation
records in `monitor_records.jsonl`, where each row carries `frame_index`,
`timestamp_sec`, `active_session_pip_box`, `pip_state`, `pip_epoch`, `pip_event`
(`lock` / `unlock` / `relock` / `null`) and `pip_reason`.

## Model

| | |
|---|---|
| architecture | TinyUNet, 4 classes, 8 base channels |
| parameters | 487,316 |
| input | 512 × 384, letterbox, `/255` normalisation only |
| checkpoint | `weights/colfov_b8s3.pt`, 1.908 MiB |
| sha256 | `df4054b8d2a22413ae232b7ea2ce01cbe70c838031167b5afb72c71ce9670713` |

The loader reads the architecture from the checkpoint's own embedded config, requires
4 classes and 8 base channels, loads the state dict strictly, and verifies the hash.
`TinyUNet` has **no architecture defaults**, so a checkpoint cannot be loaded into a
plausible-looking wrong model.

## Install

```bash
git clone <repository-url>
cd colfov
python -m venv .venv
```

Activate the environment — Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -e ".[test]"
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('weights/colfov_b8s3.pt').read_bytes()).hexdigest())"
```

## Data

**This repository contains no image or video data.** Bring your own frames, supplied
through `--image` and `--video`. The datasets used in the study are distributed by
their own maintainers under their own terms; obtain them from the official sources and
comply with those terms:

| dataset | official source |
|---|---|
| REAL-Colon | `<official dataset page>` |
| C3VDv2 | `<official dataset page>` |
| CAS-Colon | `<official dataset page>` |

Part of the training material is a hospital dataset that is not publicly
redistributable and is not linked here.

## Run

Single frame — four-class mask plus frame-local diagnostics:

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image_example
```

Writes `mask.png`, `overlay.png` and `frame_result.json`. A single frame yields **no**
FOV boxes and **no** session PiP box: the FOV boxes need a calibration sample and the
PiP box is causal over a timeline.

Whole recording — all three geometry products:

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session_example
```

Writes `session_result.json`, `calibration_summary.json` and `monitor_records.jsonl`.

### Output schema

`session_result.json` (placeholder values; shapes and keys, not results):

```json
{
  "segmentation_classes": 4,
  "class_names": ["background_ui", "valid_fov_tissue", "black_corner", "popup_overlay"],
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

## How the session pass works

1. **Calibration.** Exactly 24 evenly-spaced frames are taken from the recording, and a
   model-free admissibility guard runs on that sample. If the admissible **fraction**
   falls below 0.50 the FOV geometry abstains. The rule is a fraction, not a count of
   12; at 24 frames it happens to mean 12.
2. **Monitoring.** The recording is replayed in time order at a fixed cadence, 5 Hz by
   default. The scheduler carries a fractional phase, so the long-run rate is exactly
   `min(source_fps, monitor_hz)` — a rounded integer stride would drift.
3. **Qualification.** A session box locks only once the evidence and stability gates
   pass. A confirmed relocation unlocks, rotates the epoch, and does not immediately
   relock from the evidence that justified the old position.

The session example performs **causal, fixed-cadence offline replay**. It does not
claim validation of capture-card timing, asynchronous decoding, or live-stream deadline
behaviour. It is not a real-time system and has not been validated as one.

## Tests

```bash
pytest -q
```

55 tests covering the checkpoint gate, four-class output, the 24-frame geometry
contract, the cadence contract, the causal state-transition contract, and fail-closed
decoding. They use **synthetic fixtures only** — no clinical media is required or
distributed. This is software correctness, not evidence of clinical performance.

## Intended use and limitations

- **Research use only.** Not a medical device, not for diagnosis or clinical decisions.
- Reported evaluation results are in the paper (see *Citation*); they are not
  reproduced here as headline numbers.
- The evaluation behind the paper is a development evaluation on a limited number of
  recordings; it is not an independent confirmatory validation.
- Behaviour on equipment, resolutions or overlay styles outside the development
  material is unknown.
- The three geometry products abstain by design. An abstention is the intended
  fail-closed outcome, not a failure to be worked around with a fallback box.

## License

All repository content is **All Rights Reserved** — source code, model weights,
configuration files, documentation, test fixtures and figures alike. Public
availability does not grant permission to use, execute, modify, redistribute or create
derivative works. Written permission from the copyright holder is required; see
`LICENSE`.

Figure assets carry the same terms, and the clinical frames composited into them are
not licensed or separately distributed (`assets/RIGHTS.md`).

## Citation

The accompanying manuscript is not yet submitted. Citation details will be added here
and in `CITATION.cff` on publication.
