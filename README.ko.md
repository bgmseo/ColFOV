# ColFOV

[English](README.md)

내시경 프레임은 깨끗한 이미지가 아닙니다. 광학 시야 외에 검은 모서리, 프로세서 UI
테두리, letterbox 패딩, 그리고 때때로 picture-in-picture 창이 함께 들어 있습니다. 이
프레임을 그대로 downstream 모델(depth·detection·classification)에 넣으면 상당수 픽셀이
조직이 아닙니다.

ColFOV는 조직인 부분을 찾아, 잘라 쓸 수 있는 사각형을 돌려줍니다.

코드와 model weights는 과학적 투명성을 위해 공개합니다. 라이선스 조건은 아직 정하지
않았으며, 원고 투고 시점에 맞춰 추가할 예정입니다.

![ColFOV workflow](assets/fig1_workflow.png)

## 언제 쓰면 좋은가

- 대장내시경 프레임에 depth·detection·classification을 돌리기 전에 UI와 검은 테두리를
  먼저 걷어내고 싶을 때
- 여러 프로세서나 여러 기관의 영상을 다뤄서, 하드코딩한 crop 사각형이 그대로 통하지 않을 때
- 시술 중간에 picture-in-picture 창이 뜨는데, 고정된 모서리라고 가정하지 않고 **언제**
  떠 있고 **어디에** 있는지 알아야 할 때
- 추측하지 않는 전처리가 필요할 때 — 근거가 없으면 그럴듯한 틀린 박스 대신 `null`과
  사유 코드를 돌려줍니다

실시간 capture-card 파이프라인에는 맞지 않고(오프라인 replay입니다), 임상 용도도
아닙니다.

## 무엇이 나오는가

신경망이 내는 것은 하나뿐입니다 — 픽셀별 클래스 맵.

```
mask[y, x] ∈ {0, 1, 2, 3}
```

| 값 | 클래스 | 의미 |
|---:|---|---|
| 0 | `background_ui` | 프로세서 UI, letterbox, 광학 시야 바깥 전부 |
| 1 | `valid_fov_tissue` | 사용 가능한 내시경 조직 |
| 2 | `black_corner` | 프레임 안쪽의 어두운 주변부 |
| 3 | `popup_overlay` | 덧씌워진 하위 창 |

![Label examples](assets/sup_fig3_label_pairs.png)

나머지는 전부 **그 마스크로부터 계산한 geometry**입니다 — 여러분 프레임의 좌표계로 된
사각형 3개:

| box | 무엇을 감싸는가 | 주 용도 |
|---|---|---|
| `inner_fov_box` | 안정적인 조직 **안쪽**에 들어가는 1.25 비율 사각형 | 모델 입력 crop: UI·검은 모서리·패딩 없음 |
| `full_fov_box` | 안정적인 시야 전체의 외접 사각형 | 디스플레이용, 또는 주변부를 잃으면 안 되는 crop |
| `active_session_pip_box` | 그 시점에 살아 있던 PiP 창 | 하위 뷰를 따로 라우팅하거나, 주 crop에서 제외 |

## Box를 실제로 쓰는 법

박스는 `(x1, y1, x2, y2)` 정수이고 `x2`/`y2`는 **exclusive**, 원본 프레임 해상도
기준입니다. 그래서 바로 슬라이싱됩니다:

```python
import cv2, json

result = json.load(open("outputs/session_example/session_result.json"))
frame = cv2.imread("/path/to/frame.png")

box = result["inner_fov_box"]
if box is not None:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]        # 이걸 downstream 모델에 넣으면 됩니다
else:
    crop = frame                      # ColFOV가 abstain함. fallback은 사용자 판단
```

FOV 박스 두 개는 **세션 단위**입니다 — 영상에서 한 번 계산해 그 영상 전체 프레임에
적용합니다. PiP 박스는 다릅니다. 시간에 따라 바뀌므로 프레임별로 읽어야 합니다:

```python
import json

live = {}
for line in open("outputs/session_example/monitor_records.jsonl"):
    r = json.loads(line)
    live[r["frame_index"]] = r["active_session_pip_box"]   # 그 시점의 박스 또는 None
```

`null`은 실패가 아니라 의도된 답입니다. 그 사각형에 대한 근거가 없었다는 뜻이고, 사유
코드가 어떤 조건이 충족되지 않았는지 알려줍니다. **fallback 박스로 대체하지 마세요** —
잘못된 crop은 그 뒤의 모든 것을 조용히 망가뜨립니다.

### PiP 박스를 프레임별로 읽어야 하는 이유

시술 중 PiP 창이 이동하면, 영상 전체에 맞는 사각형은 존재하지 않습니다.
`session_result.json`의 `final_active_session_pip_box`는 **마지막 상태**일 뿐이라, 이걸
이전 시각에 적용하면 창이 옮겨지기 전 구간에서 틀린 박스가 됩니다.
`monitor_records.jsonl`을 쓰세요 — 각 행에 `frame_index`, `timestamp_sec`,
`active_session_pip_box`, `pip_state`, `pip_epoch`, `pip_event`
(`lock` / `unlock` / `relock` / `null`), `pip_reason`이 들어 있습니다.

## 자동으로 맞춰지는 것 / 사용자가 정하는 것 / 고정된 것

**입력에 따라 자동으로 맞춰집니다**

- 박스 좌표는 여러분 프레임의 해상도 그대로 나옵니다 — 별도 rescale 불필요
- 모니터링 주기는 `min(source_fps, --monitor-hz)`입니다. 25 fps 영상에 `--monitor-hz 5`면
  5 Hz, 3 fps 영상이면 3 Hz — 소스보다 빠르게 샘플링할 수는 없기 때문입니다
- 이 checkpoint는 class-3 채널이 있어 popup 검출이 활성화됩니다. 3-class 모델이라면
  "popup이 없다"가 아니라 "popup을 볼 수 없다"로 보고됩니다

**사용자가 정합니다**

| 옵션 | 기본값 | 효과 |
|---|---|---|
| `--device` | `cpu` | `cuda`가 대략 5–10배 빠름, [속도](#속도) 참조 |
| `--monitor-hz` | `5.0` | PiP 모니터가 보는 빈도. 낮출수록 빠르고 거칠어짐 |
| `--weights` | `weights/colfov_b8s3.pt` | |
| `--out` | `outputs/...` | 출력 디렉터리 |

**고정이며 일부러 노출하지 않았습니다**

calibration 24 프레임, admissible 비율 0.50, inner box 비율 1.25, agreement threshold
0.4, stability gate(IoU 0.95 / 면적비 0.97), drift threshold(IoU 0.50), 최소 routed
프레임 수와 temporal bin 수. 모델이 평가된 동결 값들입니다. 바꾸면 보고된 어떤 동작과도
대응하지 않는 박스가 나오므로 명령행 옵션으로 두지 않았습니다.

## 속도

단일 프레임, 배칭 없음. RTX 5060 Ti + Intel CPU 실측:

| | 1920×1080 | 1350×1080 | 1280×720 |
|---|---:|---:|---:|
| mask만, GPU | 4.0 ms | 3.5 ms | 3.3 ms |
| mask + popup + routing, GPU | 48 ms | 36 ms | 21 ms |
| mask만, CPU | 220 ms | 165 ms | 109 ms |
| mask + popup + routing, CPU | 267 ms | 199 ms | 131 ms |

신경망 자체는 작습니다(487,316 파라미터). mask 이후 비용의 대부분은 OpenCV로 CPU에서
도는 popup 후처리와 routing gate입니다.

영상 전체 기준, GPU·1280×720·5 Hz 모니터링·디코딩 포함: **영상 1분당 약 8.5초 연산**,
실시간 대비 약 7배 빠릅니다. 모니터링된 프레임만 분할하므로, 59 fps 영상을 5 Hz로 보면
대략 12프레임당 1장이지 전 프레임이 아닙니다.

## 설치

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

```bash
pip install -e ".[test]"
```

## 실행

저장소에는 **이미지·영상 데이터가 들어 있지 않습니다.** 본인 파일 경로를 넘기세요.

프레임 1장 — 마스크와 프레임 단위 진단값:

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image_example
```

`mask.png`, `overlay.png`, `frame_result.json`이 나옵니다. 프레임 1장으로는 FOV 박스도
PiP 박스도 나오지 않습니다 — FOV 박스는 calibration 표본이, PiP 박스는 타임라인이
필요합니다.

영상 전체 — 사각형 3개 전부:

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session_example --device cuda
```

`session_result.json`, `calibration_summary.json`, `monitor_records.jsonl`이 나옵니다.

### 파일별 내용

`session_result.json` — 세션 단위 답. 키와 형태만, 값은 placeholder:

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

`monitor_records.jsonl` — 모니터링된 관측 1건당 1행. PiP 출력의 정본입니다.

`calibration_summary.json` — 24개 calibration 프레임 중 몇 개가 admissible이었는지,
FOV geometry가 abstain했다면 그 사유.

## 세션 처리 방식

1. **Calibration.** 영상에서 균등 간격으로 24 프레임을 뽑아, 모델 없이 도는 guard
   (동적 범위, crushed/saturated 비율, gradient energy)로 걸러냅니다. 50% 미만이
   통과하면 FOV 박스 둘 다 abstain합니다. 개수가 아니라 **비율** 규칙이며, 24개
   기준에서 결과적으로 12개입니다.
2. **Monitoring.** 영상을 시간 순서대로 고정 cadence로 재생합니다. 스케줄러가 분수
   phase를 이월하므로 장기 평균 주기가 정확히 `min(source_fps, monitor_hz)`입니다 —
   정수 stride로 반올림하면 1시간에 걸쳐 드리프트가 쌓입니다.
3. **Qualification.** 시간적으로 충분히 퍼진 routed 근거가 stability 검사를 통과해야만
   PiP 박스가 lock됩니다. 위치 이동이 확정되면 unlock하고 epoch을 회전시키며, 예전
   위치를 정당화하던 근거로 즉시 재잠금하지 않습니다.

이것은 causal 오프라인 replay입니다. 어떤 시각에 대해 보고되는 값은 그 시점까지 도착한
근거만 씁니다. capture-card 타이밍, 비동기 디코딩, 라이브 스트림 데드라인은 검증 대상이
아닙니다.

## 모델

TinyUNet, 4 classes, base channels 8, 487,316 파라미터. 입력 512 × 384, letterbox,
정규화는 `/255`만. 로더는 checkpoint 자체의 config에서 구조를 읽어 4-class/base-8을
강제하고, strict로 로드하며, `weights/SHA256SUMS`와 대조합니다. `TinyUNet`에는 아키텍처
기본값이 없어서, 그럴듯해 보이는 잘못된 모델에 checkpoint가 조용히 로드될 수 없습니다.

## 데이터

포함되어 있지 않습니다. 연구에 사용된 데이터셋은 각 관리 주체가 각자의 조건으로
배포합니다 — 공식 출처에서 직접 받고 그 조건을 따르세요:

| 데이터셋 | 데이터셋 논문 (data-availability 항목을 따라 내려받으세요) |
|---|---|
| REAL-Colon | https://doi.org/10.1038/s41597-024-03359-0 |
| C3VDv2 | https://doi.org/10.1038/s41597-026-07471-1 |
| CAS-Colon | https://doi.org/10.1038/s41597-025-05588-3 |

학습 자료의 일부는 공개 재배포가 불가능한 병원 데이터이며 여기에 링크하지 않습니다.

## 테스트

```bash
pytest -q
```

checkpoint gate, 4-class 출력, 24-frame geometry 계약, cadence 계약, causal
state-transition 계약, fail-closed 디코딩, 그리고 CLI 예제가 export된 dataclass와
어긋나지 않는지를 검사합니다. **합성 fixture만** 사용하며 임상 미디어는 필요하지도,
동봉되지도 않습니다. 소프트웨어 정합성 검증이지 임상 성능의 근거가 아닙니다.

## 사용 목적과 한계

- 연구용입니다. 의료기기가 아니며 진단이나 임상 의사결정에 쓸 수 없습니다.
- 이 작업의 근거가 된 평가는 제한된 수의 영상에 대한 development evaluation이며 독립적인
  확증 검증이 아닙니다. 보고된 수치는 논문에 있습니다.
- 개발 자료 밖의 장비·해상도·오버레이 양식에서의 동작은 알 수 없습니다.
- Abstention은 설계된 동작입니다. `null` 박스는 의도된 fail-closed 결과입니다.

## 인용

관련 원고는 아직 투고 전입니다. 게재 시 여기와 `CITATION.cff`에 인용 정보를 추가합니다.
