# ColFOV

[English](README.md)

대장내시경 프레임에서 실제 내시경 영상인 부분만 잘라내고, picture-in-picture 창이 떠
있으면 그 위치를 찾아줍니다.

![ColFOV workflow](assets/fig1_workflow.png)

내시경 프레임에는 조직이 아닌 것들이 함께 담깁니다. 프로세서 UI 테두리, 광학계가 닿지
않는 검은 모서리, letterbox 패딩, 그리고 두 번째 화면을 띄우는 picture-in-picture 창까지
있습니다. ColFOV는 프레임을 이 네 클래스로 분할한 뒤, 잘라 쓸 수 있는 사각형 3개로
바꿔줍니다. 사용 가능한 시야 전체를 감싸는 Full-FOV bbox, 안정적인 조직 안쪽에 1.25 비율로
들어가는 Inner-FOV bbox, 그리고 하위 창을 위한 PiP bbox입니다. 앞의 둘은 영상마다 한 번
계산해 그대로 재사용하고, PiP bbox는 창이 움직이기 때문에 시간에 따라 추적합니다.

코드와 model weights는 과학적 투명성을 위해 공개합니다. 라이선스 조건은 아직 정하지
않았으며 원고 투고 시점에 맞춰 추가할 예정입니다.

## 설치

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

## 사용

저장소에는 이미지·영상 데이터가 들어 있지 않으니 본인 파일 경로를 넘기면 됩니다.

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session
```

사각형 3개가 담긴 `session_result.json`, `calibration_summary.json`, 그리고 관측 시점마다
한 행씩 쌓이는 `monitor_records.jsonl`이 나옵니다. 박스는 정수 `(x1, y1, x2, y2)`이고
`x2`/`y2`는 exclusive, 원본 프레임 해상도 기준이라 그대로 슬라이싱됩니다.

```python
import cv2, json

result = json.load(open("outputs/session/session_result.json"))
frame  = cv2.imread("/path/to/frame.png")

box = result["inner_fov_box"]
crop = frame[box[1]:box[3], box[0]:box[2]] if box else frame
```

프레임 1장으로도 돌아가지만 마스크와 프레임 단위 PiP 후보까지만 나옵니다. FOV 박스는
calibration 표본이, PiP 박스는 타임라인이 있어야 하기 때문입니다.

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image
```

PyTorch가 CUDA 빌드라면 두 명령 모두에 `--device cuda`를 붙일 수 있습니다. 기본
`pip install torch`는 CUDA 빌드가 아닌 경우가 많은데, 그럴 때 예제가 assertion으로 죽지
않고 안내 메시지를 냅니다.

## 박스가 `null`로 나올 때

셋 중 무엇이든 `null`이 될 수 있고, 이건 실패가 아니라 의도된 답입니다. 그 사각형을
뒷받침할 근거가 없었다는 뜻이며 — 쓸 만한 calibration 프레임이 모자랐거나, PiP 창이 lock할
만큼 안정적인 근거를 보여주지 못했거나 — 어떤 조건이 충족되지 않았는지는 함께 나오는 사유
코드가 알려줍니다. 이때 임의의 fallback 사각형으로 대체하는 것만은 하지 마세요. 잘못된
crop은 오류 하나 내지 않으면서 그 뒤의 모든 것을 망가뜨립니다.

## PiP 박스 읽는 법

FOV 박스 두 개는 영상 전체에 그대로 유효하므로 `session_result.json`에서 한 번 읽고 잊어도
됩니다. PiP 박스는 다릅니다. 시술 중간에 창이 옮겨지면 그 영상 전체에 맞는 사각형은 아예
존재하지 않기 때문에, 요약의 `final_active_session_pip_box`는 마지막 상태일 뿐이고 이걸
이전 시각에 적용하면 이동 전 구간이 전부 틀립니다. 시점별로 읽으세요.

```python
live = {}
for line in open("outputs/session/monitor_records.jsonl"):
    r = json.loads(line)
    live[r["frame_index"]] = r["active_session_pip_box"]   # 박스 또는 None
```

각 행에는 `timestamp_sec`, `pip_state`, `pip_epoch`, `pip_reason`과 함께 `lock` ·
`unlock` · `relock` · `null` 중 하나인 `pip_event`가 들어 있어서, 창이 언제 나타나고
움직이고 사라졌는지 그대로 따라갈 수 있습니다.

<details>
<summary><b>출력 스키마</b></summary>

`session_result.json` — 값은 placeholder이고 키와 형태만 보여줍니다.

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

나올 수 있는 사유 코드:

```
ok                             박스가 살아 있음
pip_session_box_unavailable    lock된 적이 없거나, unlock 후 아직 재잠금 전
pip_ttl_expired                이전에 lock했으나 이후 새 근거가 없음
insufficient_temporal_evidence routed 프레임 또는 time bin 부족
insufficient_split_evidence    stability 검사를 돌릴 bin이 부족
session_box_failed_stability   split 양쪽이 불일치
no_reliable_popup_footprint    popup은 보이나 쓸 만한 박스가 없음
model_has_no_class3_channel    3-class 모델: popup이 없는 게 아니라 못 봄
```

`calibration_summary.json`은 24개 calibration 프레임 중 몇 개가 쓸 만했는지와 FOV geometry가
abstain한 경우 그 사유를 담습니다. `monitor_records.jsonl`이 PiP 출력의 정본입니다.

</details>

<details>
<summary><b>네 클래스, 그리고 신경망이 실제로 내놓는 것</b></summary>

신경망이 만드는 것은 픽셀별 클래스 맵 하나뿐이고, 이 저장소의 모든 bounding box는 그
마스크에서 나중에 계산한 geometry입니다.

```
mask[y, x] ∈ {0, 1, 2, 3}
```

| 값 | 클래스 | figure 표기 | 의미 |
|---:|---|---|---|
| 0 | `background_ui` | UI / background | 프로세서 UI, letterbox, 광학 시야 바깥 전부 |
| 1 | `valid_fov_tissue` | Tissue | 사용 가능한 내시경 조직 |
| 2 | `black_corner` | Corner | 프레임 안쪽의 어두운 주변부 |
| 3 | `popup_overlay` | PiP | 덧씌워진 하위 창 |

![Label examples](assets/sup_fig3_label_pairs.png)

</details>

<details>
<summary><b>자동으로 맞춰지는 것 / 사용자가 정하는 것 / 고정된 것</b></summary>

세 가지는 따로 지시하지 않아도 입력을 따라갑니다. 박스 좌표는 원본 프레임 해상도 그대로
나오므로 rescale이 필요 없고, 모니터링 주기는 `min(source_fps, --monitor-hz)`라서 소스보다
빠르게 샘플링되지 않습니다. 3 fps 영상은 무엇을 넘기든 3 Hz로 봅니다. popup 검출은 이
checkpoint에 class-3 채널이 있어서 켜지며, 3-class 모델이었다면 "popup이 없다"가 아니라
"popup을 볼 수 없다"로 보고됩니다.

네 가지는 사용자가 정합니다.

| 옵션 | 기본값 | 효과 |
|---|---|---|
| `--device` | `cpu` | `cuda`가 대략 5–10배 빠름 |
| `--monitor-hz` | `5.0` | PiP 모니터가 보는 빈도. 낮출수록 빠르고 거칠어짐 |
| `--weights` | `weights/colfov_b8s3.pt` | |
| `--out` | `outputs/...` | 출력 디렉터리 |

나머지는 전부 동결이며 일부러 명령행에 노출하지 않았습니다. calibration 24 프레임, admissible
비율 0.50, inner box 비율 1.25, agreement threshold 0.4, stability gate(IoU 0.95 / 면적비
0.97), drift threshold 0.50, 최소 routed 프레임 수와 time bin 수가 그렇습니다. 모델이 평가된
값들이라, 바꾸면 보고된 어떤 동작과도 대응하지 않는 박스가 나옵니다.

</details>

<details>
<summary><b>속도</b></summary>

단일 프레임, 배칭 없음. RTX 5060 Ti + Intel CPU 실측입니다.

| | 1920×1080 | 1350×1080 | 1280×720 |
|---|---:|---:|---:|
| mask만, GPU | 4.0 ms | 3.5 ms | 3.3 ms |
| mask + popup + routing, GPU | 48 ms | 36 ms | 21 ms |
| mask만, CPU | 220 ms | 165 ms | 109 ms |
| mask + popup + routing, CPU | 267 ms | 199 ms | 131 ms |

신경망은 487,316 파라미터로 작은 편이라, mask 이후 비용의 대부분은 OpenCV로 CPU에서 도는
popup 후처리와 routing gate가 차지합니다.

영상 전체로 보면 GPU·1280×720·5 Hz 모니터링에 디코딩까지 포함해 **영상 1분당 약 8.5초**,
실시간의 약 7배 속도입니다. 모니터링된 프레임만 분할하므로 59 fps 영상을 5 Hz로 보면 전
프레임이 아니라 대략 12장당 1장을 처리합니다.

</details>

<details>
<summary><b>영상 한 편이 처리되는 과정</b></summary>

먼저 calibration입니다. 영상에서 균등 간격으로 24 프레임을 뽑고, 모델을 전혀 쓰지 않는
guard가 동적 범위와 crushed/saturated 픽셀 비율, gradient energy만 보고 걸러냅니다. 절반이
안 남으면 FOV 박스 둘 다 abstain합니다. 개수가 아니라 비율 규칙이고, 24개 기준에서 결과적으로
12개가 됩니다.

그다음 영상을 시간 순서대로 고정 cadence로 재생합니다. 스케줄러가 정수 stride로 반올림하지
않고 분수 phase를 이월하므로, 장기 평균 주기가 정확히 `min(source_fps, monitor_hz)`이고 한
시간짜리 영상에서도 드리프트가 쌓이지 않습니다.

PiP 박스는 시간적으로 충분히 퍼진 근거가 stability 검사를 통과해야 lock되고, 위치 이동이
확정되면 예전 위치를 정당화하던 근거로 즉시 재잠금하는 대신 unlock하고 epoch을 회전시킵니다.
그 과정 내내 어떤 시각에 대해 보고되는 값은 그 시점까지 도착한 근거만 씁니다. causal 오프라인
replay이며, capture-card 타이밍이나 비동기 디코딩, 라이브 스트림 데드라인은 검증 대상이
아닙니다.

</details>

<details>
<summary><b>모델과 데이터</b></summary>

TinyUNet, 4 classes, base channels 8, 487,316 파라미터. 입력은 512 × 384 letterbox이고
정규화는 `/255`만 합니다. 로더는 checkpoint 자체의 config에서 구조를 읽어 4-class/base-8을
강제하고, strict로 로드하며, `weights/SHA256SUMS`와 대조합니다. `TinyUNet`에 아키텍처
기본값이 없어서 그럴듯해 보이는 잘못된 모델에 checkpoint가 조용히 로드될 수 없습니다.

데이터는 포함되어 있지 않습니다. 연구에 쓴 데이터셋은 각 관리 주체가 각자의 조건으로
배포하므로 공식 출처에서 직접 받고 그 조건을 따르세요.

| 데이터셋 | 데이터셋 논문 — data-availability 항목을 따라 내려받으세요 |
|---|---|
| REAL-Colon | https://doi.org/10.1038/s41597-024-03359-0 |
| C3VDv2 | https://doi.org/10.1038/s41597-026-07471-1 |
| CAS-Colon | https://doi.org/10.1038/s41597-025-05588-3 |

학습 자료의 일부는 공개 재배포가 불가능한 병원 데이터이며 여기에 링크하지 않습니다.

</details>

<details>
<summary><b>테스트</b></summary>

```bash
pytest -q
```

checkpoint gate, 4-class 출력, 24-frame geometry 계약, cadence 계약, causal
state-transition 계약, fail-closed 디코딩, 그리고 CLI 예제가 export된 dataclass와
어긋나지 않는지를 검사합니다. 합성 fixture만 쓰므로 임상 미디어는 필요하지도 동봉되지도
않습니다. 소프트웨어 정합성 검증이지 임상 성능의 근거가 아닙니다.

</details>

## 한계

연구용입니다. 의료기기가 아니고 진단이나 임상 의사결정에 쓸 수 없습니다. 이 작업의 근거가 된
평가는 제한된 수의 영상에 대한 development evaluation이지 독립적인 확증 검증이 아니며, 보고된
수치는 여기가 아니라 논문에 있습니다. 개발 자료 밖의 장비·해상도·오버레이 양식에서 어떻게
동작할지는 알 수 없습니다. abstention은 설계된 동작이므로 `null` 박스는 의도된 fail-closed
결과이며 우회할 대상이 아닙니다.

## 인용

관련 원고는 아직 투고 전입니다. 게재 시 여기와 `CITATION.cff`에 인용 정보를 추가합니다.
