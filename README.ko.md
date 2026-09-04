# ColFOV

[English](README.md)

**대장내시경 AI 모델에 넣는 이미지에는 내시경 시야가 아닌 것이 많이 섞여 있습니다.**
프로세서 UI 테두리가 감싸고, 광학계가 닿지 않는 모서리는 까맣고, letterbox 여백이 붙고,
두 번째 화면을 보여주는 picture-in-picture 창이 뜨기도 합니다. depth든 detection이든
classification이든, 이 픽셀들이 전부 같이 입력으로 들어갑니다. 그렇다고 특정 장비에 맞춰
crop 좌표를 박아두면 다음 장비에서 안 맞습니다.

![ColFOV workflow](assets/fig1_workflow.png)

ColFOV는 프레임을 네 클래스로 나눈 다음 잘라 쓸 사각형 셋을 뽑아냅니다. 쓸 수 있는 시야
전체를 감싸는 **Full-FOV bbox**, 그 안에서 조직만 골라 1.25 비율로 채운 **Inner-FOV
bbox**, 그리고 하위 창을 잡는 **PiP bbox**입니다. 앞의 둘은 영상당 한 번만 구하면 끝이지만,
PiP 창은 시술 도중에 옮겨 다니므로 계속 추적합니다.

셋 중 무엇을 쓸지는 하려는 일에 달렸습니다. Inner-FOV는 UI도 모서리도 없는 깨끗한
사각형이라 depth나 classification 모델에 넣기 좋고, Full-FOV는 주변부까지 남기니까 화면
가장자리에 주석이 걸릴 수 있는 경우에 맞습니다. 각 선택이 downstream에 어떤 차이를 만드는지는
논문에 있습니다. 여기서 중요한 건 그게 선택이고, 그 선택을 겉으로 드러냈다는 점입니다.

라이선스는 아직 정하지 않았고, 원고 투고할 때 함께 정리할 예정입니다.

## 설치

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

## 써보기

이미지나 영상은 저장소에 없습니다. 쓰려는 데이터를 먼저 내려받은 다음, 그 파일 경로를
넘기세요.

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session
```

사각형 셋이 담긴 `session_result.json`, `calibration_summary.json`, 그리고 관측 시점마다 한
줄씩 쌓이는 `monitor_records.jsonl`이 나옵니다. 좌표는 정수 `(x1, y1, x2, y2)`, `x2`와
`y2`는 exclusive, 기준은 원본 해상도라서 바로 슬라이싱하면 됩니다.

```python
import cv2, json

result = json.load(open("outputs/session/session_result.json"))
frame  = cv2.imread("/path/to/frame.png")

box = result["inner_fov_box"]
crop = frame[box[1]:box[3], box[0]:box[2]] if box else frame
```

프레임 한 장만 넣어도 되는데, 그때는 마스크와 프레임 단위 PiP 후보까지만 나옵니다.

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image
```

FOV 박스가 안 나오는 건 설계가 그렇기 때문입니다. 한 장만 봐서는 잠깐 가려진 것 — 점막이
렌즈를 덮었거나, 세척 중이거나, 순간적으로 어두워진 것 — 과 진짜 시야 경계를 구분할 수
없습니다. 그래서 여러 장이 공통으로 인정하는 영역에서 박스를 뽑습니다. PiP 박스에 타임라인이
필요한 이유도 같습니다. 창이 한자리에 충분히 오래 머물러야 "진짜 떠 있다"고 인정합니다.

PyTorch가 CUDA 빌드면 두 명령 다 `--device cuda`를 붙일 수 있습니다. 그냥
`pip install torch`로 깔면 CPU 빌드인 경우가 많은데, 그때는 assertion으로 죽지 않고 왜 안
되는지 알려줍니다.

## 박스가 `null`로 나오면

셋 다 `null`이 될 수 있습니다. 고장이 아니라 "근거가 없어서 답하지 않겠다"는 뜻입니다.
쓸 만한 calibration 프레임이 모자랐거나, PiP 창이 lock할 만큼 안정적으로 잡히지 않은
경우인데, 어느 조건이 걸렸는지는 옆에 붙는 사유 코드로 알 수 있습니다.

이때 적당한 사각형을 대신 넣는 것만은 피하세요. 잘못된 crop은 에러 한 번 안 내고 그 뒤
단계를 전부 망칩니다.

## PiP 박스는 시점별로 읽으세요

FOV 박스 둘은 영상 내내 그대로라 `session_result.json`에서 한 번 읽으면 됩니다.

PiP는 다릅니다. 창이 중간에 옮겨지면 영상 전체에 맞는 사각형이라는 게 없습니다. 요약에 있는
`final_active_session_pip_box`는 말 그대로 마지막 상태라, 이걸 앞부분에 적용하면 이동 전
구간이 통째로 틀립니다.

```python
live = {}
for line in open("outputs/session/monitor_records.jsonl"):
    r = json.loads(line)
    live[r["frame_index"]] = r["active_session_pip_box"]   # 박스 또는 None
```

각 줄에 `timestamp_sec`, `pip_state`, `pip_epoch`, `pip_reason`이 함께 들어 있고,
`pip_event`가 `lock` · `unlock` · `relock` · `null` 중 하나로 찍힙니다. 창이 언제 나타나고
움직이고 사라졌는지 그대로 따라갈 수 있습니다.

<details>
<summary><b>출력 스키마</b></summary>

`session_result.json`입니다. 값은 자리표시자고 키와 형태만 보시면 됩니다.

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

사유 코드는 이렇습니다.

```
ok                             박스가 살아 있음
pip_session_box_unavailable    lock한 적 없거나, unlock 후 아직 재잠금 전
pip_ttl_expired                전에 lock했지만 그 뒤로 새 근거가 없음
insufficient_temporal_evidence routed 프레임이나 time bin이 모자람
insufficient_split_evidence    stability 검사를 돌릴 bin이 모자람
session_box_failed_stability   split 양쪽이 어긋남
no_reliable_popup_footprint    popup은 보이는데 쓸 만한 박스가 안 나옴
model_has_no_class3_channel    3-class 모델이라 popup을 못 봄 (없는 게 아님)
```

`calibration_summary.json`에는 24장 중 몇 장이 쓸 만했는지, abstain했다면 왜 그랬는지가
들어갑니다. PiP 출력의 정본은 `monitor_records.jsonl`입니다.

</details>

<details>
<summary><b>네 클래스와 세 박스</b></summary>

신경망이 내놓는 건 픽셀별 클래스 맵 하나뿐입니다. 이 저장소의 bounding box는 전부 그
마스크에서 나중에 계산한 값입니다.

```
mask[y, x] ∈ {0, 1, 2, 3}
```

| 값 | 클래스 | figure 표기 | 무엇인가 |
|---:|---|---|---|
| 0 | `background_ui` | UI / background | 프로세서 UI, letterbox, 광학 시야 바깥 |
| 1 | `valid_fov_tissue` | tissue | 쓸 수 있는 내시경 조직 |
| 2 | `black_corner` | black corner | 프레임 안쪽의 검은 주변부 |
| 3 | `popup_overlay` | PiP | 덧씌워진 하위 창 |

![Three b-boxes from semantic segmentation](assets/sup_fig3_label_pairs.png)

공개 데이터와 병원 데이터 모두 해상도와 화면 구성이 제각각이지만, 같은 네 클래스에서 같은
세 박스가 나옵니다.

</details>

<details>
<summary><b>알아서 맞춰지는 것 / 정할 수 있는 것 / 못 바꾸는 것</b></summary>

세 가지는 따로 안 건드려도 입력에 맞춰집니다. 박스 좌표는 원본 해상도 그대로 나오니 다시
스케일할 일이 없고, 모니터링 주기는 `min(source_fps, --monitor-hz)`라 3 fps 영상은 뭘 넣든
3 Hz로 봅니다. 소스보다 빠르게 볼 수는 없으니까요. popup 검출은 이 checkpoint에 class-3
채널이 있어서 켜집니다. 3-class 모델이면 "popup 없음"이 아니라 "popup을 못 봄"으로
보고합니다.

정할 수 있는 건 네 개입니다.

| 옵션 | 기본값 | 효과 |
|---|---|---|
| `--device` | `cpu` | `cuda`가 5–10배쯤 빠름 |
| `--monitor-hz` | `5.0` | PiP를 얼마나 자주 볼지. 낮추면 빠르고 거칠어짐 |
| `--weights` | `weights/colfov_b8s3.pt` | |
| `--out` | `outputs/...` | 출력 폴더 |

나머지는 전부 고정이고 일부러 명령행에 안 내놨습니다. calibration 24장, admissible 비율
0.50, inner box 비율 1.25, agreement threshold 0.4, stability gate(IoU 0.95 / 면적비 0.97),
drift threshold 0.50, 최소 routed 프레임과 time bin 수가 그렇습니다. 모델을 평가할 때 쓴
값들이라, 바꾸면 논문에 적힌 어떤 동작과도 대응하지 않는 박스가 나옵니다.

</details>

<details>
<summary><b>속도</b></summary>

프레임 한 장, 배칭 없이, RTX 5060 Ti와 Intel CPU에서 잰 값입니다.

| | 1920×1080 | 1350×1080 | 1280×720 |
|---|---:|---:|---:|
| mask만, GPU | 4.0 ms | 3.5 ms | 3.3 ms |
| mask + popup + routing, GPU | 48 ms | 36 ms | 21 ms |
| mask만, CPU | 220 ms | 165 ms | 109 ms |
| mask + popup + routing, CPU | 267 ms | 199 ms | 131 ms |

신경망은 487,316 파라미터로 작습니다. mask 이후가 오래 걸리는 이유는 popup 후처리와 routing
gate가 OpenCV로 CPU에서 돌기 때문입니다.

영상 한 편 기준으로는 GPU·1280×720·5 Hz에 디코딩까지 합쳐 **1분짜리 영상에 약 8.5초**,
실시간의 7배쯤 됩니다. 모니터링한 프레임만 분할하니까 59 fps 영상을 5 Hz로 보면 12장에 한
장꼴입니다.

`--monitor-hz`가 속도와 시간 해상도를 맞바꾸는 손잡이입니다. 데이터에 따라 조절하면 됩니다.
낮추면 빨리 끝나지만 PiP 창을 늦게 잡거나 짧게 뜬 건 놓칠 수 있고, 올리면 짧은 변화까지
잡는 대신 연산이 그만큼 늘어납니다. FOV 박스는 calibration 단계에서 나오므로 이 값과
무관합니다.

</details>

<details>
<summary><b>영상 한 편이 처리되는 순서</b></summary>

먼저 calibration입니다. 영상에서 균등 간격으로 24장을 뽑고, 모델을 안 쓰는 guard가 동적
범위와 crushed/saturated 비율, gradient energy만 보고 걸러냅니다. 절반이 안 남으면 FOV 박스
둘 다 abstain합니다. 개수가 아니라 비율 기준인데, 24장이면 결과적으로 12장입니다.

그다음 영상을 시간 순서대로 일정 간격으로 훑습니다. 스케줄러가 정수 stride로 반올림하지 않고
분수 phase를 이월해서, 평균 주기가 정확히 `min(source_fps, monitor_hz)`로 유지됩니다. 한
시간짜리 영상에서도 어긋나지 않습니다.

PiP 박스는 시간적으로 충분히 퍼진 근거가 stability 검사를 통과해야 lock됩니다. 위치 이동이
확정되면 unlock하고 epoch을 돌리는데, 이때 예전 위치를 뒷받침하던 근거로 곧바로 다시
잠그지는 않습니다. 그리고 어떤 시점의 값이든 그 시점까지 들어온 근거만 씁니다. causal
오프라인 replay라서, capture-card 타이밍이나 비동기 디코딩, 라이브 스트림 데드라인은 검증
대상이 아닙니다.

</details>

<details>
<summary><b>모델과 데이터</b></summary>

TinyUNet, 4 classes, base channels 8, 487,316 파라미터. 입력은 512 × 384 letterbox고
정규화는 `/255`만 합니다. 로더가 checkpoint 안의 config에서 구조를 읽어 4-class/base-8을
확인하고, strict로 올리고, `weights/SHA256SUMS`와 대조합니다. `TinyUNet`에 아키텍처 기본값을
두지 않아서, 엉뚱한 모델에 checkpoint가 조용히 얹히는 일이 생기지 않습니다.

데이터는 안 들어 있습니다. 연구에 쓴 데이터셋은 각 배포처의 조건이 따로 있으니 공식 출처에서
직접 받으시고 그 조건을 따르세요.

| 데이터셋 | 데이터셋 논문 — data-availability 항목 참고 |
|---|---|
| REAL-Colon | https://doi.org/10.1038/s41597-024-03359-0 |
| C3VDv2 | https://doi.org/10.1038/s41597-026-07471-1 |
| CAS-Colon | https://doi.org/10.1038/s41597-025-05588-3 |

학습에 쓴 자료 중 일부는 공개 재배포가 안 되는 병원 데이터라 여기 링크하지 않았습니다.

</details>

<details>
<summary><b>테스트</b></summary>

```bash
pytest -q
```

checkpoint gate, 4-class 출력, 24장 geometry 계약, cadence 계약, causal state-transition
계약, fail-closed 디코딩, 그리고 CLI 예제가 export된 dataclass와 어긋나지 않는지를 봅니다.
전부 합성 fixture라 임상 자료는 필요 없고 동봉하지도 않았습니다. 소프트웨어가 맞게 도는지를
보는 것이지 임상 성능의 근거는 아닙니다.

</details>

## 한계

연구용입니다. 의료기기가 아니고 진단이나 임상 판단에 쓸 수 없습니다.

근거가 된 평가는 제한된 수의 영상을 대상으로 한 development evaluation이지 독립적인 확증
검증이 아닙니다. 수치는 여기 말고 논문에 있습니다. 개발에 쓰지 않은 장비나 해상도, 오버레이
양식에서 어떻게 동작할지는 알 수 없습니다. abstain은 설계대로 도는 것이므로 `null` 박스를
우회하려 들지 마세요.

## 인용

원고는 아직 투고 전입니다. 게재되면 여기와 `CITATION.cff`에 인용 정보를 넣겠습니다.
