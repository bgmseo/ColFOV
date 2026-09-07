# ColFOV

[English](README.md)

ColFOV는 대장내시경 영상에서 분석 목적에 맞는 입력 영역을 선택하는 경량 semantic
segmentation 모델과 후처리를 제공합니다. 조직, 광학 모서리, 녹화 인터페이스와 보조
화면을 구분하고 여러 프레임의 예측으로 **video-specific bbox**를 도출합니다.

![ColFOV workflow](assets/fig1_workflow.png)

## 모델과 출력

4-class U-Net은 487,316개 파라미터를 사용하며, RGB 입력을 종횡비를 유지한
512 × 384 letterbox로 처리합니다. 모델의 직접 출력은 semantic mask입니다.
Bbox는 별도 object detector가 아니라 이 마스크를 이용한 후처리로 계산합니다.

| Label | 영상 내용 |
|---|---|
| 0 — Background/UI | 주 내시경 시야 밖의 녹화 인터페이스와 배경 |
| 1 — Tissue field | 조직, 내강, 수술 도구를 포함하는 주 내시경 영상 |
| 2 — Optical corner | 광학 시야의 사각 영역 안에 있는 비조직 모서리 |
| 3 — PiP | 내시경 보조 영상과 메뉴 등을 포함하는 보조 화면 |

주 시야에 대해 두 가지 입력 영역을 제공합니다.

- **Full-FOV bbox**는 광학 모서리를 포함한 예측 내시경 시야를 보존합니다.
  주변부 조직이나 관찰 대상을 유지하는 것이 중요한 분석에 사용할 수 있습니다.
- **Inner-FOV bbox**는 안정적으로 예측된 조직 안에 1.25 종횡비의 사각형을 배치하여
  광학 모서리와 녹화 인터페이스를 제외합니다. 일부 주변부 조직도 제외되므로
  분석 목적과 후속 모델의 학습 입력을 고려해 선택해야 합니다.

두 bbox는 영상별로 도출하며 화면 구성이 유지되는 동안 고정됩니다.
연구에서는 공개 영상과 병원 영상을 평가하고, 별도로 미지 processor family에서
시야 위치 추정 성능을 확인했습니다. Depth estimation과 주석 보존 실험은 입력 영역
선택의 영향을 평가하며, 한 crop이 모든 후속 모델의 성능을 개선한다는 뜻은 아닙니다.

**선택적 PiP 처리**는 검출된 보조 화면의 내시경 영상 여부와 시간적 안정성을 확인한
후 bbox를 활성화합니다. 보조 화면 검출과 내시경 입력으로 사용할 수 있다는 판단은
다르며, 활성 PiP bbox는 시간에 따라 달라질 수 있습니다.

## 설치

```bash
git clone https://github.com/bgmseo/ColFOV.git
cd ColFOV
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

## 사용

영상과 이미지는 별도로 준비해야 하며, 임상 미디어는 저장소에 포함되지 않습니다.

```bash
# 단일 이미지 semantic mask
python examples/infer_image.py --image /path/to/frame.png --out outputs/image

# PiP 모니터링 없이 영상별 FOV bbox 계산
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session --no-pip

# PiP 처리를 포함하는 영상 분석
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session
```

CUDA 지원 PyTorch 환경에서는 `--device cuda`를 추가합니다.
기본 체크포인트는 `weights/colfov_b8s3.pt`이며 로딩 시 구조와 체크섬을 검증합니다.

영상 예제는 `session_result.json`과 `calibration_summary.json`을 생성합니다.
`full_fov_box`와 `inner_fov_box`는 원본 프레임 좌표의 `(x1, y1, x2, y2)`이며,
`x2`, `y2`는 포함하지 않는 상한입니다.

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

`null`은 필요한 검사를 통과한 bbox가 없다는 뜻입니다. 사유 코드를 확인하고
임의의 대체 영역을 ColFOV 출력으로 취급하지 마세요. 단일 이미지 추론은 마스크와
프레임 단위 PiP 후보를 반환하며, 영상별 FOV bbox를 계산하지 않습니다.

PiP 활성 시 `monitor_records.jsonl`의 `active_session_pip_box`를 시점별로 사용합니다.
요약의 `final_active_session_pip_box`는 마지막 상태이므로 이전 프레임 전체에
적용하면 안 됩니다.

`--no-pip` (API: `pip_enabled=False`)은 주 시야 calibration을 바꾸지 않고 PiP
모니터링, 내용 확인과 좌표 갱신을 건너뜁니다. 모니터 파일도 생성하지 않습니다.
`pip_disabled`는 처리하지 않았다는 뜻이지 PiP가 없다는 판정이 아닙니다.

## 데이터와 인용

공개 데이터는 각 배포처의 이용 조건에 따라 제공됩니다.

- [REAL-Colon](https://doi.org/10.1038/s41597-024-03359-0)
- [C3VDv2](https://doi.org/10.1038/s41597-026-07471-1)
- [CAS-Colon](https://doi.org/10.1038/s41597-025-05588-3)

병원 데이터는 재배포하지 않습니다. 논문 인용 정보와 라이선스는 확정 후 추가합니다.
이 패키지는 연구용이며 임상 판단을 위한 도구가 아닙니다.

## 테스트

```bash
pytest -q
```

합성 fixture로 모델 로딩, 마스크 출력, bbox 도출과 PiP 제어를 검사합니다.
