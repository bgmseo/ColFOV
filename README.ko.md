# ColFOV

[English](README.md)

ColFOV는 내시경 프레임을 의미 기반의 네 클래스로 분할하고, 그 결과로부터
`inner_fov_box`, `full_fov_box`, 조건부 세션 단위
`active_session_pip_box`를 산출합니다.

코드와 모델 가중치는 논문과 관련된 과학적 투명성을 위해 공개되어 있습니다.
이 저장소는 **오픈소스 소프트웨어가 아닙니다**. 자세한 조건은
[라이선스](#라이선스)를 확인하십시오.

아래 명령은 공개된 인터페이스를 설명하기 위한 것입니다. 저장소 라이선스는
코드나 가중치를 실행하거나 다른 방식으로 사용할 권리를 부여하지 않습니다.
사용하려면 저작권자의 사전 서면 허가가 필요합니다.

![ColFOV workflow](assets/fig1_workflow.png)

## 모델이 출력하는 것과 출력하지 않는 것

신경망이 직접 출력하는 것은 **분할 마스크 하나뿐**입니다.

```text
logits  [B, 4, H, W]
mask    [H, W], values in {0, 1, 2, 3}
```

| 값 | 클래스 | 의미 |
|---:|---|---|
| 0 | `background_ui` | 화면 인터페이스, 레터박스 및 광학 시야 밖의 영역 |
| 1 | `valid_fov_tissue` | 사용할 수 있는 내시경 조직 영역 |
| 2 | `black_corner` | 프레임 주변부의 어두운 영역 |
| 3 | `popup_overlay` | 화면 위에 표시된 보조 창 |

![Label examples](assets/sup_fig3_label_pairs.png)

두 이미지는 동반 원고를 위해 준비된 figure를 평면화한 사본입니다. 데이터셋이나
inference 입력 예시가 아닙니다. 권리와 출처는 `assets/RIGHTS.md`와
`assets/PROVENANCE.md`를 확인하십시오.

**세 bounding box는 모두 신경망의 직접 출력이 아니라 분할 마스크에서 계산되는
파생 geometry입니다.** 각 결과는 근거가 부족하면 `null`일 수 있습니다.

| 필드 | 계산 근거 | `null`이 되는 경우 |
|---|---|---|
| `inner_fov_box` | 24개 calibration frame에서 안정적인 class-1 조직 영역 | admissible frame 비율이 50% 미만일 때 |
| `full_fov_box` | 같은 24개 frame의 안정적인 class-1 또는 class-2 영역 | 같은 조건 |
| `active_session_pip_box` | recording 전체를 시간 순서로 causal monitoring | lock되지 않았거나, 근거·TTL·안정성 조건을 통과하지 못했을 때 |

세 key는 session output에 항상 존재합니다. `null`은 reason code를 동반하는
**fail-closed abstention**이며 오류나 임의의 fallback box가 아닙니다.

### `active_session_pip_box`는 시간에 따라 달라집니다

이 값은 recording 전체에 하나로 고정되는 속성이 아닙니다. 해당 시점까지 도착한
근거만 사용해 **그 시점에 활성화된 box**를 나타냅니다. PiP 창이 이동하는
recording에는 시간에 따라 둘 이상의 올바른 box가 존재할 수 있습니다.

`session_result.json`의 `final_active_session_pip_box`는 **마지막 상태만**
나타냅니다. 이를 과거 frame에 소급 적용하지 마십시오. 시점별 결과는
`monitor_records.jsonl`을 사용하십시오. 각 행에는 `frame_index`,
`timestamp_sec`, `active_session_pip_box`, `pip_state`, `pip_epoch`,
`pip_event`(`lock` / `unlock` / `relock` / `null`)와 `pip_reason`이 기록됩니다.

## 모델

| 항목 | 값 |
|---|---|
| architecture | TinyUNet, 4 classes, 8 base channels |
| parameters | 487,316 |
| input | 512 × 384, letterbox, `/255` normalization only |
| checkpoint | `weights/colfov_b8s3.pt`, 1.908 MiB |
| sha256 | `df4054b8d2a22413ae232b7ea2ce01cbe70c838031167b5afb72c71ce9670713` |

loader는 checkpoint 내부 config에서 architecture를 읽고, 4 classes와 8 base
channels를 강제하며, state dict를 strict mode로 불러오고 SHA-256을 검사합니다.
`TinyUNet`에는 architecture 기본값이 없으므로 다른 구조로 조용히 잘못 불러오는
경로가 없습니다.

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

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

그다음:

```bash
pip install -e ".[test]"
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('weights/colfov_b8s3.pt').read_bytes()).hexdigest())"
```

## 데이터

**이 저장소에는 모델 입력용 독립 데이터셋, sample frame 또는 video가 없습니다.**
`assets/RIGHTS.md`의 제한 조건 아래 동반 원고 figure를 평면화한 사본 두 개만
포함합니다. 입력은 사용자가 직접 준비하여 `--image` 또는 `--video`로 전달해야
합니다.

연구에서 사용한 공개 데이터셋은 각 관리자가 자신의 조건으로 배포합니다. 다음
공식 페이지에서 직접 받아 해당 조건을 준수하십시오. 이 저장소는 데이터 파일을
복제하거나 재배포하지 않습니다.

| 데이터셋 | 공식 출처 |
|---|---|
| REAL-Colon | [공식 Figshare 배포 페이지](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) |
| C3VDv2 | [공식 프로젝트 및 다운로드 페이지](https://durrlab.github.io/C3VDv2/) |
| CAS-Colon | [공식 Figshare DOI](https://doi.org/10.6084/m9.figshare.28287929) |

학습 자료의 일부는 공개 재배포할 수 없는 병원 데이터셋이며 이 저장소에서 링크하거나
배포하지 않습니다.

## 실행 인터페이스

단일 frame — 4-class mask와 frame-local diagnostics:

```bash
python examples/infer_image.py --image /path/to/frame.png --out outputs/image_example
```

`mask.png`, `overlay.png`, `frame_result.json`을 기록합니다. 단일 frame만으로는
FOV box나 session PiP box를 만들지 않습니다. FOV box는 calibration sample이,
PiP box는 시간 순서의 causal evidence가 필요합니다.

전체 recording — 세 geometry product:

```bash
python examples/infer_session.py --video /path/to/video.mp4 --out outputs/session_example
```

`session_result.json`, `calibration_summary.json`, `monitor_records.jsonl`을
기록합니다.

### Output schema

다음은 결과 수치가 아니라 key와 shape만 보여주는 placeholder입니다.

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

## Session pass의 동작

1. **Calibration:** recording 전체에서 정확히 24개의 evenly-spaced frame을 먼저
   선택하고, 해당 sample에 model-free admissibility guard를 적용합니다. admissible
   비율이 0.50 미만이면 FOV geometry는 abstain합니다.
2. **Monitoring:** recording을 시간 순서로 기본 5 Hz에서 replay합니다. fractional
   phase scheduler를 사용하므로 장기 관측률은 정확히
   `min(source_fps, monitor_hz)`가 됩니다.
3. **Qualification:** evidence와 stability gate를 통과해야 session box가 lock됩니다.
   위치 변경이 확인되면 unlock하고 epoch를 회전하며, 이전 위치를 만든 근거로 즉시
   relock하지 않습니다.

session example은 **causal fixed-cadence offline replay**입니다. capture card timing,
asynchronous decoding 또는 live-stream deadline을 검증한 real-time system이라고
주장하지 않습니다.

## 테스트

```bash
pytest -q
```

57개 테스트가 checkpoint gate, 4-class output, 24-frame geometry contract, cadence,
causal state transition과 fail-closed decoding을 검사합니다. 모든 테스트는 합성
fixture만 사용하며 임상 media를 요구하거나 배포하지 않습니다. 이는 소프트웨어
정합성 검증이지 임상 성능의 근거가 아닙니다.

## 사용 목적과 한계

- 과학적 투명성을 위한 공개 기록입니다. 실행 또는 그 밖의 사용에는 사전 서면
  허가가 필요합니다. 의료기기, 진단 또는 임상 의사결정용이 아닙니다.
- 논문에서 보고한 평가 결과는 저장소에서 headline 수치로 반복하지 않습니다.
- 논문의 평가는 제한된 recording을 사용한 development evaluation이며 독립적인
  confirmatory validation이 아닙니다.
- development material에 없던 장비·해상도·overlay style에서의 동작은 알려지지
  않았습니다.
- 세 geometry product는 조건을 충족하지 못하면 의도적으로 abstain합니다.

## 라이선스

소스 코드, 모델 가중치, config, 문서, test fixture와 figure를 포함한 저장소의 모든
내용은 **All Rights Reserved**입니다. 공개되어 있다는 사실은 실행·사용·수정·재배포
또는 파생물 작성 권한을 부여하지 않습니다. 사용하려면 저작권자의 서면 허가가
필요합니다. 법적 효력을 갖는 전체 조건은 영문 `LICENSE`를 따릅니다.

Figure에도 같은 조건이 적용됩니다. figure에 합성된 임상 frame은 별도 사용이나
배포가 허가된 데이터가 아닙니다(`assets/RIGHTS.md`).

## 인용

동반 원고는 아직 투고 전입니다. 출판 후 이 문서와 `CITATION.cff`에 인용 정보를
추가할 예정입니다.
