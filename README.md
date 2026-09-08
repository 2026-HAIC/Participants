# 2026 HAIC CarRacing AI Challenge

이 디렉터리는 2026 HAIC CarRacing AI Challenge 참가자를 위한 공식 로컬
학습·테스트 템플릿입니다.

참가자는 기본적으로 `agent.py`의 `Agent` 클래스를 구현합니다. 제공된 로컬 실행기로
트랙, 장애물, 프레임 전처리 및 차량 손상 규칙을 적용해 에이전트를 테스트할 수 있습니다.

## 1. 권장 환경

- Python 3.10 또는 3.11
- 공식 평가 서버: Python 3.11, CPU 환경
- Windows, macOS 또는 Linux

Python 3.12 이상에서는 고정된 PyTorch 및 Box2D 버전이 설치되지 않을 수 있습니다.
가상 환경을 만든 뒤 아래 명령을 실행하십시오.

### Windows PowerShell

```powershell
cd Participants
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
cd Participants
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import gymnasium, Box2D, torch, numpy, cv2; print('설치 완료')"
```

## 2. 첫 실행

```bash
python local_runner.py --track-id 1 --seed 42
```

기본 `agent.py`는 조향이나 제동 없이 가속하는 예시입니다. 정상적으로 창이 열리고
차량이 움직이면 환경 설치가 완료된 것입니다.

주요 옵션:

| 옵션 | 설명 | 기본값 |
|---|---|---:|
| `--track-id` | 트랙 식별자(1 이상의 정수) | `1` |
| `--seed` | 트랙 생성 시드 | `42` |
| `--max-steps` | 에이전트 행동 횟수 상한 | `2000` |
| `--frame-skip` | 한 행동을 유지할 raw frame 수 | `4` |

`track-id`와 `seed`가 같으면 트랙과 장애물 배치가 동일합니다. `42`는 설치 확인을
위한 예시이며 공식 평가 시드가 아닙니다. 공개 트랙 시드가 제공되면 해당 값을
사용하십시오.

## 3. Agent 구현

수정 대상은 [agent.py](agent.py)입니다.

```python
import numpy as np


class Agent:
    def __init__(self):
        # 모델 생성 및 가중치 로드
        pass

    def reset(self, observation):
        # 선택 구현: 트랙별 RNN 상태 등을 초기화
        pass

    def act(self, observation) -> np.ndarray:
        # 반드시 shape (3,)의 유한한 수를 반환
        return np.array([steer, gas, brake], dtype=np.float32)
```

### 관측값

| 항목 | 값 |
|---|---|
| shape | `(4, 84, 84)` |
| 의미 | 최근 4개의 흑백 프레임 |
| 값 범위 | `0.0`~`1.0` |
| 현재 dtype | `float64` |

PyTorch 모델이 `float32` 입력을 사용한다면 `act()` 안에서 변환하십시오.

```python
state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
```

### 행동값

`act()`는 `[steer, gas, brake]` 순서의 길이 3 배열을 반환해야 합니다.

| 인덱스 | 의미 | 허용 범위 |
|---:|---|---:|
| `0` | 조향: 음수는 좌회전, 양수는 우회전 | `-1.0`~`1.0` |
| `1` | 가속 | `0.0`~`1.0` |
| `2` | 제동 | `0.0`~`1.0` |

NaN, 무한대, 잘못된 shape 또는 예외는 유효하지 않은 행동으로 처리됩니다. 공식
서버는 해당 행동을 무동작 `[0, 0, 0]`으로 대체하며, 유효하지 않은 행동이 10회
연속 발생하면 해당 트랙에서 리타이어시킵니다. 범위를 벗어난 유한한 행동값은 허용
범위로 잘립니다.

### 모델 가중치 로드

공식 평가는 CPU 환경에서 실행됩니다. GPU에서 저장한 PyTorch 가중치는 CPU로
명시하여 로드하는 것을 권장합니다.

```python
self.model_path = __file__[:-len("agent.py")] + "model.pth"
self.model.load_state_dict(
    torch.load(self.model_path, map_location="cpu")
)
self.model.eval()
```

공식 서버의 현재 작업 디렉터리는 제출 ZIP의 루트와 다를 수 있습니다. 따라서
`torch.load("model.pth")`처럼 현재 작업 디렉터리에 의존하지 말고 위 예시처럼
`__file__`을 기준으로 제출 파일의 경로를 만드십시오. 모델 추론 중에는 일반적으로
`torch.no_grad()`를 사용하십시오.

## 4. 실행 제한

공식 평가의 기본 제한은 다음과 같습니다.

| 항목 | 제한 |
|---|---:|
| `Agent` import 및 생성 | 10초 |
| `reset()` 한 번 | 5초 |
| `act()` 한 번 | 5초 |
| 참가자 프로세스 메모리 | 1,024 MB |
| 유효하지 않은 행동 | 10회 연속 시 리타이어 |

`reset()`은 공식 서버에서 선택 메서드입니다. 다만 이 템플릿에는 기본 메서드가 이미
포함되어 있으므로, 사용하지 않는 경우 빈 메서드로 유지해도 됩니다.

공식 서버는 이 템플릿의 `requirements.txt`에 명시된 패키지를 사용합니다. 제출 ZIP에
별도의 `requirements.txt`를 넣거나 로컬에서 패키지를 추가해도 서버에 새 패키지가
설치되지 않습니다. 제출 전에 새로 만든 Python 3.11 가상 환경에서 에이전트가 제공된
`requirements.txt`만으로 실행되는지 확인하십시오.

## 5. 트랙과 장애물

환경은 Gymnasium `CarRacing-v2`를 기반으로 하며 도로 위에 물리 장애물 6개가
배치됩니다.

- 트랙과 장애물은 `(track_id, seed)`에 따라 결정됩니다.
- 같은 입력은 항상 같은 결과를 생성합니다.
- 장애물은 시작 및 결승 부근을 제외하고 서로 일정 거리 이상 떨어져 배치됩니다.
- 각 장애물은 한 트랙에서 최초 충돌 한 번만 손상으로 집계됩니다.
- 같은 장애물과 계속 접촉하거나 떨어졌다가 다시 충돌해도 추가 손상은 없습니다.
- 다른 장애물과 충돌하면 별도의 손상으로 집계됩니다.
- 프레임 스킵 도중 발생한 충돌도 누락하지 않습니다.

새 트랙이 시작되면 장애물의 충돌 기록과 차량 손상이 모두 초기화됩니다.

## 6. 차량 손상

장애물 충돌 1회당 손상이 20% 증가합니다.

| 누적 충돌 | 접지력 | 엔진 출력 | 조향 반응 | 결과 |
|---:|---:|---:|---:|---|
| 0회 | 100% | 100% | 100% | 정상 |
| 1회 | 90% | 95% | 95% | 주행 계속 |
| 2회 | 80% | 90% | 90% | 주행 계속 |
| 3회 | 70% | 85% | 85% | 주행 계속 |
| 4회 | 60% | 80% | 80% | 주행 계속 |
| 5회 | - | - | - | 충돌 리타이어 |

잔디 마찰 계수는 표준 환경과 같은 `0.6`입니다.

## 7. 종료와 점수

에피소드는 다음 조건 중 하나에서 종료됩니다.

- 트랙 타일의 95% 이상 방문
- 충돌 손상 100% 도달
- 음수 보상이 101 행동 스텝 연속 발생하여 오프트랙 판정
- 차량이 플레이 영역 밖으로 이탈
- `max-steps` 소진
- 유효하지 않은 행동 10회 연속 발생

오프트랙은 차량 좌표만으로 판단하지 않습니다. 한 행동의 프레임 스킵 구간에서 합산한
원본 CarRacing 보상이 음수이면 카운터가 증가하고, 0 이상이면 즉시 초기화됩니다.
따라서 정지하거나 역주행하여 새 도로 타일을 방문하지 않는 경우도 포함될 수 있습니다.

최종 점수:

- 완주(진행률 95% 이상): `점수 = 사용한 행동 스텝 수`
- 미완주: `점수 = -진행률`

완주 기록은 더 적은 스텝이 좋은 기록입니다. 미완주 기록은 진행률이 높을수록
`-1.0`에 가까워집니다.

로컬 실행 중 표시되는 `누적 보상`은 공식 점수가 아닙니다. 이는 원본 CarRacing의
학습 보상으로, raw frame마다 `-0.1`, 처음 방문한 타일마다 `+1000/N`이 적용됩니다.

## 8. 제출 ZIP

제출물은 `.zip` 형식이어야 하며 ZIP의 최상위에 `agent.py`가 있어야 합니다.

올바른 구조:

```text
submission.zip
├─ agent.py
├─ model.pth
└─ 필요한 사용자 모듈 및 데이터 파일
```

잘못된 구조:

```text
submission.zip
└─ Participants/
   └─ agent.py
```

제출물 제한:

| 항목 | 제한 |
|---|---:|
| ZIP 내부 파일 수 | 최대 1,000개 |
| 압축 해제 후 전체 크기 | 최대 2 GB |
| 개별 파일 크기 | 최대 10 MB |
| 개별 파일 압축률 | 최대 100배 |

현재 개별 파일 제한이 10 MB이므로 `model.pth`를 포함한 모든 파일이 각각 10 MB를
넘지 않아야 합니다. 여러 파일로 분할한 가중치를 사용할 때도 전체 메모리 제한을
준수해야 합니다.

제출물의 모든 `.py` 파일은 정적 검사를 받습니다. 다음 import는 허용되지 않습니다.

```text
ctypes, importlib, multiprocessing, os, pathlib, resource,
shutil, signal, socket, subprocess, sys
```

다음 동적 코드 실행 함수도 허용되지 않습니다.

```text
compile, eval, exec, __import__
```

실행 파일 및 네이티브 라이브러리 형식인 `.com`, `.dll`, `.dylib`, `.exe`, `.msi`,
`.scr`, `.so`도 ZIP에 포함할 수 없습니다.

### 제출 전 체크리스트

- ZIP 최상위에 `agent.py`가 있는가?
- `Agent` 클래스와 `act()` 메서드가 있는가?
- 모든 행동이 shape `(3,)`의 유한한 숫자인가?
- CPU 환경에서 모델을 로드할 수 있는가?
- 제공된 패키지만 사용하는가?
- 각 모델 파일이 10 MB 이하인가?
- 금지된 import와 함수가 없는가?
- `.venv`, `__pycache__`, 학습 데이터 및 체크포인트 백업본을 제외했는가?

## 9. 파일 구성

| 경로 | 역할 | 참가자 수정 여부 |
|---|---|---|
| `agent.py` | 에이전트 및 모델 추론 구현 | 수정 |
| `requirements.txt` | 로컬·서버 Python 패키지 버전 | 수정하지 않음 |
| `local_runner.py` | 로컬 주행 실행 및 점수 출력 | 제출 불필요 |
| `env_wrapper.py` | 전처리, 프레임 스킵·스택, 로컬 규칙 적용 | 수정하지 않음 |
| `damage.py` | 로컬 충돌 손상 계산 | 수정하지 않음 |
| `core/track_variables.py` | 결정적 장애물 배치 | 수정하지 않음 |
| `core/obstacle_contacts.py` | 장애물 접촉 판정 | 수정하지 않음 |
| `core/vendor/` | 공식 CarRacing 및 Box2D 차량 물리 | 수정하지 않음 |

`core/`, `env_wrapper.py`, `damage.py`를 변경하면 로컬 결과가 공식 평가와 달라질 수
있습니다. 이 파일들은 로컬 학습·테스트용이며 제출 ZIP에는 포함할 필요가 없습니다.

## 10. 문제 확인

환경 설치가 실패하면 다음 정보를 함께 확인하십시오.

```bash
python --version
python -m pip --version
python -m pip list
```

특히 Python 버전이 3.10 또는 3.11인지, 현재 가상 환경의 Python과 `pip`가 같은
경로를 사용하는지 먼저 확인하십시오.
