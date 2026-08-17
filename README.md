# 2026 HAIC - CarRacing AI Challenge 🏎️

2026 HAIC 참가자를 위한 공식 로컬 학습·테스트 템플릿입니다. 참가자는 `agent.py`의
`Agent`를 구현하고, 공식 평가와 동일한 장애물 및 충돌 손상 환경에서 모델을 확인할 수 있습니다.

## 설치

Python 가상 환경을 만든 뒤 활성화하고 의존성을 설치합니다. Python 3.10 또는 3.11을
권장합니다.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`의 `gymnasium[box2d]` 항목이 Gymnasium과 Box2D를 함께 설치합니다.
설치 여부는 다음 명령으로 확인할 수 있습니다.

```bash
python -c "import gymnasium, Box2D; print('설치 완료')"
```

## 로컬 실행

```bash
python local_runner.py --track-id 1 --seed 42
```

`track-id`와 `seed`가 같으면 트랙과 장애물 배치도 동일합니다. `42`는 로컬 연습용
예시일 뿐 공식 평가 시드가 아닙니다. 공개 트랙 시드가 제공되면 해당 값을 사용하십시오.

주요 옵션:

```text
--track-id    트랙 식별자 (기본 1)
--seed        트랙 생성 시드 (기본 42)
--max-steps   에이전트 행동 횟수 상한 (기본 2000)
--frame-skip  행동 하나를 유지할 raw frame 수 (기본 4)
```

## 모델 인터페이스

수정 대상은 기본적으로 `agent.py`입니다.

```python
class Agent:
    def __init__(self):
        pass

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.array([steer, gas, brake], dtype=np.float32)
```

- 관측값: 84×84 흑백 이미지 4프레임 스택
- `steer`: `-1.0`~`1.0`
- `gas`: `0.0`~`1.0`
- `brake`: `0.0`~`1.0`

## 공식 환경의 물리 변인

표준 Gymnasium `CarRacing-v2`와 달리 도로 위에 피할 수 있는 장애물이 배치됩니다.
장애물은 `(track_id, seed)`로 결정되며 같은 입력에서는 항상 같은 위치에 생성됩니다.
장애물 하나와 계속 붙어있는 동안은 한 번만 손상으로 집계되지만, 완전히
떨어졌다가 같은 장애물에 다시 부딪히면 별개의 충돌로 다시 집계됩니다.

| 상태 | 접지력 | 엔진 출력 | 조향 반응 |
|---|---:|---:|---:|
| 충돌 전 | 100% | 100% | 100% |
| 1회 충돌 | 90% | 95% | 95% |
| 2회 충돌 | 80% | 90% | 90% |
| 3회 충돌 | 70% | 85% | 85% |
| 4회 충돌 | 60% | 80% | 80% |
| 5회 충돌 | 리타이어 | 리타이어 | 리타이어 |

프레임 스킵 도중 발생한 충돌도 누락하지 않고 한 번의 손상 이벤트로 반영합니다.
새 트랙을 시작하면 누적 손상은 초기화됩니다. 잔디 마찰은 현재 표준 환경과 같은
`0.6`을 유지합니다.

## 파일 안내

- `agent.py`: 참가자가 구현할 에이전트
- `local_runner.py`: 공식 로컬 환경 실행기
- `env_wrapper.py`: 관측 전처리, 프레임 스킵·스택 및 손상 적용
- `damage.py`: 공식 충돌 손상 계산식
- `core/track_variables.py`: 결정적 장애물 배치
- `core/vendor/`: 공식 평가에 사용하는 vendored CarRacing 물리 환경

`core/`, `env_wrapper.py`, `damage.py`를 임의로 수정하면 로컬 결과와 서버 평가 결과가
달라질 수 있습니다.

## 제출

제출 ZIP의 루트에는 최소한 `agent.py`가 있어야 합니다. 모델 가중치 등 `Agent` 실행에
필요한 파일도 함께 포함하십시오. `local_runner.py`, `env_wrapper.py`, `damage.py`, `core/`는
로컬 학습·테스트용이며 제출할 필요가 없습니다.
