import numpy as np

class Agent:
    def __init__(self):
        """
        [필수 구현] 에이전트 초기화
        - 평가 서버에서 Agent 객체를 생성할 때 호출됩니다.
        - 모델 가중치(pth 등)를 로드하거나 하이퍼파라미터를 초기화하세요.
        - 주의: 초기화 시간이 너무 오래 걸리면 타임아웃으로 리타이어 처리될 수 있습니다.
        """
        # 예시: self.model = torch.load('my_model.pth')
        pass

    def reset(self, observation):
        """
        [선택 구현] 매 트랙(에피소드)이 시작될 때마다 호출됩니다.
        - 이전 트랙의 잔여 상태(RNN 히든 스테이트 등)를 초기화할 때 사용합니다.
        """
        pass

    def act(self, observation) -> np.ndarray:
        """
        [필수 구현] 현재 관측 상태를 바탕으로 다음 행동을 결정합니다.
        
        :param observation: 현재 게임 화면 (Stacking 및 Grayscale 전처리 완료 상태)
        :return: np.ndarray shape (3,) 
                 [steer, gas, brake]
                 - steer: -1.0 (좌) ~ 1.0 (우)
                 - gas: 0.0 ~ 1.0 (가속)
                 - brake: 0.0 ~ 1.0 (제동)
        """
        # [작성 예시] 
        # action = self.model.predict(observation)
        # return action
        
        # 더미 액션 반환 (직진)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
