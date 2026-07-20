import gymnasium as gym
import cv2
import numpy as np

def image_preprocessing(img):
    """
    이미지 해상도를 84x84로 축소하고 흑백으로 변환하여 연산량을 줄입니다[cite: 3].
    """
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class CarEnvironment(gym.Wrapper):
    """
    프레임 스킵과 스태킹을 적용한 커스텀 환경 래퍼입니다[cite: 3].
    """
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        self.stack_state = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # 초기 시작 시 카메라 줌인 대기 시간 동안 아무 동작도 하지 않음[cite: 3]
        for _ in range(self._no_operation):
            # Continuous Action Space: [steer, gas, brake]
            observation, _, terminated, truncated, _ = self.env.step(np.array([0.0, 0.0, 0.0]))
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)

        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        for _ in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info