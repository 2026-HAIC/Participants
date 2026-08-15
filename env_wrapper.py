import gymnasium as gym
import cv2
import numpy as np

from damage import CollisionDamage

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
        self.damage = CollisionDamage()

    @property
    def warmup_steps(self):
        return self._no_operation

    def _apply_damage_effects(self):
        effects = self.damage.effects
        self.unwrapped.car.set_damage_effects(
            effects.grip_multiplier,
            effects.engine_multiplier,
            effects.steering_multiplier,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        # 초기 시작 시 카메라 줌인 대기 시간 동안 아무 동작도 하지 않음[cite: 3]
        for _ in range(self._no_operation):
            # Continuous Action Space: [steer, gas, brake]
            observation, _, terminated, truncated, _ = self.env.step(np.array([0.0, 0.0, 0.0]))
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)

        self.damage.reset()
        self._apply_damage_effects()
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        collision = False
        for _ in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            collision = collision or bool(info.get("collision", False))
            if terminated or truncated:
                break

        crashed = self.damage.update(collision)
        self._apply_damage_effects()

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        info = dict(info)
        info.update(
            collision=collision,
            damage=self.damage.damage,
            damage_effects=self.damage.effects,
            retire_reason="crash" if crashed else None,
        )
        terminated = terminated or crashed
        return self.stack_state, total_reward, terminated, truncated, info
