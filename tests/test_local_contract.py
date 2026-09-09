import time
import unittest

import gymnasium as gym
import numpy as np

from damage import CollisionDamage
from env_wrapper import CarEnvironment, image_preprocessing
from local_runner import safe_act, safe_reset


class TestAgentContract(unittest.TestCase):
    def test_safe_act_accepts_and_clips_valid_action(self):
        class Agent:
            def act(self, observation):
                return [2.0, -1.0, 0.5]

        action, valid = safe_act(Agent(), None)

        self.assertTrue(valid)
        np.testing.assert_array_equal(action, [1.0, 0.0, 0.5])

    def test_safe_act_replaces_invalid_results(self):
        agents = [
            type("MissingAct", (), {})(),
            type("BadShape", (), {"act": lambda self, observation: [0.0, 1.0]})(),
            type("NotFinite", (), {"act": lambda self, observation: [0.0, np.nan, 0.0]})(),
        ]

        for agent in agents:
            with self.subTest(agent=type(agent).__name__):
                action, valid = safe_act(agent, None)
                self.assertFalse(valid)
                np.testing.assert_array_equal(action, [0.0, 0.0, 0.0])

    def test_safe_act_times_out(self):
        class SlowAgent:
            def act(self, observation):
                time.sleep(0.1)
                return [0.0, 0.0, 0.0]

        action, valid = safe_act(SlowAgent(), None, timeout_sec=0.01)

        self.assertFalse(valid)
        np.testing.assert_array_equal(action, [0.0, 0.0, 0.0])

    def test_safe_reset_is_optional_and_propagates_errors(self):
        safe_reset(object(), None)

        class BrokenReset:
            def reset(self, observation):
                raise RuntimeError("reset failed")

        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            safe_reset(BrokenReset(), None)


class _DummyCar:
    def set_damage_effects(self, grip, engine, steering):
        self.effects = (grip, engine, steering)


class _DummyEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.car = _DummyCar()
        self.track = [None] * 10
        self.tile_visited_count = 0

    def reset(self, *, seed=None, options=None):
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.full((96, 96, 3), 255, dtype=np.uint8), 0.0, False, False, {
            "collision": False,
        }


class TestEnvironmentContract(unittest.TestCase):
    def test_preprocessing_returns_float32_unit_range(self):
        image = np.full((96, 96, 3), 255, dtype=np.uint8)

        observation = image_preprocessing(image)

        self.assertEqual(observation.shape, (84, 84))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(float(observation.min()), 1.0)
        self.assertEqual(float(observation.max()), 1.0)

    def test_wrapper_observation_matches_declared_space(self):
        environment = CarEnvironment(_DummyEnvironment(), no_operation=0)

        observation, _info = environment.reset()

        self.assertEqual(environment.observation_space.shape, (4, 84, 84))
        self.assertEqual(environment.observation_space.dtype, np.float32)
        self.assertTrue(environment.observation_space.contains(observation))

    def test_fifth_collision_retires(self):
        damage = CollisionDamage()

        for _ in range(4):
            self.assertFalse(damage.update(True))
        self.assertTrue(damage.update(True))
        self.assertEqual(damage.damage, 1.0)


if __name__ == "__main__":
    unittest.main()
