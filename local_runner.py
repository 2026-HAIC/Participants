import argparse

import numpy as np
from gymnasium.wrappers import TimeLimit

from core.vendor.car_racing import CarRacing
from env_wrapper import CarEnvironment
from agent import Agent


def run_local_test(track_id, seed, max_steps, frame_skip):
    print("=== 시작: 로컬 환경 테스트 ===")
    
    raw_frame_budget = max_steps * frame_skip + 200
    env = CarEnvironment(
        TimeLimit(
            CarRacing(continuous=True, render_mode="human"),
            max_episode_steps=raw_frame_budget,
        ),
        skip_frames=frame_skip,
    )
    
    # 참가자 에이전트 로드
    print("에이전트를 초기화합니다...")
    agent = Agent()
    
    # 환경 리셋 및 에이전트 리셋
    observation, info = env.reset(seed=seed, options={"track_id": track_id})
    agent.reset(observation)
    
    total_reward = 0
    steps = 0
    done = False
    
    print("시뮬레이션을 시작합니다.")
    while not done and steps < max_steps:
        # 에이전트가 행동 결정
        action = agent.act(observation)
        
        # 환경에 행동 적용
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 100 == 0:
            print(
                f"진행 스텝: {steps}, 누적 보상: {total_reward:.2f}, "
                f"손상: {info['damage']:.0%}"
            )

    print("=== 종료: 로컬 환경 테스트 ===")
    print(f"최종 스텝: {steps}")
    print(f"최종 누적 보상: {total_reward:.2f}")
    if info.get("retire_reason"):
        print(f"리타이어 사유: {info['retire_reason']}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 HAIC 공식 로컬 주행 환경")
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()
    run_local_test(args.track_id, args.seed, args.max_steps, args.frame_skip)
