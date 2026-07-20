import gymnasium as gym
import numpy as np
from env_wrapper import CarEnvironment
from agent import Agent

def run_local_test():
    print("=== 시작: 로컬 환경 테스트 ===")
    
    # 환경 초기화 (화면에 렌더링하도록 설정)
    # API 명세에 따라 continuous=True 적용[cite: 1]
    env = gym.make('CarRacing-v2', continuous=True, render_mode='human')
    env = CarEnvironment(env)
    
    # 참가자 에이전트 로드
    print("에이전트를 초기화합니다...")
    agent = Agent()
    
    # 환경 리셋 및 에이전트 리셋
    observation, info = env.reset(seed=42)
    agent.reset(observation)
    
    total_reward = 0
    steps = 0
    done = False
    
    print("시뮬레이션을 시작합니다.")
    while not done and steps < 2000:
        # 에이전트가 행동 결정
        action = agent.act(observation)
        
        # 환경에 행동 적용
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 100 == 0:
            print(f"진행 스텝: {steps}, 누적 보상: {total_reward:.2f}")

    print("=== 종료: 로컬 환경 테스트 ===")
    print(f"최종 스텝: {steps}")
    print(f"최종 누적 보상: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    run_local_test()