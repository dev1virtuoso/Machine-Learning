import argparse
import os
import time
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from master import HeraclesMaster, MotorCommand

class SimpleHeraclesEnv(gym.Env):
    def __init__(self, port=None):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(1,), dtype=np.float32)
        self.master = HeraclesMaster(port)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.master.send_motor_command(MotorCommand(1, 0x01, 0, 1500))
        return np.array([1500.0], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        val = int(1500 + action[0] * 5)
        self.master.send_motor_command(MotorCommand(1, 0x01, 0, val))
        obs = np.array([float(val)], dtype=np.float32)
        reward = -abs(action[0] - 50)
        done = self.step_count >= 100
        return obs, reward, done, {}

    def close(self):
        self.master.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=5000)
    args = parser.parse_args()

    env = SimpleHeraclesEnv()
    vec_env = DummyVecEnv([lambda: env])

    if args.train:
        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save("heracles_ppo")
    else:
        model = PPO.load("heracles_ppo", env=vec_env)
        mean_r, std_r = evaluate_policy(model, vec_env, n_eval_episodes=5)
        print(f"Evaluation: {mean_r:.2f} ± {std_r:.2f}")

    env.close()

if __name__ == "__main__":
    main()