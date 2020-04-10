from typing import Tuple
import numpy as np
import torch
import gym
import argparse
import os
from robocup_env.envs import RoboCup

import utils
import TD3


# Runs policy for X episodes and returns average reward
def viz_policy(policy, env_name, seed, eval_episodes=10) -> float:
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_reward = 0
        while not done:
            eval_env.render()
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
        print("episode reward: ", episode_reward)
        avg_reward += episode_reward

    avg_reward /= eval_episodes
    eval_env.close()

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def get_latest() -> Tuple[str, int]:
    import re
    models_dir = "./models/"
    files = [file for file in os.listdir(models_dir)]
    filepaths = [os.path.join(models_dir, file) for file in files]
    last_file = max(filepaths, key=os.path.getctime)

    root_re = r"(.*)_(\d+)_(?:actor|critic)(?:_optimizer)?"
    root_string = re.match(root_re, last_file).group(1)
    timestep = int(re.match(root_re, last_file).group(2))

    root_string = f"{root_string}_{timestep}"

    return root_string, timestep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="robocup_env:robocup-collect-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    env = gym.make(args.env)

    state_dim = 10
    # state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    policy = TD3.TD3(**kwargs)

    last_timestep = 0
    while True:
        policy_file, timestep = get_latest()
        policy.load(policy_file)

        if timestep != last_timestep:
            print(f"Timestep: {timestep}")
            last_timestep = timestep
        viz_policy(policy, args.env, args.seed, 10)


if __name__ == "__main__":
    main()
