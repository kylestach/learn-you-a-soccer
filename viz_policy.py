from robocup_env.envs.robocup import RoboCup
import logging
import numpy as np
import torch

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from wrappers.normalized_actions import NormalizedActions

save_dir = "./saved_models"
log_dir = "./runs"

env_name = "collect"
seed = 1337

gamma = 0.99
tau = 0.001
noise_stddev = 0.5
hidden_size = [400, 300]
render_eval = True

n_test_cycles = 50  # Num. of episodes in the evaluation phases


def create_runs_dir():
    from pathlib import Path
    Path("./runs").mkdir(parents=True, exist_ok=True)


def get_run_num() -> int:
    return 1
    # from os import listdir
    # from os.path import isdir, join
    # import re
    #
    # folders = [f for f in listdir(log_dir) if isdir(join(log_dir, f))]
    # run_num_pattern = f"{env_name}_([0-9]+)"
    # for f in folders:
    #     result = re.search(run_num_pattern, f)
    #     if result is not None:


def main():
    checkpoint_dir = f"{save_dir}/{env_name}"
    create_runs_dir()

    env = NormalizedActions(RoboCup())

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Defined and build a DDPG agent
    observation_space_size = 10
    agent = DDPG(gamma, tau, hidden_size,
                 # env.observation_space.shape[0],
                 observation_space_size,
                 env.action_space,
                 checkpoint_dir=checkpoint_dir)

    # checkpoint_path = None
    checkpoint_path = "saved_models/collect/ep_2886522.pth.tar"
    start_step, memory = agent.load_checkpoint(checkpoint_path)

    # Create logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using {}".format(device))

    # Start training
    logger.info('Testing agent on {} env'.format(env_name))

    test_rewards = []
    for j in range(n_test_cycles):
        state = torch.Tensor([env.reset()]).to(device)

        test_reward = 0
        while True:
            if render_eval:
                env.render()

            action = agent.calc_action(state)  # Selection without noise

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            test_reward += reward

            next_state = torch.Tensor([next_state]).to(device)

            state = next_state
            if done:
                break
        test_rewards.append(test_reward)

    logger.info("Mean test reward {}".format(np.mean(test_rewards)))

    env.close()


if __name__ == '__main__':
    main()
