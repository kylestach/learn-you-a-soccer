from robocup_env.envs.base.robocup import RoboCup
from torch.utils.tensorboard import SummaryWriter

import logging
import time
import random
import numpy as np
import torch

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory
from wrappers.normalized_actions import NormalizedActions

save_dir = "./saved_models"
log_dir = "./runs"

env_name = "collect"
seed = 1337

gamma = 0.99
tau = 0.001
noise_stddev = 0.5
hidden_size = [400, 300]
test_cycles = 10

replay_size = 1e6
load_model = False
render_train = False
render_eval = False
timesteps = 1e7
batch_size = 128
n_test_cycles = 10  # Num. of episodes in the evaluation phases
TEST_EVERY_N_EPISODES: int = 100
SAVE_EVERY_N_EPISODES: int = 10000


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
    writer = SummaryWriter(f"{log_dir}/{env_name}_{get_run_num()}")

    env = NormalizedActions(RoboCup())

    reward_threshold = np.inf

    # Set a random seed for all used libraries
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    memory = ReplayMemory(int(replay_size))

    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=noise_stddev * np.ones(nb_actions))

    # Counters
    start_step = 0
    if load_model:
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Create logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using {}".format(device))

    # Start training
    logger.info('Train agent on {} env'.format(env_name))
    logger.info('Doing {} timesteps'.format(timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    episode_num = 0

    try:
        while timestep <= timesteps:
            ou_noise.reset()
            epoch_return = 0

            state = torch.Tensor([env.reset()]).to(device)
            episode_start_time = time.time()
            episode_timestep = 0
            while True:
                if render_train:
                    env.render()

                action = agent.calc_action(state, ou_noise)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                episode_timestep += 1
                epoch_return += reward

                mask = torch.Tensor([done]).to(device)
                reward = torch.Tensor([reward]).to(device)
                next_state = torch.Tensor([next_state]).to(device)

                memory.push(state, action, mask, next_state, reward)

                state = next_state

                epoch_value_loss = 0
                epoch_policy_loss = 0

                if done:
                    break
            timestep += episode_timestep
            episode_time = time.time() - episode_start_time
            fps = episode_timestep / episode_time

            writer.add_scalar('episode/episode_time', episode_time, episode_num)
            writer.add_scalar('episode/fps', fps, episode_num)

            rewards.append(epoch_return)
            value_losses.append(epoch_value_loss)
            policy_losses.append(epoch_policy_loss)
            writer.add_scalar('epoch/return', epoch_return, epoch)

            # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
            if episode_num % TEST_EVERY_N_EPISODES == 0:
                t += 1
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

                mean_test_rewards.append(np.mean(test_rewards))

                for name, param in agent.actor.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                for name, param in agent.critic.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
                logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                            "mean reward: {}, mean test reward {}".format(epoch,
                                                                          timestep,
                                                                          rewards[-1],
                                                                          np.mean(rewards[-10:]),
                                                                          np.mean(test_rewards)))

                # Save if the mean of the last three averaged rewards while testing
                # is greater than the specified reward threshold
                # TODO: Option if no reward threshold is given
                if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                    agent.save_checkpoint(timestep, memory)
                    time_last_checkpoint = time.time()
                    logger.info('Mean test reward >= reward threshold - saved model at {}'.format(
                        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

                # Save every N episodes
                if episode_num % SAVE_EVERY_N_EPISODES == 0:
                    agent.save_checkpoint(timestep, memory)
                    logger.info(
                        'Saving every {} episodes ({}) - Saved model at {}'.format(SAVE_EVERY_N_EPISODES, episode_num,
                                                                                   time.strftime(
                                                                                       '%a, %d %b %Y %H:%M:%S GMT',
                                                                                       time.localtime())))

            epoch += 1
            episode_num += 1
    except KeyboardInterrupt:
        agent.save_checkpoint(timestep, memory)
        logger.info(
            'Got KeyboardInterrupt at episode {} - Saved model at {}'.format(episode_num,
                                                                             time.strftime(
                                                                                 '%a, %d %b %Y %H:%M:%S GMT',
                                                                                 time.localtime())))
        env.close()
        exit(0)

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()


if __name__ == '__main__':
    main()
