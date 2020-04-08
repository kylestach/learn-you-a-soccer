from robocup import RoboCup
from torch.utils.tensorboard import SummaryWriter

import logging
import time
import random
import numpy as np
import torch

import gym
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
noise_stddev = 0.2
hidden_size = [400, 300]
test_cycles = 10

replay_size = 1e6
load_model = False
render_train = False
render_eval = True
timesteps = 1e8
batch_size = 128
n_test_cycles = 10  # Num. of episodes in the evaluation phases


def create_runs_dir():
    from pathlib import Path
    Path("./runs").mkdir(parents=True, exist_ok=True)


def get_run_num() -> int:
    return 0
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

    while timestep <= timesteps:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).to(device)
        while True:
            if render_train:
                env.render()

            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
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
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()


if __name__ == '__main__':
    main()
