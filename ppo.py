from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

import gym

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    def create_robocup():
        from gym.envs.registration import register
        import robocup
        register(
            id='RoboCup-collect-v0',
            entry_point=robocup.RoboCup,
        )
        return gym.make('robocup:RoboCup-collect-v0')

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_fn', create_robocup)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 1000)
    eg.add('steps_per_epoch', 80000)
    eg.add('ac_kwargs:hidden_sizes', [(32,)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)
