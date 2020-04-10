from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch
import gym

def run_experiment(args):
    def env_fn():
        import robocup_env  # registers custom envs to gym env registry
        return gym.make(args.env_name)

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', env_fn)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--env_name', type=str, default="CustomEnv-v0")
    parser.add_argument('--exp_name', type=str, default='ddpg-custom')
    args = parser.parse_args()

    run_experiment(args)
