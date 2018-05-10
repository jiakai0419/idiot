from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_discrete_env
from model import ActorCritic
# from test import test
from train import train

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--seed', type=int, default=1103515245,
                    help='random seed (default: 1103515245)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training processes to use (default: 16)')
parser.add_argument('--t-max', type=int, default=100000,
                    help='maximum length of an episode (default: 100000)')
parser.add_argument('--env-name', default='MountainCar-v0',
                    help='environment to train on (default: MountainCar-v0)')
parser.add_argument('--episode-num-per-proc', type=int, default=2000)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = create_discrete_env(args.env_name)
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []

    T = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, T))
    # p.start()
    # processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, T, lock, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
