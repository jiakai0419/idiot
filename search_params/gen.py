from __future__ import print_function
import numpy as np

N = 5


def universal():
    for _ in range(N):
        # print(np.random.uniform(low=0.1, high=1.0))
        print(np.random.uniform(low=0.01, high=0.1))
        print(np.random.uniform(low=0.001, high=0.01))
        print(np.random.uniform(low=0.0001, high=0.001))
        print(np.random.uniform(low=0.00001, high=0.0001))
        print(np.random.uniform(low=0.000001, high=0.00001))
        # print(np.random.uniform(low=0.0000001, high=0.000001))
        # print(np.random.uniform(low=0.00000001, high=0.0000001))


def mcpg_mountaincarv0():
    for _ in range(N):
        # print(np.random.uniform(low=0.1, high=1.0))
        print(np.random.uniform(low=0.01, high=0.1))
        print(np.random.uniform(low=0.001, high=0.01))
        print(np.random.uniform(low=0.0001, high=0.001))
        # print(np.random.uniform(low=0.00001, high=0.0001))
        # print(np.random.uniform(low=0.000001, high=0.00001))
        # print(np.random.uniform(low=0.0000001, high=0.000001))
        # print(np.random.uniform(low=0.00000001, high=0.0000001))


def a3c_mountaincarv0():
    for _ in range(N):
        # print(np.random.uniform(low=0.1, high=1.0))
        print(np.random.uniform(low=0.01, high=0.1))
        print(np.random.uniform(low=0.001, high=0.01))
        print(np.random.uniform(low=0.0001, high=0.001))
        # print(np.random.uniform(low=0.00001, high=0.0001))
        # print(np.random.uniform(low=0.000001, high=0.00001))
        # print(np.random.uniform(low=0.0000001, high=0.000001))
        # print(np.random.uniform(low=0.00000001, high=0.0000001))


def a2c_mountaincarv0():
    for _ in range(N):
        # print(np.random.uniform(low=0.1, high=1.0))
        print(np.random.uniform(low=0.01, high=0.1))
        print(np.random.uniform(low=0.001, high=0.01))
        print(np.random.uniform(low=0.0001, high=0.001))
        # print(np.random.uniform(low=0.00001, high=0.0001))
        # print(np.random.uniform(low=0.000001, high=0.00001))
        # print(np.random.uniform(low=0.0000001, high=0.000001))
        # print(np.random.uniform(low=0.00000001, high=0.0000001))


if __name__ == '__main__':
    a2c_mountaincarv0()
