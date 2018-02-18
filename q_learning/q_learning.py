# coding=utf-8
import random
import math


class QLearning(object):
    def __init__(self, action_set, terminal_state, arbitrarily, alpha=0.2, gamma=0.9):
        self.action_set = action_set
        self.arbitrarily = arbitrarily
        self.data = {}
        for a in action_set:
            self._set_q(terminal_state, a, 0.0)

        self.alpha = alpha
        self.gamma = gamma

        self.k = 0

    def _set_q(self, a, s, value):
        self.data['{}_{}'.format(s, a)] = value

    def _get_q(self, a, s):
        return self.data.get('{}_{}'.format(s, a), self.arbitrarily)

    def _calc_epsilon(self):
        epsilon = 1.0 / math.sqrt(self.k * 1.0)
        if self.k % 1000 == 0:
            print('[debug] [epsilon] k={}, epsilon={:.4f}'.format(self.k, epsilon))
        return epsilon

    def _e_greedy(self, state):
        if random.random() <= self._calc_epsilon():
            return random.choice(self.action_set)
        else:
            action, _ = self._greedy(state)
            return action

    def _greedy(self, state):
        max_a = self.action_set[0]
        max_q = self._get_q(state, max_a)
        for i in range(1, len(self.action_set)):
            a = self.action_set[i]
            q = self._get_q(state, a)
            if q > max_q:
                max_q = q
                max_a = a
        return max_a, max_q

    def policy(self, state):
        self.k += 1
        return self._e_greedy(state)

    def update(self, s, a, r, ss):
        q = self._get_q(s, a)
        _, mq = self._greedy(ss)
        self._set_q(s, a, q + self.alpha * (r + self.gamma * mq - q))

