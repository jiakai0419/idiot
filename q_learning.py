# coding=utf-8
import random
import math

class QLearning(object):
    def __init__(self, action_set, terminal_state, alpha=0.3, gamma=0.9, epsilon = 0.4):
        self.action_set = action_set
        self.data = {}
        for a in action_set:
            self.data['{}_{}'.format(terminal_state, a)] = 0.0
        self.d = 10.0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def e_greedy(self, state):
        self.epsilon -= 0.0001
        if math.fabs(random.random()) <= self.epsilon:
            return random.choice(self.action_set)
        else:
            a, _ = self.greedy(state)
            return a

    def greedy(self, state):
        random.shuffle(self.action_set)
        max_action = self.action_set[0]
        max_q = self.data.get('{}_{}'.format(state, max_action), self.d)
        for i in range(1, len(self.action_set)):
            a = self.action_set[i]
            q = self.data.get('{}_{}'.format(state, a), self.d)
            if q > max_q:
                max_q = q
                max_action = a
        return max_action, max_q

    def update(self, s, a, r, ss):
        q = self.data.get('{}_{}'.format(s, a), self.d)
        _, mq = self.greedy(ss)
        self.data['{}_{}'.format(s, a)] = q + self.alpha * (r + self.gamma * mq - q)

    def get_data(self):
        return self.data
