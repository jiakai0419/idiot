import math
import random
import numpy as np
import tensorflow as tf

from collections import deque

BATCH_SIZE = 64
REPLAY_SIZE = 8192

GAMMA = 0.9


def gen_weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape))


def gen_bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


class DQN(object):
    def __init__(self, env):
        self.env = env
        self.replay_buffer = deque()
        self.epsilon_k = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self._create_q_network()
        self._create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def policy(self, state, explore=True):
        if explore and random.random() <= self._calc_epsilon():
            return self.env.action_space.sample()
        q = self.Q.eval(feed_dict={
            self.state_input: [state]
        })[0]
        return np.argmax(q)

    def update(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self._training()

    def _calc_epsilon(self):
        self.epsilon_k += 1
        return 1.0 / math.sqrt(self.epsilon_k)

    def _create_q_network(self):
        hidden_layer_dim = 20
        W1 = gen_weight_variable([self.state_dim, hidden_layer_dim])
        b1 = gen_bias_variable([hidden_layer_dim])
        W2 = gen_weight_variable([hidden_layer_dim, self.action_dim])
        b2 = gen_bias_variable([self.action_dim])

        self.state_input = tf.placeholder("float", [None, self.state_dim])
        A = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q = tf.matmul(A, W2) + b2

    def _create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        y_hat = tf.reduce_sum(tf.multiply(self.Q, self.action_input), 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input-y_hat))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.cost)

    def _training(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [record[0] for record in minibatch]
        action_batch = [record[1] for record in minibatch]
        reward_batch = [record[2] for record in minibatch]
        next_state_batch = [record[3] for record in minibatch]
        done_batch = [record[4] for record in minibatch]

        y_batch = []
        next_state_q = self.Q.eval(feed_dict={
            self.state_input: next_state_batch
        })
        for i in xrange(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA*np.max(next_state_q[i]))

        self.optimizer.run(feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })

