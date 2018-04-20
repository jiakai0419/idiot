import numpy as np
import tensorflow as tf

MAGIC_NUMBER = 1103515245

# reproducible
np.random.seed(MAGIC_NUMBER)
tf.set_random_seed(MAGIC_NUMBER)


def gen_weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape))


def gen_bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


# REINFORCE
class PolicyGradient(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 alpha=0.0001,
                 gamma=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma

        self.states_in_episode = []
        self.actions_in_episode = []
        self.rewards_in_episode = []

        self._create_policy_network()
        self._create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _create_policy_network(self):
        self.states_input = tf.placeholder("float32", [None, self.state_dim])

        hidden_layer_units = 12
        W1 = gen_weight_variable([self.state_dim, hidden_layer_units])
        b1 = gen_bias_variable([1, hidden_layer_units])
        W2 = gen_weight_variable([hidden_layer_units, self.action_dim])
        b2 = gen_bias_variable([1, self.action_dim])

        A = tf.nn.relu(tf.matmul(self.states_input, W1) + b1)
        self.P = tf.nn.softmax(tf.matmul(A, W2) + b2)

    def _create_training_method(self):
        self.actions_input = tf.placeholder("int32", [None, ])
        self.values_input = tf.placeholder("float32", [None, ])

        j = tf.reduce_mean(
            tf.reduce_sum(
                tf.log(self.P) * tf.one_hot(self.actions_input, self.action_dim), axis=1
            ) * self.values_input
        )
        self.loss = tf.negative(j)

        self.optimzer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)

    def policy(self, state):
        p = self.P.eval(feed_dict={
            self.states_input: state[np.newaxis, :]
        })[0]
        return np.random.choice(range(self.action_dim), p=p)

    def store_one_step_mdp(self, s, a, r):
        self.states_in_episode.append(s)
        self.actions_in_episode.append(a)
        self.rewards_in_episode.append(r)

    def update(self):
        self.optimzer.run(feed_dict={
            self.states_input: np.array(self.states_in_episode),
            self.actions_input: np.array(self.actions_in_episode),
            self.values_input:self._calc_v()
        })

        self.states_in_episode = []
        self.actions_in_episode = []
        self.rewards_in_episode = []

    def _calc_v(self):
        v = np.zeros_like(self.rewards_in_episode)
        acc = 0
        for t in reversed(range(0, len(self.rewards_in_episode))):
            acc = acc * self.gamma + self.rewards_in_episode[t]
            v[t] = acc

        # normalize
        v -= np.mean(v)
        v /= np.std(v)
        return v
