import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
from pg import PolicyGradient, MAGIC_NUMBER

EPISODE = 5000
TEST_EPISODE = 10

performance_line = []

FAST_FLAG = False

### customized for env begin ###
GOAL_POSITION = 0.5


def _calc_distance(state):
    position = state[0]
    return abs(position - GOAL_POSITION)


def _calc_progressive_reward(state, next_state):
    # return _calc_distance(state) - _calc_distance(next_state)
    return - _calc_distance(next_state)


### customized for env end ###


def visual_performance(env, agent):
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in performance_line], [p[1] for p in performance_line])
    ax.set_xlabel('episode_num')
    ax.set_ylabel('avg_reward')
    ax.set_title('performance')
    fig.savefig('{}_performance_alpha_{}_gamma_{}.png'.format(env.spec.id, agent.alpha, agent.gamma))


def main():
    # https://github.com/openai/gym/wiki/MountainCar-v0
    env = gym.make('MountainCar-v0')
    env.seed(MAGIC_NUMBER)
    agent = PolicyGradient(state_dim=env.observation_space.shape[0],
                           action_dim=env.action_space.n,
                           alpha=0.05,
                           gamma=0.97)

    for episode in xrange(1, EPISODE + 1):
        state = env.reset()
        while True:
            action = agent.policy(state)
            next_state, _, done, _ = env.step(action)
            reward = _calc_progressive_reward(state, next_state)
            agent.store_one_step_mdp(state, action, reward)
            if done:
                break
            state = next_state
        agent.update()

        if episode % 500 == 0:
            test_performance(episode, env, agent)

    visual_performance(env, agent)


def test_performance(trained_episode, env, agent):
    total_reward = 0
    for episode in xrange(1, TEST_EPISODE + 1):
        state = env.reset()
        if not FAST_FLAG:
            env.render()
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            if not FAST_FLAG:
                env.render()
            total_reward += reward
            if done:
                break
            state = next_state
    avg_reward = total_reward * 1.0 / TEST_EPISODE
    performance_line.append((trained_episode, avg_reward))
    print 'episode:{}, avg_reward:{}'.format(trained_episode, avg_reward)


if __name__ == '__main__':
    main()
