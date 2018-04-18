import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym
from pg import PolicyGradient, MAGIC_NUMBER

EPISODE = 2000
TEST_EPISODE = 10

performance_line = []


def visual_performance(agent):
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in performance_line], [p[1] for p in performance_line])
    ax.set_xlabel('episode_num')
    ax.set_ylabel('avg_reward')
    ax.set_title('performance')
    fig.savefig('performance_alpha_{}_gamma_{}.png'.format(agent.alpha, agent.gamma))


def main():
    # https://github.com/openai/gym/wiki/CartPole-v0
    env = gym.make('CartPole-v0')
    env.seed(MAGIC_NUMBER)
    agent = PolicyGradient(state_dim=env.observation_space.shape[0],
                           action_dim=env.action_space.n)

    for episode in xrange(EPISODE):
        state = env.reset()
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_one_step_mdp(state, action, reward)
            if done:
                break
            state = next_state
        agent.update()

        if episode % 100 == 0:
            test_performance(episode, env, agent)

    visual_performance(agent)


def test_performance(trained_episode, env, agent):
    total_reward = 0
    for episode in xrange(TEST_EPISODE):
        state = env.reset()
        env.render()
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
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
