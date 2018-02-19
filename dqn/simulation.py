import gym
from dqn import DQN

EPISODE = 10000
TEST_EPISODE = 10


def main():
    # https://github.com/openai/gym/wiki/CartPole-v0
    env = gym.make('CartPole-v0')
    agent = DQN(env)

    for episode in xrange(EPISODE):
        state = env.reset()
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

        if episode % 100 == 0:
            test_performance(episode, env, agent)


def test_performance(trained_episode, env, agent):
    total_reward = 0
    for episode in xrange(TEST_EPISODE):
        state = env.reset()
        env.render()
        while True:
            action = agent.policy(state, explore=False)
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            if done:
                break
            state = next_state
    print 'episode:{}, avg_reward:{}'.format(trained_episode, total_reward * 1.0 / TEST_EPISODE)


if __name__ == '__main__':
    main()
