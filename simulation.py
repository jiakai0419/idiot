# coding=utf-8

from gridworld import DijkstraWorld
from q_learning import QLearning


if __name__ == '__main__':
    env = DijkstraWorld()
    agent = QLearning([0, 1, 2, 3], 58)
    episode_num = 300
    for i in range(episode_num):
        env.reset()
        env.render()
        if i == 0:
            input("press any key to launch...")
        step_count = 0
        s = 49
        while True:
            step_count += 1
            a = agent.e_greedy(s)
            ss, reward, done, info = env.step(a)
            # print('{} {} {} {}'.format(ss, reward, done, info))
            agent.update(s, a, reward, ss)
            s = ss
            env.render()
            if done:
                print("{}th episode cost {} step".format(i+1, step_count))
                break
    # for k, v in agent.get_data().items():
    #     print(k, v)
    input("press any key to quit...")
