# coding=utf-8

from gridworld import GridWorldEnv
from q_learning import QLearning

def DijkstraWorld():
    """
    最短路径环境
    """
    world = GridWorldEnv(n_width=12,
                         n_height=8,
                         u_size=60,
                         default_reward=-1.0,
                         default_type=0,
                         windy=False)
    world.start = (1,4)
    world.ends = [(10,4)]
    world.rewards = [(10,4,100.0)]
    world.refresh_setting()
    return world


if __name__ == '__main__':
    env = DijkstraWorld()
    agent = QLearning([0, 1, 2, 3], env.get_end_state(), 10)
    episode_num = 1000
    for i in range(episode_num):
        env.reset()
        env.render()
        if i == 0:
            input("press any key to launch...")
        step_count = 0
        s = env.get_start_state()
        while True:
            step_count += 1
            a = agent.policy(s)
            ss, reward, done, info = env.step(a)
            agent.update(s, a, reward, ss)
            s = ss
            env.render()
            if done:
                print("{}th episode cost {} step".format(i+1, step_count))
                break
    input("press any key to quit...")
