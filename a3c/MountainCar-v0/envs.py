import gym
import gym.spaces


def create_discrete_env(env_id):
    env = gym.make(env_id)
    return env.unwrapped

