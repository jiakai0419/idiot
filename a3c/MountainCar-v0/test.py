import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_discrete_env
from model import ActorCritic


def test(rank, args, shared_model, T):
    torch.manual_seed(args.seed + rank)

    env = create_discrete_env(args.env_name)
    env.seed(args.seed + rank)

    print('[debug] env_id:{}'.format(args.env_name))
    print('[debug] state_dim:{}'.format(env.observation_space.shape[0]))
    print('[debug] action_dim:{}'.format(env.action_space.n))
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()

    # Sync with the shared model
    model.load_state_dict(shared_model.state_dict())

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    recent_actions = deque(maxlen=100)
    t = 0
    while True:
        t += 1

        value, logit = model.forward(state.unsqueeze(0))
        prob = F.softmax(logit, dim=1)
        action = prob.argmax(1).item()

        state, reward, done, _ = env.step(action)
        reward_sum += reward
        recent_actions.append(action)

        if t >= args.t_max:
            done = True

        if recent_actions.count(recent_actions[0]) == recent_actions.maxlen:
            done = True

        if done:
            print("Time {}, T {}, FPS {:.0f}, episode_reward {}, episode_length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                T.value,
                T.value / (time.time() - start_time),
                reward_sum,
                t))
            time.sleep(45)

            # init for new episode
            t = 0
            recent_actions.clear()
            reward_sum = 0

            state = env.reset()
            model.load_state_dict(shared_model.state_dict())

        state = torch.from_numpy(state).float()
