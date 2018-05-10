import logging

import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_discrete_env
from model import ActorCritic

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

performance_line = []


def register_logger(rank, alpha):
    root_logger = logging.getLogger("root")
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    info_handler = logging.FileHandler("info.{}.proc_{}".format(alpha, rank), 'w')
    info_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(info_handler)
    return root_logger


def visual_performance(alpha, proc_num):
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in performance_line], [p[1] for p in performance_line])
    ax.set_xlabel('episode_num')
    ax.set_ylabel('t')
    ax.set_title('PERFORMANCE last_t:{}'.format(performance_line[-1][1]))
    fig.savefig('performance_alpha_{}_proc-num_{}.png'.format(alpha, proc_num))


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, T, lock, optimizer):
    log = register_logger(rank, args.lr)

    torch.manual_seed(args.seed + rank)

    env = create_discrete_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    for episode in range(args.episode_num_per_proc):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        state = env.reset()
        state = torch.from_numpy(state).float()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        done = False

        for t in range(args.t_max):
            value, logit = model.forward(state.unsqueeze(0))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.item())
            with lock:
                T.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

            state = torch.from_numpy(state).float()

        # calc loss
        R = 0
        if not done:
            with torch.no_grad():
                value, _ = model.forward(state.unsqueeze(0))
                R = value.item()

        policy_loss = 0
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + torch.pow(advantage, 2)
            policy_loss = policy_loss - (log_probs[i] * advantage.detach() + args.entropy_coef * entropies[i])

        optimizer.zero_grad()
        # loss = policy_loss + args.value_loss_coef * value_loss
        loss = (policy_loss + args.value_loss_coef * value_loss) / (t + 1)
        with torch.no_grad():
            if (episode + 1) % 100 == 0:
                log.debug('[training] T:{} rank:{} episdoe_num:{} episode_length:{} loss:{}'.format(
                    T.value, rank, episode + 1, t + 1, loss.item()))
            performance_line.append((episode + 1, t + 1))
        loss.backward()
        ensure_shared_grads(model, shared_model)
        optimizer.step()

    if rank == 0:
        visual_performance(args.lr, args.num_processes)
