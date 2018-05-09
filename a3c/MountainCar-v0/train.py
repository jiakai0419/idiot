import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_discrete_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, T, lock, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_discrete_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    while True:
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
        loss = policy_loss + args.value_loss_coef * value_loss
        with torch.no_grad():
            print('[train debug] T:{} rank:{} episode_length:{} loss:{}'.format(
                T.value, rank, t+1, loss.item()))
        loss.backward()
        ensure_shared_grads(model, shared_model)
        optimizer.step()
