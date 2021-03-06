import gym.spaces
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAGIC_NUMBER = 1103515245


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.output = nn.Linear(20, action_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        return self.output(x)


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        return self.output(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='CartPole-v0')
    parser.add_argument('--episode-num', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(MAGIC_NUMBER)

    env = gym.make(args.env_name)
    env.seed(MAGIC_NUMBER)

    actor_net = ActorNet(env.observation_space.shape[0], env.action_space.n)
    critic_net = CriticNet(env.observation_space.shape[0])

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=args.lr)

    for episode in range(args.episode_num):
        s = env.reset()
        s = torch.from_numpy(s).float()
        rewards = 0
        while True:
            logit = actor_net.forward(s.unsqueeze(0))
            prob = F.softmax(logit, dim=1)

            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)

            ss, r, done, _ = env.step(action.item())
            rewards += r
            if not done:
                ss = torch.from_numpy(ss).float()

            td_error = None
            if done:
                td_error = r - critic_net.forward(s.unsqueeze(0))
            else:
                td_error = r + args.gamma * critic_net.forward(ss.unsqueeze(0)) - critic_net.forward(s.unsqueeze(0))
            value_loss = torch.pow(td_error, 2)
            policy_loss = -(log_prob * td_error.detach() + args.entropy_coef * entropy)

            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()

            if done:
                break

            s = ss

        if (episode + 1) % 100 == 0:
            print('episode:{}, rewards:{}'.format(episode + 1, rewards))
