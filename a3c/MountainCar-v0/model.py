import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 10)
        self.fc2 = nn.Linear(10, 5)

        self.critic_linear = nn.Linear(5, 1)
        self.actor_linear = nn.Linear(5, action_dim)

        # self.train()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.critic_linear(x), self.actor_linear(x)