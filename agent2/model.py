import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = F.relu(self.linear1(state_action))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ActionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActionNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x
