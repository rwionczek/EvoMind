import random
from collections import deque

import torch


class Policy(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.first_layer = torch.nn.Linear(input_features, 100)
        self.second_layer = torch.nn.Linear(100, output_features)

    def forward(self, input):
        x = torch.relu(self.first_layer(input))
        x = self.second_layer(x)
        return x


class MemoryChunk:
    def __init__(self, observation, action, reward, next_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

    def __repr__(self):
        return "\n" + str(self.observation) + "; " + str(self.action) + "; " + str(self.reward)

class Agent:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.policy = Policy(observation_space, self.action_space)
        self.previous_observation = None
        self.observation = None
        self.action = None
        self.memory = deque(maxlen=100)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def set_observation(self, observation):
        if self.observation is not None:
            self.previous_observation = self.observation

        self.observation = observation

    def choose_action(self):
        if random.random() < 0.05:
            self.action = random.choice(range(self.action_space))
        else:
            q_values = self.policy(torch.tensor(self.observation, dtype=torch.float32))
            self.action = q_values.argmax().item()

        return self.action

    def set_reward(self, reward):
        self.memory.append(MemoryChunk(self.observation, self.action, reward, self.previous_observation))

    def learn(self):
        batch_size = 100

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        observations = torch.tensor([m.observation for m in batch], dtype=torch.float32)
        actions = torch.tensor([m.action for m in batch], dtype=torch.int64)
        rewards = torch.tensor([m.reward for m in batch], dtype=torch.float32)
        next_observations = torch.tensor([m.next_observation for m in batch], dtype=torch.float32)

        q_values = self.policy(observations).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.policy(next_observations).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values

        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

