from collections import deque

import torch


class Policy(torch.Module):
    def __init__(self, input_features):
        self.first_layer = torch.nn.Linear(input_features, 100)
        self.second_layer = torch.nn.Linear(100, 2)

    def forward(self, input):
        pass


class MemoryChunk():
    def __init__(self, observation, action, reward, next_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

    def __repr__(self):
        return "\n" + str(self.observation) + "; " + str(self.action) + "; " + str(self.reward)

class Agent:
    def __init__(self):
        self.policy = None
        self.previous_observation = None
        self.observation = None
        self.action = None
        self.memory = deque(maxlen=100)

    def set_observation(self, observation):
        if self.observation is not None:
            self.previous_observation = self.observation

        self.observation = observation

    def choose_action(self):
        self.action = 1

        return self.action

    def set_reward(self, reward):
        self.memory.append(MemoryChunk(self.observation, self.action, reward, self.previous_observation))

    def learn(self, steps):
        pass

    def __initialize_policy(self):
        self.policy = Policy(len(self.observation))

