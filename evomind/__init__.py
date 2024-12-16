import random
from collections import deque

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.first_layer = torch.nn.Linear(input_features, 64)
        self.second_layer = torch.nn.Linear(64, 64)
        self.third_layer = torch.nn.Linear(64, output_features)

    def forward(self, input):
        x = torch.relu(self.first_layer(input))
        x = torch.relu(self.second_layer(x))
        x = self.third_layer(x)
        return x


class MemoryChunk:
    def __init__(self, observation, action, reward, next_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

    def __repr__(self):
        return "\n" + str(self.observation) + "; " + str(self.action) + "; " + str(self.reward) + "; " + str(self.next_observation)

class Agent:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.policy_network = Policy(observation_space, self.action_space).to(device)
        self.target_network = Policy(observation_space, self.action_space).to(device)
        self.previous_observation = None
        self.observation = None
        self.action = None
        self.memory = deque(maxlen=10000)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-3)
        self.exploration_probability = 1.0

        self.target_network.load_state_dict(self.policy_network.state_dict())

    def set_observation(self, observation, reward):
        if reward is not None:
            self.memory.append(MemoryChunk(self.observation, self.action, reward, observation))

        self.observation = observation

    def choose_action(self):
        if random.random() <= self.exploration_probability:
            self.action = random.choice(range(self.action_space))
        else:
            with torch.no_grad():
                q_values = self.policy_network(torch.tensor(self.observation, dtype=torch.float32, device=device))
                self.action = q_values.argmax().item()

        return self.action

    def learn(self, print_loss=False):
        batch_size = 64

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        observations = torch.tensor([m.observation for m in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([m.action for m in batch], dtype=torch.int64, device=device)
        rewards = torch.tensor([m.reward for m in batch], dtype=torch.float32, device=device)
        next_observations = torch.tensor([m.next_observation for m in batch], dtype=torch.float32, device=device)

        q_values = self.policy_network(observations).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if print_loss:
            print(f'Loss: {loss.item()}')

        self._soft_update_target_network()

    def set_exploration_probability(self, probability):
        self.exploration_probability = probability

    def _soft_update_target_network(self):
        TAU = 0.05
        target_network_state_dict = self.target_network.state_dict()
        policy_network_state_dict = self.policy_network.state_dict()

        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key] * TAU + target_network_state_dict[key]  * (1 - TAU)

        self.target_network.load_state_dict(target_network_state_dict)

