from collections import deque

import torch

from agent.memory import Memory
from agent.model import GPTModel, device, block_size


class Agent:
    def __init__(self, observation_space_size, action_space_size, action_space_continuous, reward_space_size):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.action_space_continuous = action_space_continuous
        self.reward_space_size = reward_space_size
        self.context_size = block_size
        self.memory = Memory(size=2 ** 14, observation_space_size=observation_space_size,
                             action_space_size=action_space_size,
                             reward_space_size=reward_space_size, context_size=self.context_size)
        self.model = GPTModel(observation_space_size, action_space_size, action_space_continuous, reward_space_size).to(
            device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.learn_batch_size = 2 ** 10
        self.exploration_enabled = True
        self.action_values = [
            deque(maxlen=1000) for _ in range(self.action_space_size)
        ]
        self.predicted_reward_values = deque(maxlen=200)

    def choose_action(self, observation, action):
        context_steps = self.memory.get_context_steps().to(device)
        context_steps = torch.roll(context_steps, -1, dims=0)
        context_steps[-1] = torch.cat([
            torch.tensor(observation, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32)
        ])

        if self.action_space_continuous:
            optimized_action = torch.tensor(context_steps[-1,
            self.observation_space_size:self.observation_space_size + self.action_space_size],
                                            requires_grad=True, )
            #
            # optimizer = torch.optim.Adam([optimized_action], lr=3e-4)
            #
            # for _ in range(2):
            #     input = context_steps.clone()
            #
            #     input[-1,
            #     self.observation_space_size:self.observation_space_size + self.action_space_size] = optimized_action
            #
            #     prediction, _ = self.model(input.unsqueeze(0))
            #
            #     predicted_reward = prediction[:, -1, -1]
            #
            #     self.predicted_reward_values.append(predicted_reward.item())
            #
            #     optimizer.zero_grad()
            #     (-predicted_reward).backward()
            #     optimizer.step()
            #
            # for i in range(self.action_space_size):
            #     self.action_values[i].append(optimized_action[i].item())
            #
            # if self.exploration_enabled:
            #     optimized_action = optimized_action + torch.randn_like(
            #         optimized_action) * 0.1

            random_actions = optimized_action.unsqueeze(0).repeat(256, 1)

            random_actions = torch.rand_like(random_actions) * 2.0 - 1.0

            inputs = context_steps.clone().unsqueeze(0).repeat(256, 1, 1)
            inputs[:, -1,
            self.observation_space_size:self.observation_space_size + self.action_space_size] = random_actions

            prediction, _ = self.model(inputs)

            prediction_values = torch.softmax(prediction[:, -1, -1] * 10, dim=-1)

            if self.exploration_enabled:
                best_action = prediction_values.multinomial(num_samples=1).item()
            else:
                best_action = prediction_values.argmax()

            optimized_action = random_actions[best_action]

            for i in range(self.action_space_size):
                self.action_values[i].append(optimized_action[i].item())

            self.predicted_reward_values.append(prediction[best_action, -1, -1].item())

            return optimized_action.detach().cpu()

        trajectories = context_steps.unsqueeze(0).repeat(self.action_space_size, 1, 1)
        trajectories[:, :, -1] = 0.0

        for i in range(trajectories.shape[0]):
            trajectories[i, -1, self.observation_space_size + i] = 1.0

        prediction, _ = self.model(trajectories.to(device))

        prediction_values = prediction[:, -1, -1]
        action_probabilities = torch.softmax(prediction_values * 10.0, dim=-1)

        for i in range(self.action_space_size):
            self.action_values[i].append(action_probabilities[i].item())

        if self.exploration_enabled:
            return action_probabilities.multinomial(num_samples=1).item()

        return action_probabilities.argmax().item()

    def train(self):
        xb, yb, = self.memory.get_train_batch(2 ** 10)
        xb, yb = xb.to(device), yb.to(device)

        self.model.train()

        logits, loss = self.model.forward(xb, yb)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.model.eval()

        return loss

    def enable_exploration(self):
        self.exploration_enabled = True

    def disable_exploration(self):
        self.exploration_enabled = False
