import torch

from agent.memory import Memory
from gpt import GPTModel, device, block_size


class Agent:
    def __init__(self, observation_space_size, action_space_size, reward_space_size):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.reward_space_size = reward_space_size
        self.context_size = block_size
        self.memory = Memory(size=2 ** 14, observation_space_size=observation_space_size,
                             action_space_size=action_space_size,
                             reward_space_size=reward_space_size, context_size=self.context_size)
        self.model = GPTModel(observation_space_size, action_space_size, reward_space_size).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.learn_batch_size = 2 ** 10
        self.exploration_enabled = True

    def choose_action(self, observation):
        context_steps = self.memory.get_context_steps()
        context_steps = torch.roll(context_steps, -1, dims=0)
        context_steps[-1] = torch.cat([
            torch.tensor(observation, dtype=torch.float32),
            torch.zeros(self.action_space_size + self.reward_space_size, dtype=torch.float32),
        ])

        trajectories = context_steps.unsqueeze(0).repeat(self.action_space_size, 1, 1)
        trajectories[:, :, -1] = 0.0

        for i in range(trajectories.shape[0]):
            trajectories[i, -1, self.observation_space_size + i] = 1.0

        prediction, _ = self.model(trajectories.to(device))

        prediction_values = prediction[:, -1, -1]
        action_probabilities = torch.softmax(prediction_values * 10.0, dim=-1)

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
