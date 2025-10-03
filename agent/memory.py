import torch


class Memory:
    def __init__(self, size, observation_space_size, action_space_size, reward_space_size, context_size):
        self.size = size
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.reward_space_size = reward_space_size
        self.context_size = context_size
        self.steps = torch.zeros(size, observation_space_size + action_space_size + self.reward_space_size)
        self.rewards = torch.zeros(size)
        self.train_mask = torch.zeros(size)
        self.train_mask_indexes = torch.zeros(size)

    def append_episode_begin_steps(self, first_observation):
        self.steps = torch.roll(self.steps, -self.context_size, dims=0)
        self.steps[-self.context_size:] = torch.cat([
            torch.tensor(first_observation, dtype=torch.float32),
            torch.zeros(self.action_space_size + self.reward_space_size),
        ])

        self.rewards = torch.roll(self.rewards, -self.context_size, dims=0)
        self.rewards[-self.context_size:] = 0.0

        self.train_mask = torch.roll(self.train_mask, -self.context_size, dims=0)
        self.train_mask[-self.context_size:] = 0.0

    def get_context_steps(self):
        return self.steps[-self.context_size:]

    def append_step(self, observation, action, reward):
        self.steps = torch.roll(self.steps, -1, dims=0)
        self.steps[-1] = torch.cat([
            torch.tensor(observation, dtype=torch.float32),
            action,
            torch.tensor([0.0], dtype=torch.float32)
        ])

        self.rewards = torch.roll(self.rewards, -1, dims=0)
        self.rewards[-1] = torch.tensor([reward], dtype=torch.float32)

        self.train_mask = torch.roll(self.train_mask, -1, dims=0)
        self.train_mask[-1] = torch.tensor([1.0], dtype=torch.float32)

    def append_episode_end_step(self, observation, reward):
        self.steps = torch.roll(self.steps, -1, dims=0)
        self.steps[-1] = torch.cat([
            torch.tensor(observation, dtype=torch.float32),
            torch.zeros(self.action_space_size),
            torch.tensor([0.0], dtype=torch.float32)
        ])

        self.rewards = torch.roll(self.rewards, -1, dims=0)
        self.rewards[-1] = torch.tensor([reward], dtype=torch.float32)

        self.train_mask = torch.roll(self.train_mask, -1, dims=0)
        self.train_mask[-1] = 0.0

    def recalculate_future_rewards(self):
        future_reward = 0.0

        for idx in reversed(range(self.size)):
            reward = self.rewards[idx].item()
            active = self.train_mask[idx].item()

            future_reward = reward + (0.99 * future_reward if active != 0 else 0.0)

            self.steps[idx, -1] = future_reward

        self.steps[:, -1] = self.steps[:, -1] / 100.0

    def recalculate_train_mask_indexes(self):
        self.train_mask_indexes = torch.where(self.train_mask[self.context_size:-1] == 1.0)[0]
        self.train_mask_indexes = self.train_mask_indexes + self.context_size

    def get_train_batch(self, batch_size):
        ix = torch.randint(0, self.train_mask_indexes.shape[0], (batch_size,))

        x = torch.stack(
            [self.steps[self.train_mask_indexes[i] - self.context_size + 1:self.train_mask_indexes[i] + 1]
             for
             i in
             ix])
        x[:, :, -1] = 0.0

        y = torch.stack(
            [self.steps[self.train_mask_indexes[i] - self.context_size + 2:self.train_mask_indexes[i] + 2]
             for
             i in
             ix])

        y[:, :-1, -1] = 0.0

        return x, y
