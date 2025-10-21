import numpy as np


class Memory:
    def __init__(self, size, observation_space_size, action_space_size):
        self.size = size
        self.index = -1

        self.states = np.zeros((size,) + observation_space_size)
        self.actions = np.zeros((size,) + action_space_size)
        self.rewards = np.zeros(size)
        self.values = np.zeros(size)
        self.dones = np.ones(size)
        self.train_mask = np.zeros(size)

    def append_step(self, state, action, reward, done):
        self.index = (self.index + 1) % self.size
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.train_mask[self.index] = 1.0

    def recalculate_values(self):
        value = 0.0

        for idx in range(self.size):
            idx = (self.index - idx) % self.size
            value = self.rewards[idx].item() + (0.99 * value if self.dones[idx].item() != 1.0 else 0.0)

            self.values[idx] = value

    def recalculate_train_mask(self):
        self.train_mask = np.ones(self.size)

        done_indices = np.where(self.dones == 1.0)[0] + 1

        for idx in done_indices:
            start_idx = idx - 0
            if start_idx < 0:
                start_idx = 0
            self.train_mask[start_idx:idx] = 0.0

    def get_batch(self, batch_size):
        allowed_indices = np.where(self.train_mask == 1.0)[0]
        indices = np.random.choice(allowed_indices, batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.values[indices],
        )
