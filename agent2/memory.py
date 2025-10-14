import numpy as np


class Memory:
    def __init__(self, size, observation_space_size, action_space_size):
        self.size = size

        self.states = np.zeros((size,) + observation_space_size)
        self.actions = np.zeros((size,) + action_space_size)
        self.rewards = np.zeros(size)
        self.values = np.zeros(size)
        self.dones = np.ones(size)
        self.train_mask = np.ones(size)

    def append_step(self, state, action, reward, done):
        self.states = np.roll(self.states, -1, axis=0)
        self.actions = np.roll(self.actions, -1, axis=0)
        self.rewards = np.roll(self.rewards, -1, axis=0)
        self.dones = np.roll(self.dones, -1, axis=0)

        self.states[-1] = state
        self.actions[-1] = action
        self.rewards[-1] = reward
        self.dones[-1] = done

    def recalculate_values(self):
        value = 0.0

        for idx in reversed(range(self.size)):
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
