import numpy as np


class ReplayBuffer:
    def __init__(self, size, observation_space_size, action_space_size):
        self.size = size
        self.index = -1

        self.states = np.zeros((size, observation_space_size))
        self.actions = np.zeros((size, action_space_size))
        self.rewards = np.zeros(size)
        self.next_states = np.zeros((size, observation_space_size))
        self.dones = np.ones(size)
        self.train_mask = np.zeros(size)

    def add(self, state, action, reward, next_state, done):
        self.index = (self.index + 1) % self.size
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.train_mask[self.index] = 1.0

    def sample(self, batch_size):
        allowed_indices = np.where(self.train_mask == 1.0)[0]
        indices = np.random.choice(allowed_indices, batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
