import torch
from torch import Tensor


def calculate_last_novelty(memory, block_size):
    trajectories = memory.unfold(0, block_size, 1)[:-1]
    last_trajectory = memory[-block_size:]

    batch_size = trajectories.size(0)

    flatten_trajectories = trajectories.reshape(batch_size, -1)
    flatten_last_trajectory = last_trajectory.flatten()

    distances = torch.cdist(flatten_trajectories, flatten_last_trajectory.unsqueeze(0))

    return distances.mean()


def calculate_memory_batch_probabilities(memory_actives: Tensor,
                                         block_size):
    active_ix = torch.where(memory_actives[block_size:-1] == 1.0)[0]

    possible_ix = active_ix

    possible_ix = possible_ix + block_size

    return possible_ix
