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


def calculate_memory_batch_probabilities(memory: Tensor, memory_values: Tensor, normalized_memory_novelties,
                                         block_size):
    wakeup_ix = torch.where(memory[block_size:-1, -1] == 1.0)[0]

    # min_return_to_go = memory_values[wakeup_ix].mean()
    # possible_ix = torch.where(memory_values[block_size:-1] >= min_return_to_go)[0]
    #
    # possible_ix = np.intersect1d(possible_ix, wakeup_ix)

    possible_ix = wakeup_ix

    possible_ix = possible_ix + block_size

    possible_memory_values = memory_values[possible_ix] * normalized_memory_novelties[possible_ix]

    return possible_ix, torch.softmax(possible_memory_values / 4.0, dim=0)
