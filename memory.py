import torch


def calculate_last_novelty(memory, block_size):
    trajectories = memory.unfold(0, block_size, 1)[:-1]
    last_trajectory = memory[-block_size:]

    batch_size = trajectories.size(0)

    flatten_trajectories = trajectories.reshape(batch_size, -1)
    flatten_last_trajectory = last_trajectory.flatten()

    distances = torch.cdist(flatten_trajectories, flatten_last_trajectory.unsqueeze(0))

    return distances.mean()
