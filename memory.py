import torch


def calculate_novelty(memory, block_size):
    trajectories = memory.unfold(0, block_size, 1)

    batch_size = trajectories.size(0)

    flatten_trajectories = trajectories.reshape(batch_size, -1)

    distance_matrix = flatten_trajectories.unsqueeze(1) - flatten_trajectories.unsqueeze(0)

    distance_matrix = torch.norm(distance_matrix, dim=2)

    mask = ~torch.eye(batch_size, dtype=torch.bool)

    memory_novelty = distance_matrix[mask].reshape(batch_size, -1).mean(dim=1)

    return torch.softmax(torch.cat([
        torch.full((block_size - 1,), memory_novelty.min().item()),
        memory_novelty
    ]), dim=0)
