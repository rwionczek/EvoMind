import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from gpt import GPTModel, block_size, device

trajectory_count = 512
trajectory_size = 256

environment = gymnasium.make('CartPole-v1')

trajectory_chunk_length = environment.observation_space.shape[0] + environment.action_space.n + 1
trajectories = torch.zeros(trajectory_count, trajectory_size, trajectory_chunk_length).to(device)
trajectory_values = torch.zeros(trajectory_count).to(device)

model = GPTModel(trajectory_chunk_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

learn_batch_size = 128

observation_boundaries = np.array([4.8, 5.0, 0.418, 5.0])


def normalize_observation(observation):
    return np.clip(observation / observation_boundaries, -1, 1)


def get_memory_batch():
    probabilities = torch.softmax(trajectory_values, dim=0)
    it = torch.multinomial(probabilities, learn_batch_size, replacement=True)
    ix = torch.randint(trajectory_size - block_size, (learn_batch_size,))

    x = torch.stack(
        [trajectories[it[i], ix[i]:ix[i] + block_size] for i in range(learn_batch_size)])
    x[:, -1, environment.observation_space.shape[0]:] = 0

    y = torch.stack(
        [trajectories[it[i], ix[i] + 1:ix[i] + block_size + 1] for i in
         range(learn_batch_size)])

    x, y = x.to(device), y.to(device)
    return x, y


eval_iters = 10


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_memory_batch()
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


average_trajectory_values = []

for episode in range(2000):
    observation, info = environment.reset()

    # reward = None
    total_reward = 0

    trajectory = torch.zeros(trajectory_size, trajectory_chunk_length).to(device)

    while True:
        trajectory = torch.roll(trajectory, -1, dims=0)
        trajectory[-1] = torch.cat([
            torch.tensor(normalize_observation(observation), dtype=torch.float32, device=device),
            torch.zeros(environment.action_space.n + 1, device=device),
        ])
        previous_observation = observation
        with torch.no_grad():
            prediction, _ = model(trajectory[-block_size:].unsqueeze(0))
            action_probabilities = torch.softmax(prediction[:, -1, environment.observation_space.shape[0]:
                                                                   environment.observation_space.shape[
                                                                       0] + environment.action_space.n], dim=-1)
            action = action_probabilities.multinomial(num_samples=1).item()
        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        action_tensor = torch.zeros(environment.action_space.n, device=device)
        action_tensor[action] = 1.0

        trajectory[-1] = torch.cat([
            torch.tensor(normalize_observation(previous_observation), dtype=torch.float32, device=device),
            action_tensor,
            torch.tensor([reward], dtype=torch.float32, device=device)
        ])

        total_reward += reward

        if terminated:
            break

        if truncated:
            break

    trajectories = torch.roll(trajectories, -1, dims=0)
    trajectories[-1] = trajectory

    trajectory_values = torch.roll(trajectory_values, -1, dims=0)
    trajectory_values[-1] = trajectory[:, -1].sum().item()

    average_trajectory_values.append(trajectory_values[trajectory_values != 0].mean().item())

    if (episode + 1) % 10 == 0:
        for iter in range(50):
            if iter % eval_iters == 0:
                losses = estimate_loss()
                print(f"step: {iter}, loss: {losses:.4f}")

            xb, yb = get_memory_batch()

            logits, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    if (episode + 1) % 20 == 0:
        plt.plot(average_trajectory_values, label='Average trajectory values')

        plt.xlabel('Episode')
        plt.ylabel('Average trajectory values')
        plt.legend()
        plt.show()

environment.close()
