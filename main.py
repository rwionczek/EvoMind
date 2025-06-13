from collections import deque

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from gpt import GPTModel, block_size, device

memory_size = 2 ** 12

training = True
# training = False

# environment = gymnasium.make('LunarLander-v3', render_mode='rgb_array' if training else 'human')
environment = gymnasium.make('CartPole-v1', render_mode='rgb_array' if training else 'human')

memory_chunk_length = environment.observation_space.shape[0] + environment.action_space.n + 1
memory = torch.zeros(memory_size, memory_chunk_length)
memory_paddings = torch.zeros(memory_size, dtype=torch.long)
memory_values = torch.zeros(memory_size)
memory_values_probabilities = torch.softmax(memory_values[block_size:], dim=0)

model = GPTModel(environment.observation_space.shape[0], environment.action_space.n).to(device)

model_storage = 'model.pth'
# if not training:
#     model.load_state_dict(torch.load(model_storage))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

learn_batch_size = 256

# observation_space_low = environment.observation_space.low
# observation_space_high = environment.observation_space.high


observation_space_low = np.array([-4.8, -5.0, -0.418, -5.0])
observation_space_high = np.array([4.8, 5.0, 0.418, 5.0])

start_batch = torch.cat([
    torch.zeros(memory_chunk_length),
])


def normalize_observation(observation):
    return 2.0 * (observation - observation_space_low) / (observation_space_high - observation_space_low) - 1.0


def pad_to_block(sequence):
    padded_sequence = torch.full((block_size, memory_chunk_length), 0.0)
    padded_sequence[:sequence.shape[0]] = sequence
    return padded_sequence


def create_padding_mask(padding):
    padding_mask = torch.ones(block_size)
    if padding > 0:
        padding_mask[-padding:] = 0

    return padding_mask


def get_memory_train_batch(min_return_to_go):
    # padded_memory_values = memory_values[block_size:]
    # possible_ix = torch.where((padded_memory_values >= min_return_to_go) & (padded_memory_values != 0.0))[0]
    # possible_ix = possible_ix + block_size - 1
    # ix = torch.randint(len(possible_ix), (learn_batch_size,))
    #
    # probabilities = torch.softmax(padded_memory_values, dim=0)
    # ix = torch.multinomial(probabilities, learn_batch_size, replacement=True)
    # ix = ix + block_size - 1

    possible_ix = torch.where(memory_values[block_size:] >= min_return_to_go)[0]

    possible_ix = possible_ix + block_size - 1

    ix = torch.randint(len(possible_ix), (learn_batch_size,))

    # x = torch.stack(
    #     [memory[i - block_size + 1:i + 1] for i in ix])
    # x[:, -1, environment.observation_space.shape[0]:] = 0
    #
    # y = torch.stack(
    #     [memory[i - block_size + 2:i + 2] for i in
    #      ix])

    x = torch.stack(
        [pad_to_block(memory[possible_ix[i] - block_size + memory_paddings[possible_ix[i] + 1] + 1:possible_ix[i] + 1])
         for
         i in
         ix])
    x[:, -1, environment.observation_space.shape[0]:] = 0

    y = torch.stack(
        [pad_to_block(memory[possible_ix[i] - block_size + memory_paddings[possible_ix[i] + 1] + 2:possible_ix[i] + 2])
         for
         i in
         ix])

    padding = torch.stack(
        [create_padding_mask(memory_paddings[possible_ix[i] + 1]) for i in ix]
    )

    x, y, padding = x.to(device), y.to(device), padding.to(device)
    return x, y, padding


eval_iters = 10

episode_values = deque(maxlen=200)
average_episode_values = deque(maxlen=200)
average_trajectory_values = deque(maxlen=200)

first_action_probabilities = deque(maxlen=1000)
second_action_probabilities = deque(maxlen=1000)

for episode in range(1, 10000):
    model.eval()
    observation, info = environment.reset()

    total_reward = 0
    step = 0

    while True:
        step += 1

        with torch.no_grad():
            memory = torch.roll(memory, -1, dims=0)
            memory[-1] = torch.cat([
                torch.tensor(normalize_observation(observation), dtype=torch.float32),
                torch.zeros(environment.action_space.n + 1),
            ])
            memory_paddings = torch.roll(memory_paddings, -1, dims=0)
            padding = block_size - step if step < block_size else 0
            memory_paddings[-1] = padding
            memory_batch = memory[-block_size + padding:]
            memory_batch = pad_to_block(memory_batch)
            padding_mask = create_padding_mask(padding)

            previous_observation = observation
            prediction, _ = model(memory_batch.unsqueeze(0).to(device),
                                  padding_mask=padding_mask.unsqueeze(0).to(device))

            token_index = -2 - padding if padding != block_size - 1 else 0

            action_prediction = prediction[:, token_index, environment.observation_space.shape[0]:
                                                           environment.observation_space.shape[
                                                               0] + environment.action_space.n]
            action_probabilities = torch.softmax(action_prediction, dim=-1)

            first_action_probabilities.append(action_prediction[0, 0].item())
            second_action_probabilities.append(action_prediction[0, 1].item())

            action = action_probabilities.multinomial(num_samples=1).item()

        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        action_tensor = torch.zeros(environment.action_space.n)
        action_tensor[action] = 1.0

        # normalized_reward = reward / 100.0
        normalized_reward = reward

        memory[-1] = torch.cat([
            torch.tensor(normalize_observation(previous_observation), dtype=torch.float32),
            action_tensor,
            torch.tensor([normalized_reward], dtype=torch.float32)
        ])

        total_reward += normalized_reward

        if terminated or truncated:
            break

    if not training:
        continue

    future_reward = 0.0

    for idx in reversed(range(memory_size)):
        reward = memory[idx, -1].item()

        future_reward = reward + future_reward if reward != 0 else 0.0
        memory_values[idx] = future_reward

    memory_values_probabilities = torch.softmax(memory_values[block_size:], dim=0)

    episode_values.append(total_reward)
    average_episode_value = np.mean(episode_values)
    average_episode_values.append(average_episode_value)

    if episode % 10 == 0:
        losses = torch.zeros(eval_iters)
        for iter in range(300):
            if iter % eval_iters == 0:
                print(f"step: {iter}, loss: {losses.mean():.4f}")

            xb, yb, padding = get_memory_train_batch(average_episode_value)

            model.train()

            logits, loss = model.forward(xb, yb, padding)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses[iter % eval_iters] = loss

            model.eval()

    if episode % 20 == 0:
        # plt.plot(episode_values, label='Episode values')
        # plt.plot(average_episode_values, label='Average episode values')
        # plt.plot(average_trajectory_values, label='Average trajectory values')
        # # plt.plot(max_trajectory_values, label='Max trajectory values')
        #
        # plt.xlabel('Episode')
        # plt.ylabel('Trajectory values')
        # plt.legend()
        # plt.show()
        plt.figure(figsize=(8, 5))
        plt.hist(first_action_probabilities, bins=20, color='skyblue', edgecolor='black')
        plt.hist(second_action_probabilities, bins=20, color='orange', edgecolor='black')
        plt.xlabel('Episode Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Episode Values')
        plt.grid(True, alpha=0.3)
        plt.show()

    if episode % 100 == 0:
        torch.save(model.state_dict(), model_storage)
        print(f"Saved model at episode {episode}")

environment.close()
