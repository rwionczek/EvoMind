from collections import deque

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from gpt import GPTModel, block_size, device
from memory import calculate_last_novelty

memory_size = 2 ** 12

training = True
# training = False

# environment = gymnasium.make('LunarLander-v3', render_mode='rgb_array' if training else 'human')
environment = gymnasium.make('CartPole-v1', render_mode='rgb_array' if training else 'human')

memory_chunk_length = environment.observation_space.shape[0] + environment.action_space.n + 2
memory = torch.zeros(memory_size, memory_chunk_length)
memory_values = torch.zeros(memory_size)
memory_novelties = torch.zeros(memory_size)
memory_novelty_probabilities = torch.zeros(memory_size)

model = GPTModel(environment.observation_space.shape[0], environment.action_space.n).to(device)

model_storage = 'model.pth'
if not training:
    model.load_state_dict(torch.load(model_storage))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

learn_batch_size = 128

# observation_space_low = environment.observation_space.low
# observation_space_high = environment.observation_space.high


observation_space_low = np.array([-4.8, -5.0, -0.418, -5.0])
observation_space_high = np.array([4.8, 5.0, 0.418, 5.0])


def normalize_observation(observation):
    return 2.0 * (observation - observation_space_low) / (observation_space_high - observation_space_low) - 1.0


def get_memory_train_batch():
    wakeup_ix = torch.where(memory[block_size:-1, -1] == 1.0)[0]

    min_return_to_go = memory_values[wakeup_ix].mean()
    possible_ix = torch.where(memory_values[block_size:-1] >= min_return_to_go)[0]

    possible_ix = np.intersect1d(possible_ix, wakeup_ix)

    possible_ix = possible_ix + block_size

    possible_memory_values = memory_values[possible_ix] * memory_novelty_probabilities[possible_ix]

    probabilities = torch.softmax(possible_memory_values / 4.0, dim=0)
    ix = torch.multinomial(probabilities, learn_batch_size, replacement=True)
    #
    # wakeup_ix = torch.where(memory[block_size:-1, -1] == 1.0)[0]
    #
    # min_return_to_go = memory_values[wakeup_ix].mean()
    #
    # possible_ix = torch.where(memory_values[block_size:-1] >= min_return_to_go)[0]
    #
    # possible_ix = np.intersect1d(possible_ix, wakeup_ix)
    #
    # possible_ix = possible_ix + block_size
    #
    # ix = torch.randint(len(possible_ix), (learn_batch_size,))

    x = torch.stack(
        [memory[possible_ix[i] - block_size + 1:possible_ix[i] + 1]
         for
         i in
         ix])
    x[:, -1, environment.observation_space.shape[0]:] = 0

    y = torch.stack(
        [memory[possible_ix[i] - block_size + 2:possible_ix[i] + 2]
         for
         i in
         ix])

    x, y = x.to(device), y.to(device)
    return x, y


eval_iters = 10

episode_values = deque(maxlen=200)
average_episode_values = deque(maxlen=200)
average_trajectory_values = deque(maxlen=200)

first_action_probabilities = deque(maxlen=1000)
second_action_probabilities = deque(maxlen=1000)
third_action_probabilities = deque(maxlen=1000)
fourth_action_probabilities = deque(maxlen=1000)

for episode in range(1, 10000):
    model.eval()
    observation, info = environment.reset()

    memory = torch.roll(memory, -block_size, dims=0)
    memory[-block_size:] = torch.cat([
        torch.tensor(normalize_observation(observation), dtype=torch.float32),
        torch.zeros(environment.action_space.n + 2),
    ])

    total_reward = 0

    while True:

        with torch.no_grad():
            memory = torch.roll(memory, -1, dims=0)
            memory[-1] = torch.cat([
                torch.tensor(normalize_observation(observation), dtype=torch.float32),
                torch.zeros(environment.action_space.n + 2),
            ])

            previous_observation = observation
            prediction, _ = model(memory[-block_size:].unsqueeze(0).to(device))

            action_prediction = prediction[:, -2, environment.observation_space.shape[0]:
                                                  environment.observation_space.shape[
                                                      0] + environment.action_space.n]
            action_probabilities = torch.softmax(action_prediction, dim=-1)

            first_action_probabilities.append(action_prediction[0, 0].item())
            second_action_probabilities.append(action_prediction[0, 1].item())
            # third_action_probabilities.append(action_prediction[0, 2].item())
            # fourth_action_probabilities.append(action_prediction[0, 3].item())

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
            torch.tensor([normalized_reward, 1.0], dtype=torch.float32),
        ])

        memory_novelties = torch.roll(memory_novelties, -1, dims=0)
        memory_novelties[-1] = calculate_last_novelty(memory, block_size)

        total_reward += normalized_reward

        if terminated or truncated:
            break

    if not training:
        continue

    future_reward = 0.0

    for idx in reversed(range(memory_size)):
        reward = memory[idx, -2].item()
        wakeup = memory[idx, -1].item()

        future_reward = reward + 1.0 * future_reward if wakeup != 0 else 0.0

        memory_values[idx] = future_reward

    episode_values.append(total_reward)
    average_episode_value = np.mean(episode_values)
    average_episode_values.append(average_episode_value)

    if episode % 10 == 0:
        memory_novelty_probabilities = torch.softmax(memory_novelties, dim=0)

        losses = torch.zeros(eval_iters)
        for iter in range(300):
            if iter % eval_iters == 0:
                print(f"step: {iter}, loss: {losses.mean():.4f}")

            xb, yb, = get_memory_train_batch()

            model.train()

            logits, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses[iter % eval_iters] = loss

            model.eval()

    if episode % 20 == 0:
        plt.plot(episode_values, label='Episode values')
        plt.plot(average_episode_values, label='Average episode values')
        plt.plot(average_trajectory_values, label='Average trajectory values')
        # plt.plot(max_trajectory_values, label='Max trajectory values')

        plt.xlabel('Episode')
        plt.ylabel('Trajectory values')
        plt.legend()
        plt.show()
        plt.figure(figsize=(8, 5))
        plt.hist(first_action_probabilities, bins=20, color='skyblue', edgecolor='black')
        plt.hist(second_action_probabilities, bins=20, color='orange', edgecolor='black')
        plt.hist(third_action_probabilities, bins=20, color='green', edgecolor='black')
        plt.hist(fourth_action_probabilities, bins=20, color='red', edgecolor='black')
        plt.xlabel('Episode Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Episode Values')
        plt.grid(True, alpha=0.3)
        plt.show()

    if episode % 100 == 0:
        torch.save(model.state_dict(), model_storage)
        print(f"Saved model at episode {episode}")

environment.close()
