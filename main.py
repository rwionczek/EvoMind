from collections import deque

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from gpt import GPTModel, block_size, device
from memory import calculate_memory_batch_probabilities

memory_size = 2 ** 14

training = True
training = False

environment = gymnasium.make('LunarLander-v3', render_mode='rgb_array' if training else 'human')
# environment = gymnasium.make('CartPole-v1', render_mode='rgb_array' if training else 'human')

memory_chunk_length = environment.observation_space.shape[0] + environment.action_space.n + 1
memory = torch.zeros(memory_size, memory_chunk_length)
memory_rewards = torch.zeros(memory_size)
memory_actives = torch.zeros(memory_size)
memory_batch_possible_ix = torch.zeros(memory_size)
memory_batch_probabilities = torch.zeros(memory_size)

model = GPTModel(environment.observation_space.shape[0], environment.action_space.n).to(device)

model_storage = 'model.pth'
if not training:
    model.load_state_dict(torch.load(model_storage))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

learn_batch_size = 2 ** 10

observation_space_low = environment.observation_space.low
observation_space_high = environment.observation_space.high


# observation_space_low = np.array([-4.8, -5.0, -0.418, -5.0])
# observation_space_high = np.array([4.8, 5.0, 0.418, 5.0])


def normalize_observation(observation):
    return 2.0 * (observation - observation_space_low) / (observation_space_high - observation_space_low) - 1.0


def get_memory_train_batch():
    possible_ix = memory_batch_possible_ix
    # ix = torch.multinomial(memory_batch_probabilities, learn_batch_size, replacement=True)

    ix = torch.randint(0, possible_ix.shape[0], (learn_batch_size,))

    x = torch.stack(
        [memory[possible_ix[i] - block_size + 1:possible_ix[i] + 1]
         for
         i in
         ix])
    x[:, :, -1] = 0.0

    y = torch.stack(
        [memory[possible_ix[i] - block_size + 2:possible_ix[i] + 2]
         for
         i in
         ix])

    y[:, :-1, -1] = 0.0

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

actions = [0, 0, 0, 0]

for episode in range(1, 10000):
    model.eval()
    observation, info = environment.reset()
    reward = 0

    memory = torch.roll(memory, -block_size, dims=0)
    memory[-block_size:] = torch.cat([
        torch.tensor(normalize_observation(observation), dtype=torch.float32),
        torch.zeros(environment.action_space.n + 1),
    ])

    memory_rewards = torch.roll(memory_rewards, -block_size, dims=0)
    memory_rewards[-block_size:] = 0.0

    memory_actives = torch.roll(memory_actives, -block_size, dims=0)
    memory_actives[-block_size:] = 0.0

    total_reward = 0

    while True:
        with torch.no_grad():
            memory = torch.roll(memory, -1, dims=0)
            memory[-1] = torch.cat([
                torch.tensor(normalize_observation(observation), dtype=torch.float32),
                torch.zeros(environment.action_space.n + 1),
            ])

            trajectories = memory[-block_size:].unsqueeze(0).repeat(environment.action_space.n, 1, 1)
            trajectories[:, :, -1] = 0.0

            for i in range(trajectories.shape[0]):
                trajectories[i, -1, environment.observation_space.shape[0] + i] = 1.0

            previous_observation = observation
            previous_reward = reward
            prediction, _ = model(trajectories.to(device))

            prediction_values = prediction[:, -1, -1]
            action_probabilities = torch.softmax(prediction_values * 10.0, dim=-1)

            first_action_probabilities.append(action_probabilities[0].item())
            second_action_probabilities.append(action_probabilities[1].item())
            third_action_probabilities.append(action_probabilities[2].item())
            fourth_action_probabilities.append(action_probabilities[3].item())

            if training == False or episode % 2 == 0:
                action = action_probabilities.argmax().item()
            else:
                action = action_probabilities.multinomial(num_samples=1).item()

        actions[action] += 1

        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        if abs(float(reward)) > 5:
            print(
                f"reward: {reward}\n")

        total_reward += reward

        action_tensor = torch.zeros(environment.action_space.n)
        action_tensor[action] = 1.0

        memory[-1] = torch.cat([
            torch.tensor(normalize_observation(previous_observation), dtype=torch.float32),
            action_tensor,
            torch.tensor([0.0], dtype=torch.float32)
        ])

        memory_rewards = torch.roll(memory_rewards, -1, dims=0)
        memory_rewards[-1] = torch.tensor([previous_reward], dtype=torch.float32)

        memory_actives = torch.roll(memory_actives, -1, dims=0)
        memory_actives[-1] = torch.tensor([1.0], dtype=torch.float32)

        if terminated or truncated:
            memory = torch.roll(memory, -1, dims=0)
            memory[-1] = torch.cat([
                torch.tensor(normalize_observation(observation), dtype=torch.float32),
                torch.zeros(environment.action_space.n),
                torch.tensor([0.0], dtype=torch.float32)
            ])

            memory_rewards = torch.roll(memory_rewards, -1, dims=0)
            memory_rewards[-1] = torch.tensor([reward], dtype=torch.float32)

            memory_actives = torch.roll(memory_actives, -1, dims=0)
            memory_actives[-1] = 0.0

            break

    if not training:
        continue

    future_reward = 0.0

    for idx in reversed(range(memory_size)):
        reward = memory_rewards[idx].item()
        active = memory_actives[idx].item()

        future_reward = reward + (0.95 * future_reward if active != 0 else 0.0)

        memory[idx, -1] = future_reward

    memory[:, -1] = memory[:, -1] / 100.0

    episode_values.append(total_reward)
    average_episode_value = np.mean(episode_values)
    average_episode_values.append(average_episode_value)

    print(f"Episode: {episode}")

    if episode % 10 == 0:
        memory_batch_possible_ix = calculate_memory_batch_probabilities(
            memory_actives,
            block_size,
        )

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
        plt.xlabel('Episode')
        plt.ylabel('Trajectory values')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        # plt.hist(first_action_probabilities, bins=20, color='skyblue', edgecolor='black')
        # plt.hist(second_action_probabilities, bins=20, color='orange', edgecolor='black')
        # plt.hist(third_action_probabilities, bins=20, color='green', edgecolor='black')
        # plt.hist(fourth_action_probabilities, bins=20, color='red', edgecolor='black')
        plt.bar(np.arange(0, 4), actions, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Values')
        plt.ylabel('Action probability')
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(memory[:, -1], label='Memory return to go')
        plt.plot(memory_actives, 'r.', label='Memory actives')
        plt.xlabel('Memory step')
        plt.ylabel('Memory value')
        plt.legend()
        plt.show()

    if episode % 10 == 0:
        torch.save(model.state_dict(), model_storage)
        print(f"Saved model at episode {episode}")

environment.close()
