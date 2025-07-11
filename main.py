from collections import deque

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from agent.memory import Memory
from gpt import GPTModel, block_size, device

memory_size = 2 ** 14

training = True
# training = False

environment = gymnasium.make('LunarLander-v3', render_mode='rgb_array' if training else 'human')
# environment = gymnasium.make('CartPole-v1', render_mode='rgb_array' if training else 'human')

memory = Memory(size=2 ** 14, observation_space_size=environment.observation_space.shape[0],
                action_space_size=environment.action_space.n, reward_space_size=1, context_size=block_size)

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

    memory.append_episode_begin_steps(normalize_observation(observation))

    total_reward = 0

    while True:
        with torch.no_grad():
            context_steps = memory.get_context_steps()
            context_steps = torch.roll(context_steps, -1, dims=0)
            context_steps[-1] = torch.cat([
                torch.tensor(normalize_observation(observation), dtype=torch.float32),
                torch.zeros(environment.action_space.n + 1),
            ])

            trajectories = context_steps.unsqueeze(0).repeat(environment.action_space.n, 1, 1)
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

        total_reward += reward

        action_tensor = torch.zeros(environment.action_space.n)
        action_tensor[action] = 1.0

        memory.append_step(normalize_observation(previous_observation), action_tensor, previous_reward)

        if terminated or truncated:
            memory.append_episode_end_step(normalize_observation(observation), reward)
            break

    if not training:
        continue

    memory.recalculate_future_rewards()

    episode_values.append(total_reward)
    average_episode_value = np.mean(episode_values)
    average_episode_values.append(average_episode_value)

    print(f"Episode: {episode}")

    if episode % 10 == 0:
        memory.recalculate_train_mask_indexes()

        losses = torch.zeros(eval_iters)
        for iter in range(100):
            xb, yb, = memory.get_train_batch(learn_batch_size)
            xb, yb = xb.to(device), yb.to(device)

            model.train()

            logits, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses[iter % eval_iters] = loss

            model.eval()

            if iter % eval_iters == 0:
                print(f"step: {iter}, loss: {losses.mean():.4f}")

    if episode % 20 == 0:
        plt.plot(episode_values, label='Episode values')
        plt.plot(average_episode_values, label='Average episode values')
        plt.plot(average_trajectory_values, label='Average trajectory values')
        plt.xlabel('Episode')
        plt.ylabel('Trajectory values')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.bar(np.arange(0, 4), actions, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Values')
        plt.ylabel('Action probability')
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(memory.steps[:, -1], label='Memory return to go')
        plt.plot(memory.train_mask, 'r.', label='Memory train mask')
        plt.xlabel('Memory step')
        plt.ylabel('Memory value')
        plt.legend()
        plt.show()

    if episode % 10 == 0:
        torch.save(model.state_dict(), model_storage)
        print(f"Saved model at episode {episode}")

environment.close()
