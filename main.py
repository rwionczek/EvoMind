from collections import deque

import gymnasium
import numpy as np
import torch
from matplotlib import pyplot as plt

from agent import Agent

training = True
# training = False

environment = gymnasium.make('LunarLander-v3', render_mode='rgb_array' if training else 'human')
# environment = gymnasium.make('CartPole-v1', render_mode='rgb_array' if training else 'human')

agent = Agent(observation_space_size=environment.observation_space.shape[0],
              action_space_size=environment.action_space.n, reward_space_size=1)

model_storage = 'model.pth'
if not training:
    agent.model.load_state_dict(torch.load(model_storage))

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

actions = [0, 0, 0, 0]

for episode in range(1, 10000):
    agent.model.eval()
    observation, info = environment.reset()
    reward = 0

    agent.memory.append_episode_begin_steps(normalize_observation(observation))

    total_reward = 0

    while True:
        with torch.no_grad():
            previous_observation = observation
            previous_reward = reward

            if training == False or episode % 2 == 0:
                agent.disable_exploration()
            else:
                agent.enable_exploration()

            action = agent.choose_action(normalize_observation(observation))

        actions[action] += 1

        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        total_reward += reward

        action_tensor = torch.zeros(environment.action_space.n)
        action_tensor[action] = 1.0

        agent.memory.append_step(normalize_observation(previous_observation), action_tensor, previous_reward)

        if terminated or truncated:
            agent.memory.append_episode_end_step(normalize_observation(observation), reward)
            break

    if not training:
        continue

    agent.memory.recalculate_future_rewards()

    episode_values.append(total_reward)
    average_episode_value = np.mean(episode_values)
    average_episode_values.append(average_episode_value)

    print(f"Episode: {episode}")

    if episode % 10 == 0:
        agent.memory.recalculate_train_mask_indexes()

        losses = torch.zeros(eval_iters)
        for iter in range(300):
            loss = agent.train()
            losses[iter % eval_iters] = loss

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
        plt.plot(agent.memory.steps[:, -1], label='Memory return to go')
        plt.plot(agent.memory.train_mask, 'r.', label='Memory train mask')
        plt.xlabel('Memory step')
        plt.ylabel('Memory value')
        plt.legend()
        plt.show()

    if episode % 10 == 0:
        torch.save(agent.model.state_dict(), model_storage)
        print(f"Saved model at episode {episode}")

environment.close()
