from collections import deque

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent2.memory import Memory
from agent2.model import CriticNetwork, ActionNetwork
from agent2.utils import calculate_progress

training = True

# environment = gymnasium.make('Pendulum-v1', render_mode='rgb_array' if training else 'human')
# environment = gymnasium.make('LunarLander-v3', continuous=True, render_mode='rgb_array' if training else 'human')
environment = gymnasium.make('BipedalWalker-v3', render_mode='rgb_array' if training else 'human')


def normalize_observation(observation):
    return 2.0 * (observation - environment.observation_space.low) / (
            environment.observation_space.high - environment.observation_space.low) - 1.0


def normalize_action(action):
    return 2.0 * (action - environment.action_space.low) / (
            environment.action_space.high - environment.action_space.low) - 1.0


def denormalize_action(normalized_action):
    return (normalized_action + 1.0) * (
            environment.action_space.high - environment.action_space.low) / 2.0 + environment.action_space.low


def normalize_reward(reward):
    return reward / 100.0


memory = Memory(160000, environment.observation_space.shape, environment.action_space.shape)

action_network = ActionNetwork(
    sum(environment.observation_space.shape),
    128,
    sum(environment.action_space.shape),
)

critic_network = CriticNetwork(
    sum(environment.observation_space.shape) + sum(environment.action_space.shape),
    256,
    1,
)

if not training:
    action_network.load_state_dict(torch.load('action_network.pth'))
    critic_network.load_state_dict(torch.load('critic_network.pth'))

episode_rewards = deque(maxlen=1000)
episode_novelties = deque(maxlen=1000)
episode_progresses = deque(maxlen=1000)
episode_epsilon = deque(maxlen=1000)
critic_loss = deque(maxlen=1000)
debug_actions = [deque(maxlen=2000) for _ in range(sum(environment.action_space.shape))]

epsilon = 0.0

for episode in range(3000):
    print(f"Episode {episode + 1}...")

    observation, info = environment.reset()
    observation = normalize_observation(observation)

    steps = 0
    total_reward = 0.0
    observation_novelties = torch.ones(10)

    while True:
        previous_observation = observation
        observation = torch.tensor(observation, dtype=torch.float32).to(critic_network.device)

        action = action_network(observation.unsqueeze(0))

        for i in range(sum(environment.action_space.shape)):
            debug_actions[i].append(action[0, i].detach().cpu().item())

        if training:
            action += torch.randn_like(action) * epsilon
            action = torch.clamp(action, -1.0, 1.0)

        action = denormalize_action(action.detach().cpu().numpy()).squeeze(0)

        observation, reward, terminated, truncated, info = environment.step(action)
        action = normalize_action(action)
        observation = normalize_observation(observation)
        reward = normalize_reward(reward)
        done = terminated or truncated

        total_reward += reward

        observation_novelties = torch.roll(observation_novelties, -1, dims=0)
        observation_novelties[-1] = torch.tensor(np.sum(np.abs((observation - previous_observation))))

        memory.append_step(previous_observation, action, reward, done)

        steps += 1

        if done:
            break

    episode_rewards.append(total_reward)

    episode_progress = calculate_progress(list(episode_rewards)[-100:]) * 10.0
    episode_progresses.append(episode_progress)

    average_reward = sum(list(episode_rewards)[-100:]) / 100.0

    if average_reward > 1.0:
        epsilon = 0.1
    else:
        epsilon = 0.5

    # epsilon = torch.exp(-(torch.tensor(episode_progress * 20.0).pow(2))).item() * 0.5
    episode_epsilon.append(epsilon)

    print(f"Episode total reward: {total_reward}")
    print(f"Steps: {steps}")

    if not training:
        continue

    if (episode + 1) % 1 == 0:
        print(f"Training...")

        memory.recalculate_values()
        # memory.recalculate_train_mask()

        for _ in range(50):
            observations, actions, values = memory.get_batch(1024)

            observations = torch.tensor(observations, dtype=torch.float32).to(critic_network.device)
            actions = torch.tensor(actions, dtype=torch.float32).to(critic_network.device)
            values = torch.tensor(values, dtype=torch.float32).to(critic_network.device)

            predicted_actions = action_network(observations)

            predicted_values = critic_network(observations, predicted_actions)

            action_network.optimizer.zero_grad()
            loss = -predicted_values.mean()
            loss.backward()
            action_network.optimizer.step()

            predicted_future_rewards = critic_network(observations, actions)

            critic_network.optimizer.zero_grad()
            loss = critic_network.loss(predicted_future_rewards, values.unsqueeze(1))
            loss.backward()
            critic_network.optimizer.step()

            critic_loss.append(loss.item())

    if (episode + 1) % 50 == 0:
        torch.save(action_network.state_dict(), 'action_network.pth')
        torch.save(critic_network.state_dict(), 'critic_network.pth')

        plt.plot(critic_loss, label='Critic loss')
        plt.xlabel('Episode')
        plt.ylabel('Critic loss')
        plt.legend()
        plt.show()

        plt.plot(episode_rewards, label='Episode total rewards')
        plt.plot(episode_progresses, label='Episode progress')
        plt.plot(episode_epsilon, label='Episode epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.legend()
        plt.show()

        plt.hist(debug_actions, bins=20)
        plt.xlabel('Action Values')
        plt.ylabel('Action probability')
        plt.show()
