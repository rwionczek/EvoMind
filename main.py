from collections import deque

import gymnasium
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent import Agent

writer = SummaryWriter()

training = True
# training = False

# env = gymnasium.make('Pendulum-v1', render_mode='rgb_array' if training else 'human')
env = gymnasium.make('BipedalWalker-v3', render_mode='rgb_array' if training else 'human')
# env = gymnasium.make('LunarLander-v3', continuous=True, render_mode='rgb_array' if training else 'human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = env.action_space.high[0]

agent = Agent(state_dim, action_dim)

if not training:
    agent.load('agent_state')

num_episodes = 1000
max_steps = 1000

debug_actions = [deque(maxlen=2000) for _ in range(sum(env.action_space.shape))]

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_extrinsic_reward = 0
    episode_intrinsic_reward = 0

    q1_loss = []
    q2_loss = []
    policy_loss = []
    world_model_loss = []
    alpha_loss = []
    alpha_value = []

    for step in range(max_steps):
        action, predicted_next_state = agent.select_action(state)

        scaled_action = action * action_scale

        for i in range(sum(env.action_space.shape)):
            debug_actions[i].append(scaled_action[i])

        next_state, extrinsic_reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated

        intrinsic_reward = np.mean(np.square(predicted_next_state - next_state))

        reward = extrinsic_reward + intrinsic_reward

        agent.store_transition(state, action, reward, next_state, done)

        if training:
            train_values = agent.train()
            train_world_model_value = agent.train_world_model()

            q1_loss.append(train_values[0])
            q2_loss.append(train_values[1])
            policy_loss.append(train_values[2])
            world_model_loss.append(train_world_model_value)
            alpha_loss.append(train_values[3])
            alpha_value.append(train_values[4])

        state = next_state
        episode_reward += reward
        episode_extrinsic_reward += extrinsic_reward
        episode_intrinsic_reward += intrinsic_reward

        if done:
            break

    print(f"Episode {episode}: Total Reward = {episode_reward:.2f}; Steps = {step + 1:04d}")

    if not training:
        continue

    writer.add_scalar('Loss/Q1', sum(q1_loss) / len(q1_loss), episode)
    writer.add_scalar('Loss/Q2', sum(q2_loss) / len(q2_loss), episode)
    writer.add_scalar('Loss/Policy', sum(policy_loss) / len(policy_loss), episode)
    writer.add_scalar('Loss/WorldModel', sum(world_model_loss) / len(world_model_loss), episode)
    writer.add_scalar('Loss/Alpha', sum(alpha_loss) / len(alpha_loss), episode)

    writer.add_scalar('Score/Total', episode_reward, episode)
    writer.add_scalar('Score/Extrinsic', episode_extrinsic_reward, episode)
    writer.add_scalar('Score/Intrinsic', episode_intrinsic_reward, episode)

    writer.add_scalar('Other/AlphaValue', sum(alpha_value) / len(alpha_value), episode)

    if (episode + 1) % 10 == 0:
        agent.save('agent_state')

writer.close()
env.close()
