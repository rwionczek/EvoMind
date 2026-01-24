from collections import deque

import gymnasium
from matplotlib import pyplot as plt

from agent3.agent import Agent

training = True
training = False

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

score_history = []
debug_actions = [deque(maxlen=2000) for _ in range(sum(env.action_space.shape))]
q1_losses = deque(maxlen=2000)
q2_losses = deque(maxlen=2000)
policy_losses = deque(maxlen=2000)
alpha_losses = deque(maxlen=2000)
alpha_values = deque(maxlen=2000)

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    q1_loss = []
    q2_loss = []
    policy_loss = []
    alpha_loss = []
    alpha_value = []

    for step in range(max_steps):
        action = agent.select_action(state)

        scaled_action = action * action_scale

        for i in range(sum(env.action_space.shape)):
            debug_actions[i].append(scaled_action[i])

        next_state, reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        if training:
            train_values = agent.train()

            q1_loss.append(train_values[0])
            q2_loss.append(train_values[1])
            policy_loss.append(train_values[2])
            alpha_loss.append(train_values[3])
            alpha_value.append(train_values[4])

        state = next_state
        episode_reward += reward

        if done:
            break

    score_history.append(episode_reward)

    print(f"Episode {episode}: Total Reward = {episode_reward:.2f}; Steps = {step + 1:04d}")

    if not training:
        continue

    q1_losses.append(sum(q1_loss) / len(q1_loss))
    q2_losses.append(sum(q2_loss) / len(q2_loss))
    policy_losses.append(sum(policy_loss) / len(policy_loss))
    alpha_losses.append(sum(alpha_loss) / len(alpha_loss))
    alpha_values.append(sum(alpha_value) * 10.0 / len(alpha_value))

    if (episode + 1) % 10 == 0:
        agent.save('agent_state')

        plt.hist(debug_actions, bins=20, label='Action values')
        plt.xlabel('Action Values')
        plt.ylabel('Action occurrences')
        plt.legend()
        plt.show()

        plt.plot(q1_losses, label='Q1 loss')
        plt.plot(q2_losses, label='Q2 loss')
        plt.plot(policy_losses, label='Policy loss')
        plt.plot(alpha_losses, label='Alpha loss')
        plt.plot(alpha_values, label='Alpha value')
        plt.plot(score_history, label='Score')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

env.close()
