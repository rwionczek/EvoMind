import gymnasium
import matplotlib.pyplot as plt

from evomind import Agent

environment = gymnasium.make('Acrobot-v1', render_mode='human')
agent = Agent(environment.observation_space.shape[0], environment.action_space.n)

total_rewards = []
for episode in range(10):
    observation, info = environment.reset()
    agent.set_observation(observation, None)

    reward = None
    total_reward = 0
    while True:
        observation, reward, terminated, truncated, info = environment.step(agent.choose_action())
        agent.set_observation(observation, reward)
        agent.learn(print_loss=False)

        total_reward += reward

        if terminated:
            break

        if truncated:
            break

    total_rewards.append(total_reward)

    agent.set_exploration_probability(agent.exploration_probability * 0.7)

    print(f'Exploration probability: {agent.exploration_probability}')
    print(f'Episode {episode}')

environment.close()

environment = gymnasium.make('Acrobot-v1', render_mode='human')
observation, info = environment.reset()

agent.set_exploration_probability(0.0)

reward = None
while True:
    agent.set_observation(observation, reward)
    observation, reward, terminated, truncated, info = environment.step(agent.choose_action())

    if terminated or truncated:
        break
environment.close()

plt.plot(total_rewards)
plt.title('Total Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
