import gymnasium

from evomind import Agent

environment = gymnasium.make('Acrobot-v1', max_episode_steps=100000, render_mode='human')
agent = Agent(environment.observation_space.shape[0], environment.action_space.n)

observation, info = environment.reset()

step_count = 0
while True:
    agent.set_observation(observation)
    observation, reward, terminated, truncated, info = environment.step(agent.choose_action())
    agent.set_reward(reward)

    if step_count % 100 == 0:
        agent.learn()

    step_count += 1

    if terminated or truncated:
        break

environment.close()
