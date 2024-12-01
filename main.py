import random
import gymnasium
import numpy
import torch

from evomind import Agent

environment = gymnasium.make('Acrobot-v1')
agent = Agent()

observation, info = environment.reset()

episode_over = False
while not episode_over:
    agent.set_observation(observation)
    observation, reward, terminated, truncated, info = environment.step(agent.choose_action())
    agent.set_reward(reward)

    episode_over = terminated or truncated

print(agent.memory)

environment.close()
