from collections import deque

import gymnasium
import numpy as np
import torch

from memory import Memory, MemorySequence, MemorySequenceChunk

environment = gymnasium.make('CartPole-v1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerActionPredictor(torch.nn.Module):
    def __init__(self, input_dim, n_actions, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = torch.nn.Linear(d_model, n_actions)

    def forward(self, src):
        embedded = self.embedding(src)
        transformer_out = self.transformer(embedded)
        output = self.decoder(transformer_out[-1])
        return output


model = TransformerActionPredictor(input_dim=environment.observation_space.shape[0],
                                   n_actions=environment.action_space.n).to(device)
memory = Memory(10)

total_rewards = []
for episode in range(20):
    observation, info = environment.reset()

    reward = None
    total_reward = 0
    memory_sequence = MemorySequence()
    while True:
        action = environment.action_space.sample()
        previous_observation = observation
        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        if terminated:
            reward = -100

        memory_sequence.add_chunk(MemorySequenceChunk(previous_observation, action, reward))

        total_reward += reward

        if terminated:
            break

        if truncated:
            break

    memory.add(memory_sequence)

    total_rewards.append(total_reward)

    print('Episode ', episode, '; Total reward %.1f' % total_reward, '; 100 game average %.1f' % np.mean(total_rewards[-100:]))

environment.close()
