import random

class MemorySequenceChunk:
    def __init__(self, observation, action, reward):
        self.observation = observation
        self.action = action
        self.reward = reward

class MemorySequence:
    def __init__(self):
        self.chunks: list[MemorySequenceChunk] = []

    def add_chunk(self, chunk: MemorySequenceChunk):
        self.chunks.append(chunk)

    def get_total_reward(self):
        total_reward = 0
        for chunk in self.chunks:
            total_reward += chunk.reward
        return total_reward

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: list[MemorySequence] = []

    def add(self, sequence: MemorySequence):
        if len(self.memory) == self.capacity:
            min_reward = float('inf')
            min_index = 0
            for i, seq in enumerate(self.memory):
                total_reward = seq.get_total_reward()
                if total_reward < min_reward:
                    min_reward = total_reward
                    min_index = i
            self.memory.pop(min_index)

        self.memory.append(sequence)
        
    def get_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
