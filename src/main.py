import random
import gymnasium as gym
import numpy as np

import tensorflow

from keras import __version__
tensorflow.keras.__version__ = __version__

from tensorflow.keras.models import Se

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1", render_mode="human")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Flatten(input_shape=(1, states)))
model.add(tensorflow.keras.layers.Dense(24, activation="relu"))
model.add(tensorflow.keras.layers.Dense(24, activation="relu"))
model.add(tensorflow.keras.layers.Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01,
)

agent.compile(tensorflow.keras.optimizers.Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()
