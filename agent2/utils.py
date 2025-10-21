import numpy as np


def calculate_progress(episode_rewards):
    if len(episode_rewards) < 2:
        return 0.0

    y = np.array(episode_rewards)
    x = np.arange(len(y))

    b, a = np.polyfit(x, y, 1)

    return b
