import gymnasium as gym
import numpy as np
import tensorflow as tf

gamma = 0.99
max_steps_per_episode = 10000
env = gym.make("CartPole-v1", render_mode="human")
eps = np.finfo(np.float32).eps.item()

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = tf.keras.layers.Input(shape=(num_inputs,))
common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
critic = tf.keras.layers.Dense(1)(common)

model = tf.keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
huber_loss = tf.keras.losses.Huber()
action_props_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:
    state, _ = env.reset()
    episode_reward = 0

    with tf.GradientTape(persistent=True) as tape:
        for timestep in range(1, max_steps_per_episode):
            env.render()

            state = tf.expand_dims(state, 0)

            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_props_history.append(tf.math.log(action_probs[0, action]))

            state, reward, done, _, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        history = zip(action_props_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_props_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
