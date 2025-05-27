import gymnasium
import numpy as np
import torch

from gpt import GPTModel

memory_capacity = 128
memory_sequence_length = 100

environment = gymnasium.make('CartPole-v1', render_mode='human')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory_sequence_chunk_length = environment.observation_space.shape[0] + environment.action_space.n + 1
memory = torch.zeros(memory_capacity, memory_sequence_length, memory_sequence_chunk_length).to(device)
memory_values = torch.zeros(memory_capacity, 1).to(device)
memory_sequence = torch.zeros(memory_sequence_length, memory_sequence_chunk_length).to(device)

model = GPTModel(memory_sequence_chunk_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

learn_memory_sequence_length = 8
learn_batch_size = 16


def get_memory_batch():
    im = torch.randint(memory_capacity, (learn_batch_size,))
    ix = torch.randint(memory_sequence_length - learn_memory_sequence_length, (learn_batch_size,))
    x = torch.stack([memory[im[i], ix[i]:ix[i] + learn_memory_sequence_length] for i in range(learn_batch_size)])
    y = torch.stack(
        [memory[im[i], ix[i] + 1:ix[i] + learn_memory_sequence_length + 1] for i in range(learn_batch_size)])
    x, y = x.to(device), y.to(device)
    return x, y


eval_iters = 10


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_memory_batch()
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


total_rewards = []
for episode in range(1000):
    observation, info = environment.reset()

    # reward = None
    total_reward = 0

    while True:

        with torch.no_grad():
            prediction, _ = model(memory_sequence.unsqueeze(0))
            action_probabilities = torch.softmax(prediction[:, -1, environment.observation_space.shape[0]:
                                                                   environment.observation_space.shape[
                                                                       0] + environment.action_space.n], dim=-1)
            action = action_probabilities.multinomial(num_samples=1).item()
        previous_observation = observation
        observation, reward, terminated, truncated, info = environment.step(
            action
        )

        if terminated:
            reward = -10

        action_tensor = torch.zeros(environment.action_space.n, device=device)
        action_tensor[action] = 1.0

        new_sequence = torch.cat([
            torch.tensor(observation, dtype=torch.float32, device=device),
            action_tensor,
            torch.tensor([reward], dtype=torch.float32, device=device)
        ])

        memory_sequence = torch.roll(memory_sequence, -1, dims=0)
        memory_sequence[-1] = new_sequence

        memory_sequence_value = memory_sequence[:, -1].sum()

        lower_value_indices = torch.where(memory_values <= memory_sequence_value)[0]
        if len(lower_value_indices) > 0:
            random_index = lower_value_indices[torch.randint(0, len(lower_value_indices), (1,))]
            memory[random_index] = memory_sequence
            memory_values[random_index] = memory_sequence_value

        total_reward += reward

        if terminated:
            break

        if truncated:
            break

    memory_rewards = memory[:, :, -1]
    memory_value = memory_rewards.sum()
    print('Memory value: %.1f' % memory_value.item())

    if (episode + 1) % 10 == 0:
        for iter in range(300):
            if iter % eval_iters == 0:
                losses = estimate_loss()
                print(f"step: {iter}, loss: {losses:.4f}")

            xb, yb = get_memory_batch()

            logits, loss = model.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    total_rewards.append(total_reward)

    print('Episode ', episode, '; Total reward %.1f' % total_reward,
          '; 100 game average %.1f' % np.mean(total_rewards[-100:]))

environment.close()
