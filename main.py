from datetime import datetime

import gymnasium
import numpy as np
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from agent.agent import Agent
from agent.exploration_metrics import StateVisitationTracker

writer = SummaryWriter(
    log_dir='.artifacts/tensorboard/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
)

training = True


# training = False

def create_env(hardcore: bool = False):
    # env = gymnasium.make('Pendulum-v1', render_mode='rgb_array' if training else 'human')
    env = gymnasium.make('BipedalWalker-v3', hardcore=hardcore, render_mode='rgb_array' if training else 'human')
    # env = gymnasium.make('LunarLander-v3', continuous=True, render_mode='rgb_array' if training else 'human')

    if training:
        env = RecordVideo(
            env,
            video_folder=".artifacts/records",
            episode_trigger=lambda x: x % 25 == 0,
            name_prefix="rl-video" + "-" + str(hardcore),
        )

    return env


env = create_env(hardcore=True)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = env.action_space.high[0]

manager_agent = Agent(state_dim, state_dim)
worker_agent = Agent(state_dim + state_dim, action_dim)

state_visitation_tracker = StateVisitationTracker(
    env.observation_space.low,
    env.observation_space.high,
    bins=10,
    dims=[0, 1, 2, 3],
)

if not training:
    manager_agent.load('.artifacts/manager_agent_state')
    worker_agent.load('.artifacts/worker_agent_state')

num_episodes = 2000
max_steps = 1000

step_global = 0

for episode in range(num_episodes):
    # if episode == 800:
    #     env.close()
    #     env = create_env(hardcore=True)

    state, _ = env.reset()
    state_visitation_tracker.update(state)
    episode_reward = 0
    episode_extrinsic_reward = 0
    episode_intrinsic_reward = 0

    q1_loss = []
    q2_loss = []
    policy_loss = []
    world_model_loss = []
    alpha_loss = []
    alpha_value = []

    curiosity_engine_loss = []

    target_state = np.zeros(state_dim)
    manager_reward = 0.0
    k = 10

    for step in range(max_steps):
        if step % k == 0:
            target_state = manager_agent.select_action(state)
            manager_reward = 0.0

        worker_input = np.concat([state, target_state])

        action = worker_agent.select_action(worker_input)

        scaled_action = action * action_scale

        next_state, extrinsic_reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated

        state_visitation_tracker.update(next_state)

        # extrinsic_reward = 0.01 * extrinsic_reward

        worker_reward = -np.linalg.norm(target_state - next_state)
        worker_agent.store_transition(
            worker_input,
            action,
            worker_reward,
            np.concat([next_state, target_state]),
            done,
        )

        manager_reward += extrinsic_reward

        if (step + 1) % k == 0 or done:
            intrinsic_reward = manager_agent.calculate_intrinsic_reward(state, target_state)
            manager_agent.store_transition(
                state,
                target_state,
                manager_reward + intrinsic_reward,
                next_state,
                done,
            )

        if training:
            train_values = worker_agent.train()

            q1_loss.append(train_values[0])
            q2_loss.append(train_values[1])
            policy_loss.append(train_values[2])
            # world_model_loss.append(model_loss)
            alpha_loss.append(train_values[3])
            alpha_value.append(train_values[4])

            if (step + 1) % k == 0:
                manager_agent.train()
                curiosity_engine_loss.append(manager_agent.train_curiosity_engine())

        reward = extrinsic_reward + intrinsic_reward

        state = next_state
        episode_reward += reward
        episode_extrinsic_reward += extrinsic_reward
        episode_intrinsic_reward += intrinsic_reward

        writer.add_scalars(
            'Score',
            {
                'Total': reward,
                # 'LearningProgress': learning_progress,
                'Extrinsic': extrinsic_reward,
                'Intrinsic': intrinsic_reward,
            },
            step_global,
        )

        step_global += 1

        if done:
            break

    print(f"Episode {episode}: Total Reward = {episode_reward:.2f}; Steps = {step + 1:04d}")

    if not training:
        continue

    writer.add_scalar('Loss/Q1', sum(q1_loss) / len(q1_loss), episode)
    writer.add_scalar('Loss/Q2', sum(q2_loss) / len(q2_loss), episode)
    writer.add_scalar('Loss/Policy', sum(policy_loss) / len(policy_loss), episode)
    # writer.add_scalar('Loss/WorldModel', sum(world_model_loss) / len(world_model_loss), episode)
    writer.add_scalar('Loss/Alpha', sum(alpha_loss) / len(alpha_loss), episode)

    writer.add_scalar('Loss/Curiosity', sum(curiosity_engine_loss) / len(curiosity_engine_loss), episode)

    writer.add_scalar('Score/Total', episode_reward, episode)
    writer.add_scalar('Score/Extrinsic', episode_extrinsic_reward, episode)
    writer.add_scalar('Score/Intrinsic', episode_intrinsic_reward, episode)

    writer.add_scalar('Other/AlphaValue', sum(alpha_value) / len(alpha_value), episode)

    writer.add_scalar('Exploration/StateCoverage', state_visitation_tracker.coverage, episode)
    writer.add_scalar('Exploration/UniqueStatesVisited', state_visitation_tracker.unique_states_visited, episode)

    if (episode + 1) % 10 == 0:
        agent.save('.artifacts/agent_state')

writer.close()
env.close()
