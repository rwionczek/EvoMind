from datetime import datetime

import gymnasium
import numpy as np
from gymnasium.wrappers import RecordVideo, NormalizeObservation
from torch.utils.tensorboard import SummaryWriter

from agent.exploration_metrics import StateVisitationTracker
from agent.manager import Manager
from agent.worker import Worker

writer = SummaryWriter(
    log_dir='.artifacts/tensorboard/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
)

training = True


# training = False

def create_env(hardcore: bool = False):
    # env = gymnasium.make('Pendulum-v1', render_mode='rgb_array' if training else 'human')
    env = gymnasium.make('BipedalWalker-v3', hardcore=hardcore, render_mode='rgb_array' if training else 'human')
    # env = gymnasium.make('LunarLander-v3', continuous=True, render_mode='rgb_array' if training else 'human')

    env = NormalizeObservation(env, epsilon=1e-8)

    if training:
        env = RecordVideo(
            env,
            video_folder=".artifacts/records",
            episode_trigger=lambda x: x % 25 == 0,
            name_prefix="rl-video" + "-" + str(hardcore),
        )

    return env


env = create_env(hardcore=False)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = env.action_space.high[0]

manager_agent = Manager(state_dim, state_dim)
worker_agent = Worker(state_dim + state_dim, action_dim)

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
    episode_worker_reward = 0
    episode_manager_extrinsic_reward = 0
    episode_manager_intrinsic_reward = 0

    manager_q1_loss = []
    manager_q2_loss = []
    manager_policy_loss = []
    manager_alpha_loss = []
    manager_alpha_value = []

    worker_q1_loss = []
    worker_q2_loss = []
    worker_policy_loss = []
    world_model_loss = []
    worker_alpha_loss = []
    worker_alpha_value = []

    curiosity_engine_loss = []

    source_state = state
    target_state = np.zeros(state_dim)
    manager_extrinsic_reward = 0.0
    k = 10

    for step in range(max_steps):
        if step % k == 0:
            source_state = state
            target_state = manager_agent.select_action(source_state)
            manager_extrinsic_reward = 0.0

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

        manager_extrinsic_reward += extrinsic_reward

        if (step + 1) % k == 0 or done:
            manager_agent.store_transition(
                source_state,
                target_state,
                manager_extrinsic_reward,
                next_state,
                done,
            )

            manager_intrinsic_reward = manager_agent.calculate_intrinsic_reward(source_state, target_state)
            episode_manager_extrinsic_reward += manager_extrinsic_reward
            episode_manager_intrinsic_reward += manager_intrinsic_reward

            writer.add_scalars(
                'Score',
                {
                    'ManagerExtrinsicReward': manager_extrinsic_reward,
                    'ManagerIntrinsicReward': manager_intrinsic_reward,
                },
                step_global,
            )

        if training:
            worker_train_values = worker_agent.train()

            worker_q1_loss.append(worker_train_values[0])
            worker_q2_loss.append(worker_train_values[1])
            worker_policy_loss.append(worker_train_values[2])
            worker_alpha_loss.append(worker_train_values[3])
            worker_alpha_value.append(worker_train_values[4])

            if (step + 1) % k == 0:
                manager_train_values = manager_agent.train()
                curiosity_engine_loss.append(manager_agent.train_curiosity_engine())

                manager_q1_loss.append(manager_train_values[0])
                manager_q2_loss.append(manager_train_values[1])
                manager_policy_loss.append(manager_train_values[2])
                manager_alpha_loss.append(manager_train_values[3])
                manager_alpha_value.append(manager_train_values[4])

        state = next_state
        episode_worker_reward += worker_reward

        writer.add_scalars(
            'Score',
            {
                'WorkerReward': worker_reward,
            },
            step_global,
        )

        step_global += 1

        if done:
            break

    print(f"Episode {episode}: Worker Reward = {episode_worker_reward:.2f}; Steps = {step + 1:04d}")

    if not training:
        continue

    writer.add_scalar('Loss/Manager/Q1', sum(manager_q1_loss) / len(manager_q1_loss), episode)
    writer.add_scalar('Loss/Manager/Q2', sum(manager_q2_loss) / len(manager_q2_loss), episode)
    writer.add_scalar('Loss/Manager/Policy', sum(manager_policy_loss) / len(manager_policy_loss), episode)
    # writer.add_scalar('Loss/WorldModel', sum(world_model_loss) / len(world_model_loss), episode)
    writer.add_scalar('Loss/Manager/Alpha', sum(manager_alpha_loss) / len(manager_alpha_loss), episode)

    writer.add_scalar('Loss/Worker/Q1', sum(worker_q1_loss) / len(worker_q1_loss), episode)
    writer.add_scalar('Loss/Worker/Q2', sum(worker_q2_loss) / len(worker_q2_loss), episode)
    writer.add_scalar('Loss/Worker/Policy', sum(worker_policy_loss) / len(worker_policy_loss), episode)
    # writer.add_scalar('Loss/WorldModel', sum(world_model_loss) / len(world_model_loss), episode)
    writer.add_scalar('Loss/Worker/Alpha', sum(worker_alpha_loss) / len(worker_alpha_loss), episode)

    if len(curiosity_engine_loss) > 0:
        writer.add_scalar('Loss/Curiosity', sum(curiosity_engine_loss) / len(curiosity_engine_loss), episode)

    writer.add_scalar('Score/Worker', episode_worker_reward, episode)
    writer.add_scalar('Score/ManagerExtrinsic', episode_manager_extrinsic_reward, episode)
    writer.add_scalar('Score/ManagerIntrinsic', episode_manager_intrinsic_reward, episode)

    writer.add_scalar('Other/Manager/AlphaValue', sum(worker_alpha_value) / len(worker_alpha_value), episode)
    writer.add_scalar('Other/Worker/AlphaValue', sum(worker_alpha_value) / len(worker_alpha_value), episode)

    writer.add_scalar('Exploration/StateCoverage', state_visitation_tracker.coverage, episode)
    writer.add_scalar('Exploration/UniqueStatesVisited', state_visitation_tracker.unique_states_visited, episode)

    if (episode + 1) % 10 == 0:
        manager_agent.save('.artifacts/manager_agent_state')
        worker_agent.save('.artifacts/worker_agent_state')

writer.close()
env.close()
