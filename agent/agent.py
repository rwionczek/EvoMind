import os

import numpy as np
import torch
import torch.nn.functional as F

from agent.curiosity import DisagreementCuriosityEngine
from agent.networks import PolicyNetwork, SoftQNetwork
from agent.replay_buffer import ReplayBuffer


class RewardNormalizer:
    def __init__(self, alpha=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.ema_mean = 0.0
        self.ema_sq = 0.0

    def normalize(self, reward):
        self.ema_mean = self.alpha * self.ema_mean + (1 - self.alpha) * reward
        self.ema_sq = self.alpha * self.ema_sq + (1 - self.alpha) * (reward ** 2)

        var = self.ema_sq - self.ema_mean ** 2

        std = np.sqrt(max(var, 0.0)) + self.epsilon

        return (reward - self.ema_mean) / std


class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=None):
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")

        print(f"Using device: {self.device}")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.q_net1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=lr)

        self.target_q_net1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.disagreement_curiosity_engine = DisagreementCuriosityEngine(state_dim, action_dim)
        self.disagreement_curiosity_engine.models.to(self.device)

        self.gamma = gamma
        self.tau = tau

        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(1000000, state_dim, action_dim)
        self.replay_buffer_short = ReplayBuffer(32, state_dim, action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy()[0]

    def calculate_intrinsic_reward(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        return self.disagreement_curiosity_engine.get_intrinsic_reward(state, action).item()

    def train(self):
        states, actions, extrinsic_rewards, next_states, dones = self.replay_buffer.sample(1024)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        extrinsic_rewards = torch.tensor(extrinsic_rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

        intrinsic_rewards = self.disagreement_curiosity_engine.get_intrinsic_reward(states, actions).unsqueeze(1)
        rewards = 0.0 * extrinsic_rewards + intrinsic_rewards

        q1_current = self.q_net1(states, actions)
        q2_current = self.q_net2(states, actions)

        next_actions, next_log_probs = self.policy_net.sample(next_states)

        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_probs

        q_target = rewards + (1 - dones) * self.gamma * next_q_target

        q1_loss = F.mse_loss(q1_current, q_target.detach())
        q2_loss = F.mse_loss(q2_current, q_target.detach())

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        sampled_actions, log_probs = self.policy_net.sample(states)
        q1_pi = self.q_net1(states, sampled_actions)
        q2_pi = self.q_net2(states, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_probs - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.q_net1, self.target_q_net1)
        self._soft_update(self.q_net2, self.target_q_net2)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()

    def train_curiosity_engine(self):
        state, action, _, next_state, _ = self.replay_buffer.sample(1024)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        loss = self.disagreement_curiosity_engine.train_step(state, action, next_state)

        return loss

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.replay_buffer_short.add(state, action, reward, next_state, done)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.policy_net.state_dict(), directory + "/policy.pt")
        torch.save(self.q_net1.state_dict(), directory + "/q1.pt")
        torch.save(self.q_net2.state_dict(), directory + "/q2.pt")
        torch.save(self.log_alpha, directory + "/alpha.pt")

    def load(self, directory):
        self.policy_net.load_state_dict(torch.load(directory + "/policy.pt"))
        self.q_net1.load_state_dict(torch.load(directory + "/q1.pt"))
        self.target_q_net1.load_state_dict(torch.load(directory + "/q1.pt"))
        self.q_net2.load_state_dict(torch.load(directory + "/q2.pt"))
        self.target_q_net2.load_state_dict(torch.load(directory + "/q2.pt"))
        self.log_alpha = torch.load(directory + "/alpha.pt")
