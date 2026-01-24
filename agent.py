import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, size, observation_space_size, action_space_size):
        self.size = size
        self.index = -1

        self.states = np.zeros((size, observation_space_size))
        self.actions = np.zeros((size, action_space_size))
        self.rewards = np.zeros(size)
        self.next_states = np.zeros((size, observation_space_size))
        self.dones = np.ones(size)
        self.train_mask = np.zeros(size)

    def add(self, state, action, reward, next_state, done):
        self.index = (self.index + 1) % self.size
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.train_mask[self.index] = 1.0

    def sample(self, batch_size):
        allowed_indices = np.where(self.train_mask == 1.0)[0]
        indices = np.random.choice(allowed_indices, batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class SoftQNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.mean_head = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_head = torch.nn.Linear(hidden_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        noise = torch.randn_like(mean)
        action = mean + noise * std

        log_prob = -0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(1, keepdim=True)

        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(1, keepdim=True)

        return action, log_prob


class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, target_entropy=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.gamma = gamma
        self.tau = tau

        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(1000000, state_dim, action_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy_net.sample(state)
        return action.cpu().numpy()[0]

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(1024)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)

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

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, file):
        torch.save(self.policy_net.state_dict(), file + ".policy.pt")
        torch.save(self.q_net1.state_dict(), file + ".q1.pt")
        torch.save(self.q_net2.state_dict(), file + ".q2.pt")
        torch.save(self.log_alpha, file + ".alpha.pt")

    def load(self, file):
        self.policy_net.load_state_dict(torch.load(file + ".policy.pt"))
        self.q_net1.load_state_dict(torch.load(file + ".q1.pt"))
        self.target_q_net1.load_state_dict(torch.load(file + ".q1.pt"))
        self.q_net2.load_state_dict(torch.load(file + ".q2.pt"))
        self.target_q_net2.load_state_dict(torch.load(file + ".q2.pt"))
        self.log_alpha = torch.load(file + ".alpha.pt")
