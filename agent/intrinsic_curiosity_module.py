import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.net(state)


class InverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, phi_state, phi_next_state):
        x = torch.cat([phi_state, phi_next_state], dim=-1)
        return self.net(x)


class ForwardModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, phi_state, action_one_hot):
        x = torch.cat([phi_state, action_one_hot], dim=-1)
        return self.net(x)


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=128):
        super().__init__()

        self.encoder = FeatureEncoder(state_dim, feature_dim)
        self.inverse_model = InverseModel(feature_dim, action_dim)
        self.forward_model = ForwardModel(feature_dim, action_dim)

    def forward(self, state, next_state, action):
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)

        predicted_action = self.inverse_model(phi_state, phi_next_state)

        action_one_hot = F.one_hot(action, num_classes=predicted_action.shape[-1]).float()
        predicted_phi_next = self.forward_model(phi_state, action_one_hot)

        return phi_state, phi_next_state, predicted_action, predicted_phi_next

    def loss(self, state, next_state, action):
