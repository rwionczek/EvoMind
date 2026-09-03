import torch
import torch.nn as nn


class EnsembleForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        return self.network(inputs)


class DisagreementCuriosityEngine:
    def __init__(self, state_dim, action_dim, ensemble_size=4, lr=1e-4):
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList([
            EnsembleForwardModel(state_dim, action_dim) for _ in range(ensemble_size)
        ])

        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

    def get_intrinsic_reward(self, state, action):
        self.models.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(state, action)
                predictions.append(pred)

            predictions = torch.stack(predictions, dim=0)

            ensemble_variance = torch.var(predictions, dim=0).mean(dim=-1)

        return ensemble_variance

    def train_step(self, state, action, next_state):
        self.models.train()
        self.optimizer.zero_grad()

        total_loss = 0
        for model in self.models:
            pred_next_state = model(state, action)
            total_loss += self.criterion(pred_next_state, next_state).mean()

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item() / self.ensemble_size
