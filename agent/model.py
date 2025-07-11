import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

block_size = 16
batch_size = 16
max_iters = 100
learning_rate = 3e-4
eval_iters = 100
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.1


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, observation_size, action_size, reward_size):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.reward_size = reward_size
        chunk_size = observation_size + action_size + reward_size
        self.input_layer = nn.Linear(chunk_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, chunk_size)

    def __init_weights(self, module):
        if (isinstance(module, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, trajectory, targets=None):
        B, T, C = trajectory.shape

        input = self.input_layer(trajectory)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = input + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        outputs = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = outputs.shape
            outputs = outputs.view(B * T, C)
            targets = targets.view(B * T, C)

            observation_outputs = outputs[:, :self.observation_size]
            observation_targets = targets[:, :self.observation_size]
            observation_loss = F.mse_loss(observation_outputs, observation_targets)

            action_outputs = outputs[:, self.observation_size:self.observation_size + self.action_size]
            action_targets = targets[:, self.observation_size:self.observation_size + self.action_size].argmax(dim=1)
            action_loss = F.cross_entropy(action_outputs, action_targets)

            reward_outputs = outputs[:, -self.reward_size:]
            reward_targets = targets[:, -self.reward_size:]
            reward_loss = F.mse_loss(reward_outputs, reward_targets)

            loss = observation_loss + action_loss + reward_loss

        return outputs, loss
