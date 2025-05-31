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
n_embd = 512
n_head = 2
n_layer = 2
dropout = 0.2


# chars = ""
# with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
#     chars = sorted(list(set(text)))
#
# vocab_size = len(chars)
#
# string_to_int = {ch: i for i, ch in enumerate(chars)}
# int_to_string = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [string_to_int[c] for c in s]
# decode = lambda l: ''.join([int_to_string[i] for i in l])
#
# data = torch.tensor(encode(text), dtype=torch.long)
#
# n = int(0.8 * len(data))
# train_data = data[:n]
# val_data = data[n:]


# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i + block_size] for i in ix])
#     y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y


# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


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
    def __init__(self, memory_sequence_chunk_size):
        super().__init__()
        self.input_layer = nn.Linear(memory_sequence_chunk_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, memory_sequence_chunk_size)

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
        x = self.blocks(x)
        x = self.ln_f(x)
        outputs = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = outputs.shape
            outputs = outputs.view(B * T, C)
            targets = targets.view(B * T, C)
            loss = F.mse_loss(outputs, targets)

        return outputs, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            props = F.softmax(logits, dim=1)
            index_next = torch.multinomial(props, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# model = GPTModel(vocab_size)
# m = model.to(device)
#
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for iter in range(max_iters):
#     if iter % eval_iters == 0:
#         losses = estimate_loss()
#         print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
#
#     xb, yb = get_batch('train')
#
#     logits, loss = model.forward(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# print(loss.item())
#
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)
