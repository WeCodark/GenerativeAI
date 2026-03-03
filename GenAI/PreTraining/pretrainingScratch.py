import torch
import torch.nn as nn
import torch.nn.functional as F
import math

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read() # Entire Shakespear text data

# We are extracting all unique characters from the text
char = sorted(list(set(text)))
vocab_size = len(char)

# Let's Create lookup table
# string to index
# index to string
stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}

# Convert String --> List of integer
def encode(s):
    return [stoi[c] for c in s]

# Convert List of integer --> String
def decode(l):
    return "".join([itos[i] for i in l])

# we will convert entire dataset into 1D tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters

batch_size = 32 # no of independent sequences we process in parallel
# Model sees 128 character at once and predicts the next character
block_size = 128 # maximum context length for a single sequence (how many previous tokens model can look at)
max_iters = 3000 # total training steps
learning_rate = 3e-4 # step size for optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_embd = 128  # Embedding Dimension (size of vector representation of each token)
n_head = 4    # Number of attention heads
n_layer = 4   # Number of transformer blocks
dropout = 0.2 # It randomly sets 20% of the neurons to 0 during training (regularization technique to prevent overfitting)


# Batch Sampling Function
# Each chunk is length of block_size
# We add 1 to the chunk to predict the next character --> this is the target
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # We randomly select 'batch_size' number of starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # We stack these chunks to create a batch of input sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    # We stack the next characters to create a batch of target sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# We will Create Self-Attention Head

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        # Linear Layers to project input to key, query, and value
        # Query -> What I am looking for
        # Key -> What I have
        # Value -> What I will give you if you match
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # We are creating lower triangle matrix for casual masking
        # This is used to prevent the model from attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Batch, Time, Channels
        B, T, C = x.shape
        # Project input to key, query, and value
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # Each token compares with every other token
        wei = (q @ k.transpose(-2, -1))
        # scale by sqrt(head_size) for numerical stability
        wei = wei / math.sqrt(k.size(-1))

        # Apply causal mask (prevent attending to future tokens)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Softmax to get attention weights
        wei = F.softmax(wei, dim=-1)
        # Apply dropout
        wei = self.dropout(wei)
        # Multiply with value to get output
        out = wei @ v
        return out

# MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()


        # We are creating multiple attention heads
        # Each head will have its own key, query, and value
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # We are concatenating the output of all heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # here we are concatenate outputs of all heads along the embedding dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # We are applying linear transformation to the concatenated output
        out = self.dropout(self.proj(out))
        return out

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()


        # We are creating a feed forward network
        # It will have two linear layers
        # The first layer will have 4 * n_embd neurons
        # The second layer will have n_embd neurons
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer Block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # This prevents 

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# GPT Model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Token Embedding table
        self.token_embedding = nn.Embedding(vocab_size,n_embd)
        # Position Embedding table
        self.position_embedding = nn.Embedding(block_size,n_embd)
        # Transformer stack
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # Layer Normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # Output Layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        # Token embeddings
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        # Position embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, n_embd)
        # Combine token and position embeddings
        x = tok_emb + pos_emb # (B, T, n_embd)
        # Pass through transformer blocks
        x = self.blocks(x) # (B, T, n_embd)
        # Apply layer normalization
        x = self.ln_f(x) # (B, T, n_embd)
        # Apply output layer
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # flatten the logits and targets for cross entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # generation function (AutoRegressive decoding)
    def generate(self, idx, max_new_tokens,temperature=1, top_k = None):

        for _ in range(max_new_tokens):
            # crop context if it is longer than block size
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature
            # apply top-k filtering

            if top_k is not None:
                # keep only top_k highest logits
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                next_token = indices.gather(-1, torch.multinomial(probs,1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs,1)
            # append the next token to the sequence
            idx = torch.cat((idx, next_token), dim=1)
        return idx

## Training Loop

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # sample a batch of data
    xb, yb = get_batch('train')
    # forward pass
    logits, loss = model(xb, yb)
    # backward pass computes the gradient for every parameter
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # update the parameters weights
    optimizer.step()
    # print the loss
    if iter % 300 == 0:
        print(f"Iteration {iter}, Loss: {loss.item()}")

# We will Generate Text from Prompt

prompt = 'ROMEO:\n'

context = torch.tensor([encode(prompt)],dtype=torch.long).to(device)
generated = model.generate(
    context,
    max_new_tokens = 300,
    temperature = 0.8,
    top_k = 40
)

print("ShakespearGPT Output:")
print(decode(generated[0].tolist()))
