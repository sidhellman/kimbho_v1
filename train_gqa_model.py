import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1) RMSNorm
# -------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, emb_dim)
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.shape[-1]))
        return (x / (rms + self.eps)) * self.weight

# -------------------------------------------------
# 2) Rotary Positional Embeddings (RoPE)
# -------------------------------------------------
def build_rope_sin_cos(seq_len, dim, base=10000.0, device=None):
    if device is None:
        device = "cpu"
    position_ids = torch.arange(seq_len, device=device).unsqueeze(1)  # shape (seq_len, 1)
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=device)
    inv_base = 1.0 / (base ** (freq_seq * 2.0 / dim))
    angles = position_ids.float() * inv_base.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return sin, cos

def apply_rope(q_or_k, sin, cos):
    bsz, seq_len, dim = q_or_k.shape
    half_dim = dim // 2
    x_reshaped = q_or_k.view(bsz, seq_len, half_dim, 2)

    cos_ = cos.unsqueeze(0).unsqueeze(-1)  # shape (1, seq_len, half_dim, 1)
    sin_ = sin.unsqueeze(0).unsqueeze(-1)

    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]

    rope_x0 = x0 * cos_[..., 0] - x1 * sin_[..., 0]
    rope_x1 = x1 * cos_[..., 0] + x0 * sin_[..., 0]

    rope_reshaped = torch.stack([rope_x0, rope_x1], dim=-1)
    rope_out = rope_reshaped.view(bsz, seq_len, dim)
    return rope_out

# -------------------------------------------------
# 3) Grouped Query Attention Head
# -------------------------------------------------
class GroupedQueryAttentionHead(nn.Module):
    """
    A single "head" of grouped-query attention with Rotary Positional Embeddings.
    """
    def __init__(self, emb_dim, d_h, group_size=8, rope_base=10000.0):
        super().__init__()
        self.W_Q = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_K = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_V = nn.Parameter(torch.rand(emb_dim, d_h))
        self.d_h = d_h
        self.group_size = group_size
        self.rope_base = rope_base

    def forward(self, x, mask, sin, cos):
        # (batch_size, seq_len, d_h)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Apply RoPE
        Q = apply_rope(Q, sin, cos)
        K = apply_rope(K, sin, cos)

        bsz, seq_len, _ = Q.shape
        outputs = []

        # Chunk Q into groups
        for start_idx in range(0, seq_len, self.group_size):
            end_idx = min(start_idx + self.group_size, seq_len)
            Q_chunk = Q[:, start_idx:end_idx, :]  # (bsz, chunk_size, d_h)

            scores = Q_chunk @ K.transpose(-2, -1) / math.sqrt(self.d_h)
            # Slicing the mask for the chunk
            chunk_mask = mask[start_idx:end_idx, :]  # shape: (chunk_size, seq_len)
            chunk_mask = chunk_mask.unsqueeze(0).expand(bsz, -1, -1)

            # Apply mask
            scores = scores.masked_fill(chunk_mask == 0, float("-inf"))

            # Attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (bsz, chunk_size, seq_len)
            chunk_output = attn_weights @ V  # (bsz, chunk_size, d_h)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)  # (bsz, seq_len, d_h)

# -------------------------------------------------
# 4) Multi-Head Grouped Query Attention
# -------------------------------------------------
class MultiHeadGQAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, group_size=8, rope_base=10000.0):
        super().__init__()
        d_h = emb_dim // num_heads
        self.heads = nn.ModuleList([
            GroupedQueryAttentionHead(emb_dim, d_h, group_size, rope_base)
            for _ in range(num_heads)
        ])
        self.W_0 = nn.Parameter(torch.rand(emb_dim, emb_dim))

    def forward(self, x, mask, sin, cos):
        head_outputs = [head(x, mask, sin, cos) for head in self.heads]
        multi_head_out = torch.cat(head_outputs, dim=-1)  # (bsz, seq_len, emb_dim)
        return multi_head_out @ self.W_0

# -------------------------------------------------
# 5) MLP
# -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W_1 = nn.Parameter(torch.rand(emb_dim, emb_dim * 4))
        self.B_1 = nn.Parameter(torch.rand(emb_dim * 4))
        self.W_2 = nn.Parameter(torch.rand(emb_dim * 4, emb_dim))
        self.B_2 = nn.Parameter(torch.rand(emb_dim))

    def forward(self, x):
        x = x @ self.W_1 + self.B_1
        x = torch.relu(x)
        x = x @ self.W_2 + self.B_2
        return x

# -------------------------------------------------
# 6) DecoderBlock
# -------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, group_size=8, rope_base=10000.0):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadGQAttention(emb_dim, num_heads, group_size, rope_base)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    def forward(self, x, mask, sin, cos):
        attn_out = self.attn(self.norm1(x), mask, sin, cos)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

# -------------------------------------------------
# 7) DecoderLanguageModel
# -------------------------------------------------
class DecoderLanguageModel(nn.Module):
    """
    GPT-like decoder-only language model using Grouped-Query Attention + RoPE.
    """
    def __init__(
        self, 
        vocab_size, 
        emb_dim, 
        num_heads, 
        num_blocks, 
        pad_idx, 
        group_size=8, 
        rope_base=10000.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, group_size, rope_base)
            for _ in range(num_blocks)
        ])
        self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.pad_idx = pad_idx
        self.group_size = group_size
        self.rope_base = rope_base

    def forward(self, x):
        bsz, seq_len = x.shape
        x = self.embedding(x)  # (bsz, seq_len, emb_dim)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # RoPE sin/cos
        d_h = self.emb_dim // self.num_heads
        sin, cos = build_rope_sin_cos(seq_len, d_h, base=self.rope_base, device=x.device)

        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, mask, sin, cos)

        logits = x @ self.output  # (bsz, seq_len, vocab_size)
        return logits

# -------------------------------------------------
# 8) Simple Dataset Example
# -------------------------------------------------
class RandomTextDataset(Dataset):
    """
    This is just a dummy dataset that returns random tokens, for demonstration.
    Replace with your real dataset that tokenizes text.
    """
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # input_ids: shape (seq_len,)
        # target_ids: shape (seq_len,) which is input_ids shifted by 1 in a real LM dataset
        # For demonstration, we'll just create random tokens
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,))
        # Shifted by 1 in a real scenario, but let's keep it random here
        target_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,))
        return input_ids, target_ids

# -------------------------------------------------
# 9) Training Script
# -------------------------------------------------
def train_model():
    # Hyperparameters
    vocab_size = 1000
    emb_dim = 64
    num_heads = 8
    num_blocks = 2
    pad_idx = 0
    group_size = 4
    rope_base = 10000.0
    seq_len = 32
    batch_size = 16
    num_samples = 10000   # total random "samples" in our dummy dataset
    num_epochs = 3
    lr = 1e-4

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderLanguageModel(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        pad_idx=pad_idx,
        group_size=group_size,
        rope_base=rope_base
    ).to(device)

    # Create dataset + dataloader (dummy random data)
    dataset = RandomTextDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)     # (batch_size, seq_len)
            target_ids = target_ids.to(device)   # (batch_size, seq_len)

            # Forward pass
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropy:
            # CrossEntropy wants (N, C) vs (N,), so we flatten
            batch_size_, seq_len_, vocab_size_ = logits.shape
            logits_reshaped = logits.view(batch_size_ * seq_len_, vocab_size_)
            target_reshaped = target_ids.view(batch_size_ * seq_len_)

            # Compute loss
            loss = criterion(logits_reshaped, target_reshaped)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Logging
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step+1}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # (Optional) Save checkpoint each epoch
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")
    return model

# -------------------------------------------------
# 10) Example: Running the whole script
# -------------------------------------------------
if __name__ == "__main__":
    trained_model = train_model()
    # Now you can do inference, e.g. using your decoding function.
