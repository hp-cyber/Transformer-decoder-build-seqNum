import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, 
                 d_ff=2048, max_len=2048, dropout=0.1):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
    def forward(self, x):
        # Get sequence length and device
        b, seq_len = x.size()
        device = x.device
        
        # Token embeddings
        tok_emb = self.token_embedding(x)  # (b, seq_len, d_model)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :seq_len, :]  # (1, seq_len, d_model)
        x = tok_emb + pos_emb  # (b, seq_len, d_model)
        
        # Transformer decoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output projection
        logits = self.output(x)  # (b, seq_len, vocab_size)
        
        return logits

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with pre-norm
        attn_output = x + self.dropout(self.self_attn(self.norm1(x)))
        
        # Feed-forward with pre-norm
        output = attn_output + self.dropout(self.feed_forward(self.norm2(attn_output)))
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal (autoregressive) mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)  # Using GELU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Example usage
if __name__ == "__main__":
    vocab_size = 10000
    model = DecoderOnlyTransformer(vocab_size)
    
    # Example input: batch of token indices
    x = torch.randint(0, vocab_size, (2, 16))  # (batch_size=2, seq_len=16)
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")  # Should be (2, 16, 10000)