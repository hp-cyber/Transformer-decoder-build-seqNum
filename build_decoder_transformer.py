import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: dimensionality of the embeddings.
            max_len: maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# -----------------------------
# Multi-Head Self-Attention with Causal Masking
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: dimensionality of the model.
            num_heads: number of attention heads.
            dropout: dropout rate.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers to project input to queries, keys, and values
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to query, key, and value vectors
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Split into multiple heads and transpose for attention computation
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, seq, d_k)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq, seq)
        
        # Create a causal mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute attention output
        out = torch.matmul(attn, v)  # (batch, heads, seq, d_k)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_linear(out)
        return out

# -----------------------------
# Feed Forward Network
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: dimensionality of the model.
            d_ff: dimensionality of the feed-forward layer.
            dropout: dropout rate.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# -----------------------------
# Transformer Decoder Block
# -----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Single transformer decoder block.
        """
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention sublayer with residual connection
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward sublayer with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# -----------------------------
# Decoder-Only Transformer
# -----------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        """
        Args:
            vocab_size: size of the vocabulary.
            d_model: dimensionality of embeddings and model.
            num_layers: number of decoder blocks.
            num_heads: number of attention heads.
            d_ff: dimensionality of the feed-forward network.
            max_len: maximum sequence length.
            dropout: dropout rate.
        """
        super(DecoderOnlyTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        # Final projection layer to logits
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: input tensor of token indices with shape (batch_size, seq_len)
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.token_embedding(x)    # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000   # Example vocabulary size
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    max_len = 100
    dropout = 0.1

    # Instantiate model
    model = DecoderOnlyTransformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
    
    # Example input: batch of sequences of token indices (batch_size, seq_len)
    batch_size = 2
    seq_len = 20
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(sample_input)
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)
