import math
import torch
import torch.nn.functional as F
from torch import nn

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
    print("  [scaled_dot_product_attention]")
    print(f"    query shape: {query.shape}")  # (B, heads, T_q, d_k)
    print(f"    key shape:   {key.shape}")    # (B, heads, T_k, d_k)
    print(f"    value shape: {value.shape}")  # (B, heads, T_k, d_v)

    d_k = query.size(-1)
    
    # (B, heads, T_q, T_k)
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    print(f"    attention_scores shape: {attention_scores.shape}")
    
    scaled_scores = attention_scores / math.sqrt(d_k)
    print(f"    scaled_scores shape: {scaled_scores.shape}")

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scaled_scores, dim=-1)
    print(f"    attention_weights shape: {attention_weights.shape}")

    # (B, heads, T_q, d_v)
    output = torch.matmul(attention_weights, value)
    print(f"    output shape: {output.shape}")
    return output, attention_weights


# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        print("[MultiHeadAttention]")
        B, T, C = x.size()
        print(f"  Input x shape: {x.shape} (B={B}, T={T}, C={C})")

        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        print(f"  Q shape after linear: {Q.shape}")
        print(f"  K shape after linear: {K.shape}")
        print(f"  V shape after linear: {V.shape}")

        # Split into heads and transpose: (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        print(f"  Q reshaped: {Q.shape}")
        print(f"  K reshaped: {K.shape}")
        print(f"  V reshaped: {V.shape}")

        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        print(f"  attn_output shape: {attn_output.shape}")

        # Concatenate heads: (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        print(f"  attn_output after concat: {attn_output.shape}")

        # Final linear projection
        output = self.out_proj(attn_output)
        print(f"  Final output shape: {output.shape}")
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        self.pos = torch.arange(0, max_len).unsqueeze(1)
        
        # some magic which enables pytorch to run faster
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1e5) / d_model)) 

        self.pe[:, ::2] = torch.sin(self.pos * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.pos * self.div_term)

        self.pe = self.pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor):
        x += self.pe

class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        

# ====== Testing the module ======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define dimensions
batch_size = 2
seq_len = 5
d_model = 64
num_heads = 8

# Create dummy input tensor (e.g., token embeddings)
x_0 = torch.randn(batch_size, seq_len, d_model).to(device)  # shape: (2, 5, 64)

# Instantiate attention module
attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)

# Forward pass
x_1 = attention(x_0)

print(f"\nFinal output shape: {x_1.shape}")