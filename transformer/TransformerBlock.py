import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForward import FeedForward



class TransformerBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout = 0.1):
        super().__init__()
        
        # Define constants
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Define layers
        self.attn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadAttention(embed_dim, num_heads, dropout)
        )
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FeedForward(embed_dim, hidden_dim, dropout)
        )
        
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x