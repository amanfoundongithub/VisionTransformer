import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()
        
        # Define constants
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.head_dim  = self.embed_dim // self.num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    
    def forward(self, x):
        """ 
        x : [batch_size, num_patches, embed_size]
        """
        batch_size, num_patches, embed_size = x.shape 
        
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        # Attention
        attn = (q @ k.transpose(-2, -1))/(self.head_dim ** 0.5)
        attn = self.attn_drop(F.softmax(attn, dim = -1))
        
        # Linear layer
        x = attn @ v 
        x = x.transpose(1, 2).reshape(batch_size, num_patches, embed_size)
        x = self.proj_drop(self.proj(x))
        
        return x 
    