import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class PositionalEmbedding(nn.Module):
    
    def __init__(self, num_patches, embed_dim, dropout = 0.1):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):        
        return self.dropout(x + self.pos_embed)