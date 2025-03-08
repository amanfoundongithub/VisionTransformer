import torch 
import torch.nn as nn 
import torch.nn.functional as F


class FeedForward(nn.Module):
    
    def __init__(self, in_features, hidden_features, dropout = 0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout)
        )
    
    
    def forward(self, x):
        return self.mlp(x) 