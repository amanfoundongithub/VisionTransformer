import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from preprocessing.PatchEmbedding import PatchEmbedding
from transformer.PositionalEmbedding import PositionalEmbedding
from transformer.TransformerBlock import TransformerBlock


class VisionTransformerClassifier(nn.Module):
    
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_blocks, num_heads, hidden_dim, rgb = True, dropout = 0.1):
        super().__init__()    
        
        self.embed_dim = embed_dim    
        
        # Define layers
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim, rgb)
        
        # Positional Embedding 
        self.pos_embedding   = PositionalEmbedding(self.patch_embedding.num_patches, embed_dim, dropout)
        
        # Class token 
        self.cls_token       = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
        # Initialize weights
        self._init_weights()
    
    
    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                # Truncated normal initialization for linear layers
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Truncated normal initialization for convolutional layers
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm: weights to 1 and bias to 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(init_func)
        
        # Initialize the class token separately
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim = 1)
        
        x = self.pos_embedding(x) 
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x) 
        
        cls_out = x[:, 0]
        return self.head(cls_out)