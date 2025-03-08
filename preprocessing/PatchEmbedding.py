import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class PatchEmbedding(nn.Module):
    """ 
    Implements the preprocessing block to patch the images and extract features
    """
    def __init__(self, img_size, patch_size, embed_dim = 256, rgb = True):
        super().__init__()
        
        self.in_channels = 1
        if rgb:
            self.in_channels = 3
        
        # Define constants
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.embed_size  = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Define feature network
        self.features = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = embed_dim, 
            kernel_size = patch_size, 
            stride = patch_size
        )
        
        
    def forward(self, x):
        """ 
        x : Image of shape [batch_size, channels, height, width]
        """
        
        x = self.features(x) 
        x = x.flatten(2).transpose(1, 2) 
        return x 
    
    