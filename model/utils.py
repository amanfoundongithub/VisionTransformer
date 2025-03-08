import torch
import torch.optim as optim

from transformer.VisionTransformer import VisionTransformerClassifier

from hyperparameters import image_size, patch_size, embed_dim, num_blocks, num_heads, hidden_dim, rgb , dropout
from hyperparameters import learning_rate, nesterov_momentum, nesterov_needed

from model.optimizer import TransformerOptimizer

def create_classifier_model(num_classes : int):
    return VisionTransformerClassifier(
        image_size, patch_size, num_classes, embed_dim, num_blocks, num_heads, hidden_dim, rgb, dropout
    )
    

def get_trainer(model, warmup_steps = 4000):
    # return optim.SGD(model.parameters(), lr = learning_rate, momentum = nesterov_momentum, nesterov = nesterov_needed)
    return TransformerOptimizer(model, warmup_steps)
    
def get_no_of_parameters(model, only_trainable = False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else: 
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu" 


    
    