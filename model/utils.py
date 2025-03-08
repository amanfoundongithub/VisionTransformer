import torch
import torch.optim as optim

from transformer.VisionTransformer import VisionTransformerClassifier

from hyperparameters import image_size, patch_size, embed_dim, num_blocks, num_heads, hidden_dim, rgb , dropout

from model.optimizer import TransformerOptimizer

def get_device() -> str:
    """Get the appropriate device for PyTorch tensor operations.

    This function checks if a CUDA-enabled GPU is available. If so, it returns "cuda" to perform computations on the GPU. Otherwise, it returns "cpu" to use the CPU.

    Returns:
        str: The device string ("cuda" or "cpu").
    """
    return "cuda" if torch.cuda.is_available() else "cpu" 

def create_classifier_model(num_classes : int) -> VisionTransformerClassifier:
    """Create a Vision Transformer classifier model.

    This function instantiates a VisionTransformerClassifier model with the specified hyperparameters.

    Args:
        num_classes (int): The number of classes for the classification task.

    Returns:
        VisionTransformerClassifier: The initialized Vision Transformer classifier model.
    """
    return VisionTransformerClassifier(
        image_size, patch_size, num_classes, embed_dim, num_blocks, num_heads, hidden_dim, rgb, dropout
    )
    

def load_classifier_model(model_path : str, num_classes : int) -> VisionTransformerClassifier:
    """Load a pre-trained Vision Transformer classifier model.

    This function creates a Vision Transformer classifier model and loads pre-trained weights from the specified path.

    Args:
        model_path (str): The path to the saved model weights.
        num_classes (int): The number of classes for the classification task.

    Returns:
        VisionTransformerClassifier: The Vision Transformer classifier model with loaded weights.
    """
    vision_model = create_classifier_model(num_classes)
    vision_model.load_state_dict(torch.load(model_path))
    return vision_model
    

def get_trainer(model, warmup_steps = 4000) -> TransformerOptimizer:
    """Get a trainer object for optimizing the Vision Transformer model.

    This function creates a TransformerOptimizer object, which is responsible for handling the optimization process of the model.

    Args:
        model: The Vision Transformer model to be optimized.
        warmup_steps (int, optional): The number of warm-up steps for the learning rate scheduler. Defaults to 4000.

    Returns:
        TransformerOptimizer: The trainer object.
    """
    return TransformerOptimizer(model, warmup_steps)
    
def get_no_of_parameters(model, only_trainable = False) -> int:
    """Calculate the number of parameters in a model.

    This function computes the total number of parameters in the given model. It can optionally count only the trainable parameters.

    Args:
        model: The model for which to calculate the number of parameters.
        only_trainable (bool, optional): Whether to count only trainable parameters. Defaults to False.

    Returns:
        int: The total number of parameters in the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else: 
        return sum(p.numel() for p in model.parameters())




    
    