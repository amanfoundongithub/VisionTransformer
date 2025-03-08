

# Dataset
dataset_name = "cifar100"
num_classes  = 100

# Validation split
validation_split = 0.15

# Batch size
batch_size = 128

# Shuffle? 
shuffle = True



####################################################################################

############################ Model Parameters ######################################

# Image size
image_size = 32
# Patch size 
patch_size = 4
# Embedding dimension
embed_dim  = 256
# Number of blocks
num_blocks = 12
# Number of heads
num_heads  = 8
# Hidden dimension
hidden_dim = 1024
# RGB ? 
rgb        = True 
# Dropout 
dropout    = 0.1


######################### Training Parameters ######################################

# Number of epochs 
no_of_epochs = 50

# Learning rate
learning_rate = 1e-2

# Nesterov momentum if needed
nesterov_momentum = 0.9
nesterov_needed   = True 

####################################################################################


