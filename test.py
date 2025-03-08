from data.utils import load_dataset, create_dataloader

from model.utils import load_classifier_model, get_device
from hyperparameters import dataset_name, batch_size

import torch 
import torch.nn as nn  
from tqdm import tqdm 

# Set up device 
device = get_device()

# Load the dataset
dataset, num_classes = load_dataset(dataset = dataset_name, train_split = False)

# Create the data loaders
dataloader = create_dataloader(dataset, batch_size, shuffle = False) 

# Model path
model_path = "./model/training_results/cifar100/20250308_103803/model.pth" 

# Load the model
classifier_model = load_classifier_model(model_path, num_classes).to(device)

# Set up criteria  
criterion = nn.CrossEntropyLoss()

# Use this model for inference
classifier_model.eval()

#
running_valid_loss = 0.0
running_valid_correct = 0
running_valid_total = 0

valid_bar = tqdm(dataloader, desc="Evaluation[Test]", leave=False)
with torch.no_grad():
    for inputs, labels in valid_bar:
        inputs, labels = inputs.to(device), labels.to(device)
            
        outputs = classifier_model(inputs)
        loss = criterion(outputs, labels)
            
        running_valid_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        running_valid_total += labels.size(0)
        running_valid_correct += (predicted == labels).sum().item()

        # Update progress bar with current loss and accuracy
        current_loss = running_valid_loss / running_valid_total
        current_acc = 100.0 * running_valid_correct / running_valid_total
        valid_bar.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.2f}%")

avg_valid_loss = running_valid_loss / len(dataset)
valid_accuracy = 100.0 * running_valid_correct / len(dataset)

print("Result [Test]:")
print(f"\tTest Loss: {avg_valid_loss:.4f}\n\tTest Accuracy: {valid_accuracy:.2f}%\n")
    
