from data.utils import load_dataset, split_dataset, create_dataloader

from model.utils import create_classifier_model, get_device, get_trainer, get_no_of_parameters

from hyperparameters import dataset_name, validation_split, batch_size, shuffle, no_of_epochs

import torch 
import matplotlib.pyplot as plt 
import torch.nn as nn  
from tqdm import tqdm 
from datetime import datetime 
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import os 

# Create a save directory under model/training using the dataset name and a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("model", "training_results", dataset_name, timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Saving models and plots to: {save_dir}")

# Set up device 
device = get_device()

# Load the dataset
dataset, num_classes = load_dataset(dataset = dataset_name)

# Split into train and validation set 
train_set, valid_set = split_dataset(dataset, validation_split)

# Create the data loaders
train_dataloader = create_dataloader(train_set, batch_size, shuffle = shuffle)
valid_dataloader = create_dataloader(valid_set, batch_size, shuffle = shuffle)

# Load the model and set up a trainer 
classifier_model = create_classifier_model(num_classes).to(device)
optimizer        = get_trainer(classifier_model) 

print(f"Number of params : {get_no_of_parameters(classifier_model)}")

# scheduler        = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
# scheduler        = ExponentialLR(optimizer, gamma = 0.9) 



# Set up criteria  
criterion = nn.CrossEntropyLoss()

# Lists to store metrics for plotting
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []
# List to store learning rate values for plotting
learning_rates = []

# Training loop begins below...
for epoch in range(no_of_epochs):
    
    # Set the training mode ON 
    classifier_model.train()
    
    # Metrics to compute at run time 
    running_train_loss = 0.0
    running_train_correct = 0
    running_train_total = 0
    
    # Training phase with progress bar
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{no_of_epochs} Training", leave=False)
    
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  
        
        outputs = classifier_model(inputs)  
        loss = criterion(outputs, labels)   
        loss.backward()                 
        
        optimizer.step()    
        
        # Track learning rate
        current_lr = optimizer.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)            

        # Accumulate training metrics
        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        running_train_total += labels.size(0)
        running_train_correct += (predicted == labels).sum().item()
        
        # Update progress bar with current loss and accuracy
        current_loss = running_train_loss / running_train_total
        current_acc = 100.0 * running_train_correct / running_train_total
        train_bar.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.2f}%")
    
    

    # Calculate average training metrics for the epoch
    avg_train_loss = running_train_loss / len(train_set)
    train_accuracy = 100.0 * running_train_correct / len(train_set)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation phase, Set model to evaluation mode
    classifier_model.eval()  
    
    running_valid_loss = 0.0
    running_valid_correct = 0
    running_valid_total = 0

    valid_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{no_of_epochs} Validation", leave=False)
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

    avg_valid_loss = running_valid_loss / len(valid_set)
    valid_accuracy = 100.0 * running_valid_correct / len(valid_set)
    valid_losses.append(avg_valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Print epoch metrics
    print(f"Epoch [{epoch+1}/{no_of_epochs}]:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"  Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%\n")
    print(f"  Current LR : {current_lr : .8f}\n")
    
    # scheduler.step(avg_valid_loss) 

# Plotting the metrics after training
epochs = range(1, no_of_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, valid_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
plt.plot(epochs, valid_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_save_path = os.path.join(save_dir, "metrics.png")
plt.savefig(plot_save_path)
plt.show()

plt.clf()
# Plotting the learning rate schedule
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, label='Learning Rate')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule over Steps')
plt.legend()
plt.grid(True)

# Save the learning rate plot
lr_plot_save_path = os.path.join(save_dir, "learning_rate_scheduling.png")
plt.savefig(lr_plot_save_path)
plt.show()

# Save the final model and the plot image in the created directory
model_save_path = os.path.join(save_dir, "model.pth")
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
print(f"Metrics plot saved to {plot_save_path}")