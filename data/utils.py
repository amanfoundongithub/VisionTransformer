from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from preprocessing.utils import get_image_transformation
from hyperparameters import no_of_epochs

import matplotlib.pyplot as plt 

def load_dataset(dataset : str = "cifar100", root_directory = "./root", train_split = True):
    
    if dataset == "cifar100":
        return CIFAR100(
            root = root_directory, 
            train = train_split, 
            download = True, 
            transform = get_image_transformation(train_split)
        ), 100
    
    elif dataset == "cifar10":
        return CIFAR10(
            root = root_directory, 
            train = train_split, 
            download = True, 
            transform = get_image_transformation(train_split)
        ), 10
    
    else:
        raise ValueError(f"The dataset {dataset} could not be found!")
    

def split_dataset(dataset, fraction):
    total_length = len(dataset)
    split_length = int(total_length * fraction)
    remaining_length = total_length - split_length

    subset1, subset2 = random_split(dataset, [split_length, remaining_length])
    return subset2, subset1


def create_dataloader(dataset, batch_size, num_workers = 1, shuffle = True):
    return DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = shuffle,
        num_workers = num_workers
    )
    

