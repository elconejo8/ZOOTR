import torch
import torch.optim as optim
from torch_dataset import ZootrDataset 
import numpy as np
import os
from pytorch_efficientnet import to_tensor, EfficientNetwork, eval_model, augment, augmented_pred
import random

random.seed(8)
load_data_objects = False
save_data_objects = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

item_path = "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Test\\Items\\"
non_item_path = "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_Frames\\"


if  load_data_objects:
    torch.load('train_data.pth')
    torch.load('valid_data.pth')
else:
    items_images = [item_path + i for i in os.listdir(item_path)]
    #Get all items
    non_items_images = [non_item_path + i for i in os.listdir(non_item_path)]
    #Get non items
    all_images = list(np.random.permutation(items_images + non_items_images))
    #Combine all images' paths and shuffle
    train_data = ZootrDataset(all_images[0:int(np.ceil(len(all_images)*0.8))], transform_=augment)
    train_data_eval = ZootrDataset(all_images[0:int(np.ceil(len(all_images)*0.8))], transform_=augmented_pred)
    valid_data = ZootrDataset(all_images[int(np.ceil(len(all_images)*0.8)):], transform_=augmented_pred)
    # Data Objects
    if save_data_objects:
        torch.save(train_data, 'train_data.pth')
        torch.save(valid_data, 'valid_data.pth')
    
    
batch_size = 20
num_workers = 0

data = {'train': all_images[0:int(np.ceil(len(all_images)*0.7))], 'val': all_images[int(np.ceil(len(all_images)*0.7)):int(np.ceil(len(all_images)*0.8))], 
        'test': all_images[int(np.ceil(len(all_images)*0.8)):]}
transformations = {'train': augment, 'val': augmented_pred, 'test': augmented_pred}
dataset_sizes = {'train': int(np.ceil(len(all_images)*0.7)), 'val': int(np.ceil(len(all_images)*0.8)) - int(np.ceil(len(all_images)*0.7)), 
                 'test': len(all_images) - int(np.ceil(len(all_images)*0.8))}

datasets = {x: ZootrDataset(data[x], transform_=transformations[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False) for x in ['train', 'val', 'test']}

