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

item_path = "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Test\\Items\\"
non_item_path = "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_Frames\\"


items_images = [item_path + i for i in os.listdir(item_path)]
#Get all items
non_items_images = [non_item_path + i for i in os.listdir(non_item_path)]
#Get non items
all_images = list(np.random.permutation(items_images + non_items_images))
#Combine all images' paths and shuffle


train_size = 0.7
val_size = 0.1
test_size = 0.2
    
train_images = all_images[0:int(np.ceil(len(all_images)*train_size))]
val_images = all_images[int(np.ceil(len(all_images)*train_size)):int(np.ceil(len(all_images)*(train_size+val_size)))]
test_images = all_images[int(np.ceil(len(all_images)*(train_size+val_size))):]

if len(train_images + val_images + test_images) != len(all_images):
    raise Exception('Train, validation and test sets have non empty intersection')

batch_size = 20
num_workers = 0

data = {'train': train_images, 'val': val_images, 'test': test_images}
transformations = {'train': augment, 'val': augmented_pred, 'test': augmented_pred}
dataset_sizes = {'train': len(train_images), 'val': len(val_images), 'test': len(test_images)}

datasets = {x: ZootrDataset(data[x], transform_=transformations[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, num_workers=num_workers, drop_last=(x != 'test'), shuffle=False) for x in ['train', 'val', 'test']}
