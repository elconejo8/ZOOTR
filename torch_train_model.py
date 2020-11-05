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
model = EfficientNetwork(output_size=2, b1=False, b2=True)
learning_rate = 0.0001
weight_decay = 0.1
epochs = 10

print('Device available now:', device)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                           drop_last=True, shuffle=False)

train_loader_eval = torch.utils.data.DataLoader(train_data_eval, batch_size=batch_size, num_workers=num_workers,
                                           drop_last=True, shuffle=False)

valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, num_workers=num_workers,
                                           drop_last=True, shuffle=False)

# Criterion
criterion = torch.nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max',
                                                       patience=1, verbose=True, factor=0.4)

train_losses = []
evaluation_losses = []

for epoch in range(1, epochs+1):
    # Sets the model in training mode
    model.train()

    train_loss = 0
    counter = 0
    for images, labels in train_loader:
        # Need to access the images
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # Clear gradients
        optimizer.zero_grad()

        # Make prediction
        out = model(images)

        # Compute loss and Backpropagate
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        counter += 1
        
        

    # Compute average epoch loss
    epoch_loss_train = train_loss / batch_size
    train_losses.append(epoch_loss_train)
    torch.save(model.cpu().state_dict(), os.path.join('Models', 'model_' + str(epoch) + '.pth'))
    train_results = eval_model(model, train_loader_eval, device)
    valid_results = eval_model(model, valid_loader, device)
    print("Epoch", epoch)
    print("Epoch training loss:", epoch_loss_train )
    print("Epoch training score:", str(train_results['Roc_score']) )
    print("Epoch validation score", str(valid_results['Roc_score']))
    break

    
train_results = eval_model(model, train_loader, device)
valid_results = eval_model(model, valid_loader, device)
