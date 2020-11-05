import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from pytorch_efficientnet import EfficientNetwork
from new_torch import train_model
from new_train import dataloaders, dataset_sizes


batch_size = 20
num_workers = 0
model = EfficientNetwork(output_size=2, b1=False, b2=True)
learning_rate = 0.0001
weight_decay = 0.1
epochs = 10


model = EfficientNetwork(output_size=2, b1=False, b2=True)
# Criterion
criterion = torch.nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max',
                                                       patience=1, verbose=True, factor=0.4)

dics = {}
dics['dataloaders'] = dataloaders
dics['dataset_sizes'] = dataset_sizes

train_model(model, criterion, optimizer, scheduler, dics, num_epochs=25)
