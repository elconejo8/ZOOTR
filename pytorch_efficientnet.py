import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision import transforms
#from albumentations import ToFloat, Resize, Compose, Normalize
#from albumentations.pytorch import ToTensor
from torch_dataset import ZootrDataset 
import numpy as np
from torch import softmax
from sklearn.metrics import roc_auc_score

def augment(x):
    transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((128, 128)),
         transforms.CenterCrop((100, 100)),
         transforms.RandomCrop((80, 80)),
         #transforms.RandomHorizontalFlip(p=0.5),
         #transforms.RandomRotation(degrees=(-90, 90)),
         #transforms.RandomVerticalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    x_trans = transform(x)
    return x_trans

def augmented_pred(x):
    transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((128, 128)),
         transforms.CenterCrop((100, 100)),
         transforms.Resize((80, 80)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    x_trans = transform(x)
    return x_trans
    


def to_tensor(x):
    transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    x_trans = transform(x)
    return x_trans


def eval_model(model, data, device):
   
    model.eval()
    actuals, predictions, all_labels = [], [], []
            
    with torch.no_grad():
                for images, labels in data:
                    images = images.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)    
                    # Prediction
                    out = model(images)
                    actuals.extend(labels.cpu().numpy().astype(int))
                    predictions.extend(softmax(out, 1).cpu().numpy())
                    all_labels.extend(labels) 

    
    # Prepare predictions and actuals
    predictions = np.array(predictions)
    # Choose label (array)
    #predicted_labels = predictions.argmax(1)
    predicted_probs = predictions[:,1]
     
    roc_score = roc_auc_score(all_labels, predicted_probs)
    
    d = {}
    d['Roc_score'] = roc_score
    d['Preds'] = predicted_probs
    d['Labels'] = all_labels
    return d

class EfficientNetwork(nn.Module):
    def __init__(self, output_size, b1=False, b2=False):
        super().__init__()
        self.b1, self.b2 = b1, b2
        
        # Define Feature part
        if b1:
            self.features = EfficientNet.from_pretrained('efficientnet-b1')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Define Classification part
        if b1:
            self.classification = nn.Linear(1280, output_size)
        elif b2:
            self.classification = nn.Linear(1408, output_size)
        else:
            self.classification = nn.Linear(1280, output_size)
        
        
    def forward(self, image, prints=False):
        if prints: print('Input Image shape:', image.shape)
        
        image = self.features.extract_features(image)
        if prints: print('Features Image shape:', image.shape)
            
        if self.b1:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1280)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1408)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1280)
        if prints: print('Image Reshaped shape:', image.shape)
        
        out = self.classification(image)
        if prints: print('Out shape:', out.shape)
        
        return out
    
