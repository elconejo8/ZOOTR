import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torchvision import transforms
from albumentations import ToFloat, Resize, Compose
from albumentations.pytorch import ToTensor

def to_tensor(x):
    compose = Compose([Resize(128,128),
                       ToFloat(max_value=255),
                       ToTensor()])
    x_trans = compose(image = x)['image']
    return x_trans

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
    
model_example = EfficientNetwork(output_size=1, b1=False, b2=True)

# Data object and Loader
example_data = ZootrDataset("C:\\Users\\Guilherme\\Documents\\ZOOTR\\Items", "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_Frames", to_tensor)
example_loader = torch.utils.data.DataLoader(example_data, batch_size = 1, shuffle=True)

# Get a sample
for image, labels in example_loader:
    images_example = image
    labels_example = torch.tensor(labels, dtype=torch.long)
    break
print('Images shape:', images_example.shape)
print('Labels:', labels, '\n')

# Outputs
out = model_example(images_example, prints=True)

# Criterion example
criterion_example = nn.CrossEntropyLoss()
loss = criterion_example(out, labels_example - 1)
print('Loss:', loss.item())



x = a0
Image.fromarray(x)
compose = Compose([Resize(128, 128),
                   ToFloat(max_value=255),
                   ToTensor()])
x_trans = compose(image = x)['image']
type(x_trans)
Image.fromarray(x_trans.detach().cpu().numpy().astype(np.uint8))

x = a0
compose = Compose([Resize(128,128),
                              ToFloat(max_value=255),
                              ToTensor()])
x_trans = compose(image = x)['image']
