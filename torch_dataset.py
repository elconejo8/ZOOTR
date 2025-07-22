import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ZootrDataset(Dataset):

    def __init__(self, instance_list, is_train=True, transform_=None):        
        self.is_train = is_train
        if self.is_train:
            self.instance_list = np.random.permutation(instance_list)
        else:
            self.instance_list = instance_list
        self.transform_ = transform_

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        img_name = self.instance_list[idx]
        image = io.imread(img_name)

        if self.transform_:
            image = self.transform_(image)


        if self.is_train:
            label = int('Random' not in self.instance_list[idx])
            return image, label
        else:
            return image

    
    def see_image(self, name):
        if type(name) == int:
            im, _ = self.__getitem__(name)
            transforms.ToPILImage()(im)
        else:
            image = io.imread(name)
            io.imshow(image)
       
    
