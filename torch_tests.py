import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


class ZootrDataset(Dataset):

    def __init__(self, items_folder_path, non_items_folder_path, is_train=True, transform_=None):

        self.items_folder_path = items_folder_path
        self.nr_positives = len(os.listdir(self.items_folder_path))
        
        self.non_items_folder_path = non_items_folder_path
        self.nr_negatives = len(os.listdir(self.non_items_folder_path))
        
        self.instance_list = [os.path.join(self.items_folder_path, i) for i in os.listdir(self.items_folder_path)] + [os.path.join(self.non_items_folder_path, i) for i in os.listdir(self.non_items_folder_path)]
        self.instance_list = np.random.permutation(self.instance_list)
        self.is_train = is_train
        self.transform_ = transform_

    def __len__(self):
        return self.nr_positives + self.nr_negatives

    def __getitem__(self, idx):
        img_name = self.instance_list[idx]
        image = io.imread(img_name)
        label = int('Random' in self.instance_list[idx])

        if self.transform_:
            image = self.transform_(image)

        return image, label
    
    def see_image(self, name):
        if type(name) == int:
            im, _ = data.__getitem__(name)
            io.imshow(im)
        else:
            image = io.imread(name)
            io.imshow(image)
       
    
    
data = ZootrDataset("C:\\Users\\Guilherme\\Documents\\ZOOTR\\Items", "C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_Frames")
data.__getitem__(0)

data.see_image(0)
data.instance_list[0:5]
data.see_image(data.instance_list[0])
