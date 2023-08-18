
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np

class Spec_Dataset(Dataset):
    def __init__(self, file_path, possiblity, label_data, transform = False):
        super().__init__()
        self.file_path = file_path
        self.length = len(file_path)
        self.possbility = possiblity
        self.label_data = label_data
        self.transform = transform
   
    def __len__(self):
        return self.length
    
    def rand1(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def rand2(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self,index):
        index = index % self.length
        data = np.load(self.file_path[index])
        bands = [0, 59,  95, 119, 149, 168, 186, 202, 287]
        data = data[:,:,bands]
        data = np.transpose(data,(2,0,1))
        x_data = torch.tensor(data,requires_grad=True,dtype=float)
        # data augmentation
        if self.transform:
            # 1.GaussianBlur
            if self.rand1() < self.possbility:
                x_data = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))(x_data)
            # 2.Random Flip
            x_data = T.RandomVerticalFlip(p=0.5)(T.RandomHorizontalFlip(p=0.5)(x_data))
            # 3.Random Rotation
            x_data = T.RandomRotation(degrees=(0, 180))(x_data)
            # 4.ElasticTransform
            # x_data = T.ElasticTransform()(x_data)
            # 4.RandomResizedCrop
            if self.rand2() < self.possbility:
                x_data = T.RandomResizedCrop(scale=(0.8, 1.0), size=(x_data.shape[1], x_data.shape[2]))(x_data)
        y_data = self.label_data[index]
        y_data = torch.tensor(y_data)
        return x_data, y_data
    
def form_dataset(f_train_x, train_batch_size, f_train_y, f_test_x, test_batch_size, f_test_y):
    train_dataset = Spec_Dataset(file_path = f_train_x, possiblity = 0.8, label_data = f_train_y, transform = True)
    train_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
    train_size = len(train_dataset)
    valid_dataset = Spec_Dataset(file_path = f_test_x, possiblity = 0.8, label_data = f_test_y, transform = False)
    valid_dataloader = DataLoader(valid_dataset, batch_size = test_batch_size, shuffle = False)
    valid_size = len(valid_dataset)
    dataloader = {'train': train_dataloader, 'valid':valid_dataloader}
    dataset_size = {'train': train_size, 'valid': valid_size}
    return dataloader, dataset_size