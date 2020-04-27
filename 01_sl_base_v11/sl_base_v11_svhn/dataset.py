
import pickle

from PIL import Image
import numpy as np 

import torch
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ImageDataset(data.Dataset):
    def __init__(self, data_dir, domain, img_size, ds, transform):
        self.transform = transform
        
        with open(f"{data_dir}/{domain}_{img_size}_{ds}.pkl", 'rb') as fin:
            self.x, self.y = pickle.load(fin)

        self.num_data = self.x.shape[0]
        
    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]

        # img = Image.fromarray((img * 255).astype(np.uint8))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.functional.to_tensor(img)

        return img, label
    
    def __len__(self):
        return self.num_data

    
def load_data(domain, data_dir, img_size, batch_size):

    normalize = transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1975, 0.2004, 0.1964])
    
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, 4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # load data
    train_lbl = ImageDataset(data_dir, domain, img_size, ds='train_lbl', transform=train_transform)
    dev_lbl = ImageDataset(data_dir, domain, img_size, ds='dev_lbl', transform=test_transform)
    test_lbl = ImageDataset(data_dir, domain, img_size, ds='test_lbl', transform=test_transform)

    train_lbl = data.DataLoader(
        train_lbl, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dev_lbl = data.DataLoader(
        dev_lbl, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_lbl = data.DataLoader(
        test_lbl, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


    print(f"train lbl dataset num data: {len(train_lbl.dataset)}")
    print(f"dev dataset num data: {len(dev_lbl.dataset)}")
    print(f"test dataset num data: {len(test_lbl.dataset)}")
    
    print(f"train lbl dataset num batches: {len(train_lbl)}")
    print(f"dev lbl dataset num batches: {len(dev_lbl)}")
    print(f"test lbl dataset num batches: {len(test_lbl)}")

    return train_lbl, dev_lbl, test_lbl

    
