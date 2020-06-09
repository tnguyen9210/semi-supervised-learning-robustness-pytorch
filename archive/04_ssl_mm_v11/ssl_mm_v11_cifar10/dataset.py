
import pickle

import numpy as np 

import torch
import torch.nn.functional as F
import torch.utils.data as data


def load_data(domain, data_dir, img_size, num_iters_per_epoch, batch_size):

    # load data
    train_unlbl = ImageDataset(data_dir, domain, img_size, ds='train_unlbl_4000')
    train_lbl = ImageDataset(data_dir, domain, img_size, ds='train_lbl_4000')
    dev_lbl = ImageDataset(data_dir, domain, img_size, ds='dev_lbl')
    test_lbl = ImageDataset(data_dir, domain, img_size, ds='test_lbl')

    print(f"train unlbl dataset num data: {train_unlbl.num_data}")
    print(f"train lbl dataset num data: {train_lbl.num_data}")
    print(f"dev dataset num data: {dev_lbl.num_data}")
    print(f"test dataset num data: {test_lbl.num_data}")

    train_unlbl_sampler = data.RandomSampler(
        train_unlbl, replacement=True, num_samples=num_iters_per_epoch*batch_size//2)
    
    train_unlbl = data.DataLoader(
        train_unlbl, batch_size=batch_size//2, num_workers=16,
        sampler=train_unlbl_sampler)

    train_lbl_sampler = data.RandomSampler(
        train_lbl, replacement=True, num_samples=num_iters_per_epoch*batch_size//2)

    train_lbl = data.DataLoader(
        train_lbl, batch_size=batch_size//2, num_workers=16,
        sampler=train_lbl_sampler)
    
    dev_lbl = data.DataLoader(
        dev_lbl, batch_size=batch_size, shuffle=False, num_workers=16)
    test_lbl = data.DataLoader(
        test_lbl, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"train unlbl dataset num batches: {len(train_unlbl)}")
    print(f"train lbl dataset num batches: {len(train_lbl)}")
    print(f"dev lbl dataset num batches: {len(dev_lbl)}")
    print(f"test lbl dataset num batches: {len(test_lbl)}")

    return train_lbl, train_unlbl, dev_lbl, test_lbl


def collate_fn(batch, transform=None):
    images, labels = list(zip(*batch))
    
    images = torch.stack(images)
    if transform is not None:
        images = transform(images)

    labels = torch.LongTensor(labels)
    
    return (images, labels)


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, domain, img_size, ds):

        with open(f"{data_dir}/{domain}_{img_size}_{ds}.pkl", 'rb') as fin:
            self.x, self.y = pickle.load(fin)
            # self.x = ((self.x+1)*127.5).astype(np.uint8)
            # self.x = (self.x*255).astype(np.uint8)
            self.x = torch.FloatTensor(self.x)

        self.num_data = self.x.shape[0]
        
    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]

        return img, label
    
    def __len__(self):
        return self.num_data

    

