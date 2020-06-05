
import pickle

import numpy as np 

import torch
import torch.nn.functional as F
import torch.utils.data as data


def load_data(domain, data_dir, img_size, num_iters_per_epoch, batch_size):

    # load data
    train_lbl = ImageDataset(data_dir, domain, img_size, ds='train_lbl_1000')
    dev_lbl = ImageDataset(data_dir, domain, img_size, ds='dev_lbl')
    test_lbl = ImageDataset(data_dir, domain, img_size, ds='test_lbl')

    print(f"train lbl dataset num data: {train_lbl.num_data}")
    print(f"dev dataset num data: {dev_lbl.num_data}")
    print(f"test dataset num data: {test_lbl.num_data}")

    train_lbl_sampler = data.RandomSampler(
        train_lbl, replacement=True, num_samples=num_iters_per_epoch*batch_size)
    
    train_lbl = data.DataLoader(
        train_lbl, batch_size=batch_size, num_workers=16,
        sampler=train_lbl_sampler)
    dev_lbl = data.DataLoader(
        dev_lbl, batch_size=batch_size, shuffle=False, num_workers=16)
    test_lbl = data.DataLoader(
        test_lbl, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"train lbl dataset num batches: {len(train_lbl)}")
    print(f"dev lbl dataset num batches: {len(dev_lbl)}")
    print(f"test lbl dataset num batches: {len(test_lbl)}")

    return train_lbl, dev_lbl, test_lbl


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, domain, img_size, ds):

        with open(f"{data_dir}/{domain}_{img_size}_{ds}.pkl", 'rb') as fin:
            self.x, self.y = pickle.load(fin)
            self.x = torch.FloatTensor(self.x)

        self.num_data = self.x.shape[0]
        
    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]

        return img, label
    
    def __len__(self):
        return self.num_data

    

