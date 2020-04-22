
import pickle

import torch
import torch.utils.data as data


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, domain, img_size, ds):
        
        with open(f"{data_dir}/{domain}_{img_size}_{ds}.pkl", 'rb') as fi:
            x, y = pickle.load(fi)
            self.x = torch.FloatTensor(x).expand(x.shape[0], 3, img_size, img_size)
            self.y = torch.LongTensor(y)

        self.num_data = self.x.shape[0]
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.num_data

    
def load_data(domain, data_dir, img_size, batch_size):

    # load data
    train_unlbl = ImageDataset(data_dir, domain, img_size, ds='train_unlbl_1000')
    train_lbl = ImageDataset(data_dir, domain, img_size, ds='train_lbl_1000')
    dev_lbl = ImageDataset(data_dir, domain, img_size, ds='dev_lbl')
    test_lbl = ImageDataset(data_dir, domain, img_size, ds='test_lbl')

    print(f"train unlbl dataset num data: {train_unlbl.num_data}")
    print(f"train lbl dataset num data: {train_lbl.num_data}")
    print(f"dev dataset num data: {dev_lbl.num_data}")
    print(f"test dataset num data: {test_lbl.num_data}")

    train_lbl_sampler = data.RandomSampler(
        train_lbl, replacement=True, num_samples=int(train_unlbl.num_data/batch_size*32))

    train_unlbl = data.DataLoader(
        train_unlbl, batch_size=batch_size, shuffle=True, num_workers=8)
    train_lbl = data.DataLoader(
        train_lbl, batch_size=32, sampler=train_lbl_sampler, num_workers=8)
    dev_lbl = data.DataLoader(
        dev_lbl, batch_size=batch_size, shuffle=False, num_workers=8)
    test_lbl = data.DataLoader(
        test_lbl, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"train unlbl dataset num batches: {len(train_unlbl)}")
    print(f"train lbl dataset num batches: {len(train_lbl)}")
    print(f"dev lbl dataset num batches: {len(dev_lbl)}")
    print(f"test lbl dataset num batches: {len(test_lbl)}")

    return train_lbl, train_unlbl, dev_lbl, test_lbl

    
