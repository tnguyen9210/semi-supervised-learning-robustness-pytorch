
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTransform(object):
    def __init__(self,
                 random_horizontal_flip=False, random_crop=False, add_noise=False):
        self.random_horizontal_flip = random_horizontal_flip
        self.random_crop = random_crop
        self.add_noise = add_noise
        
    def __call__(self, x):
        if self.random_horizontal_flip and np.random.random() < 0.5:
            x = torch.flip(x, dims=[3])
        if self.random_crop:
            _, _, h, w = x.shape
            x = F.pad(x, [2,2,2,2], mode="reflect")
            l, t = np.random.randint(0, 4), np.random.randint(0, 4)
            x = x[:,:,t:t+h,l:l+w]
        if self.add_noise:
            pass

        return x


