
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self, args):
        super(LeNet5, self).__init__()

        widen_factor = args['lenet_widen_factor']
        nchannels = [32*widen_factor, 64*widen_factor]
        droprate = args['lenet_droprate']
        
        # cnn layer 1
        self.cnn1 = nn.Sequential()
        self.cnn1.add_module('conv1', nn.Conv2d(3, nchannels[0], kernel_size=5, stride=1, padding=2))
        # self.cnn1.add_module('bn1', nn.BatchNorm2d(nchannels[0]))
        self.cnn1.add_module('relu1', nn.LeakyReLU(0.1, inplace=True))
        self.cnn1.add_module('drop1', nn.Dropout2d(droprate))
        self.cnn1.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        
        # cnn layer 2
        self.cnn2 = nn.Sequential()
        self.cnn2.add_module('conv2', nn.Conv2d(nchannels[0], nchannels[1], kernel_size=5, stride=1, padding=2))
        # self.cnn2.add_module('bn2', nn.BatchNorm2d(nchannels[1]))
        self.cnn2.add_module('relu2', nn.LeakyReLU(0.1, inplace=True))
        self.cnn2.add_module('drop2', nn.Dropout2d(droprate))
        self.cnn2.add_module('pool2', nn.MaxPool2d(kernel_size=2))

        in_dim = nchannels[1] * 7 * 7
        hidden_dim = args['fc_hidden_dim']
        droprate = args['fc_droprate']
        num_labels = 10
        
        # fc layer 1
        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(in_dim, hidden_dim))
        # self.fc.add_module('bn1', nn.BatchNorm1d(hidden_dim))
        self.fc.add_module('relu1', nn.LeakyReLU(0.1, True))
        self.fc.add_module('drop1', nn.Dropout(droprate))
        # fc out 
        self.fc.add_module('fc2', nn.Linear(hidden_dim, num_labels))
        

    def forward(self, x):
        hidden = self.cnn1(x)
        hidden = self.cnn2(hidden)
        hidden = hidden.view(hidden.shape[0], -1)
        logit = self.fc(hidden)
        return logit
    
    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


                
        
