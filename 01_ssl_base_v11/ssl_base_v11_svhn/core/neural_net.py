
import torch
import torch.nn as nn
from torch.autograd import Function

class ImgClassifierSmall(nn.Module):
    def __init__(self, args):
        super(ImgClassifierSmall, self).__init__()

        # Encoder CNN layers
        droprate1 = args['enc_droprate1']
        droprate2 = args['enc_droprate2']
        
        self.encoder = nn.Sequential(
            # cnn 1
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            # maxpool 1
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(droprate1),

            # cnn 2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            # maxpool 2
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(droprate2),
            
            # cnn 3
            nn.Conv2d(128, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # FC layers
        in_dim = 128
        hidden_dim1 = args['img_cls_hidden_dim1']
        hidden_dim2 = args['img_cls_hidden_dim2']
        droprate1 = args['img_cls_droprate1']
        droprate2 = args['img_cls_droprate2']
        num_labels = 10
        
        self.img_cls = nn.Sequential(
            # fc 1
            nn.Linear(in_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(True),
            nn.Dropout2d(droprate1),
            # # fc 2
            # nn.Linear(hidden_dim1, hidden_dim2),
            # nn.BatchNorm1d(hidden_dim2),
            # nn.ReLU(True),
            # nn.Dropout2d(droprate2),
            # fc out
            nn.Linear(hidden_dim1, num_labels)
        )
                  
    def forward(self, x):

        hidden = self.encoder(x)
        hidden = hidden.view(hidden.shape[0], -1)
        logit = self.img_cls(hidden)
        # logit = self.bn(logit)
        
        return logit

    
class ImgClassifierLarge(nn.Module):
    def __init__(self, args):
        super(ImgClassifierLarge, self).__init__()

        self.encoder = nn.Sequential(
            # cnn 1
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            # maxpool 1
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

            # cnn 2
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            # maxpool 2
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),
            
            # cnn 3
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, 10)
        self.bn = nn.BatchNorm1d(10)
                  
    def forward(self, x):

        hidden = self.encoder(x)
        hidden = hidden.view(hidden.shape[0], -1)
        logit = self.bn(self.fc(hidden))
        
        return logit



    
