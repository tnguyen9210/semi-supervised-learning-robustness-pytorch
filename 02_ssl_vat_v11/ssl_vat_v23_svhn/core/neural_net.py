
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, args):
        super(ResNet, self).__init__()
        
        self.top_bn = True
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),

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

        self.linear = nn.Linear(128, 10)
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        if self.top_bn:
            output = self.bn(output)
        return output
        

# def conv3x3(in_channels, out_channels, stride):
#     """ 3x3 convolution with padding """
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)

# def conv1x1(in_channels, out_channels, stride):
#     """ 1x1 convolution without padding"""
#     return nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                      stride=stride, padding=0, bias=False)

# class BasicBlock(nn.Module):
#     def __init__(
#             self, in_channels, out_channels, stride, droprate):
#         super(BasicBlock, self).__init__()

#         self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = conv3x3(out_channels, out_channels, stride=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.relu = nn.ReLU(inplace=True)
#         self.drop = nn.Dropout2d(droprate)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 conv1x1(in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):

#         # hidden = self.conv1(x)
#         # hidden = self.relu(hidden)

#         # hidden = self.conv2(hidden)
#         # hidden = self.drop(hidden)
#         # hidden = self.bn2(hidden + self.shortcut(x))
        
#         hidden = self.relu(self.bn1(self.conv1(x)))
#         hidden = self.drop(hidden)

#         hidden = self.bn2(self.conv2(hidden))
#         hidden = self.relu(hidden + self.shortcut(x))
        
#         return hidden


# class ResNet(nn.Module):
#     def __init__(self, args):
#         super(ResNet, self).__init__()

#         # conv layers ##
#         block = BasicBlock
#         depth = args['resnet_depth']
#         widen_factor = args['resnet_widen_factor']
#         droprate1 = args['resnet_group1_droprate']
#         droprate2 = args['resnet_group2_droprate']
#         droprate3 = args['resnet_group3_droprate']
        
#         num_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert((depth - 4) % 6 == 0)
#         n = (depth - 4) // 6

#         self.conv0 = conv3x3(3, num_channels[0], stride=1)
#         self.group1 = self.make_group(
#             block, n, num_channels[0], num_channels[1], 1, droprate1)
#         self.group2 = self.make_group(
#             block, n, num_channels[1], num_channels[2], 2, droprate2)
#         self.group3 = self.make_group(
#             block, n, num_channels[2], num_channels[3], 2, droprate3)
        
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # fc layers ##
#         in_dim = num_channels[3]
#         hidden_dim1 = args['img_cls_hidden_dim1']
#         droprate1 = args['img_cls_droprate1']
#         num_labels = 10

#         self.img_cls = nn.Linear(in_dim, num_labels)
#         # self.img_cls = nn.Sequential(
#         #     # fc 1
#         #     nn.Linear(in_dim, hidden_dim1),
#         #     nn.BatchNorm1d(hidden_dim1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(droprate1),
#         #     # fc out 
#         #     nn.Linear(hidden_dim1, num_labels))

#         # init weights
#         self.init_weights()
        
#     def forward(self, x):
        
#         # conv layers
#         hidden = self.conv0(x)
#         hidden = self.group1(hidden)
#         hidden = self.group2(hidden)
#         hidden = self.group3(hidden)
#         hidden = self.avg_pool(hidden)
#         hidden = hidden.view(hidden.shape[0], -1)
        
#         # fc layers
#         logit = self.img_cls(hidden)

#         return logit

#     def make_group(
#             self, block, num_blocks, in_channels, out_channels, stride, droprate):
#         conv_group = []
#         conv_group.append(block(in_channels, out_channels, stride, droprate))
#         for i in range(1, num_blocks):
#             conv_group.append(block(out_channels, out_channels, 1, droprate))

#         return nn.Sequential(*conv_group)
    
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2./n))
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.constant_(m.bias, 0)

                
        
