import torch
import torch.nn as nn
from torch.nn.functional import interpolate, relu, dropout
import constants

target_image_dim = constants.IMAGE_SIZE

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners

        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.rebuild_output_layer_(num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def rebuild_output_layer_(self, num_classes):
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        feats_hw = x[0].shape[2:]
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        #print(1, x.shape)
        x = [interpolate(xi, size=feats_hw, mode='bilinear', align_corners=self.align_corners)
             for xi in x[:-1]] + [x[-1]]
        #print(2, x.shape)
        x = self.linear_fuse(torch.cat(x, dim=1))
        #print(3, x.shape)
        x = self.bn(x)
        #print(4, x.shape)
        x = relu(x, inplace=True)
        #print(5, x.shape)
        x = dropout(x, p=self.dropout_p, training=self.training)
        #print(6, x.shape)
        x = self.linear_pred(x)
        #print(7, x.shape)
        x = interpolate(x, size=target_image_dim, mode='bilinear', align_corners=self.align_corners)
        return x


class Classification_head(nn.Sequential):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners

        self.layer_1 = nn.Conv2d(self.in_channels[0], 1, (1,1))
        self.layer_2 = nn.Conv2d(self.in_channels[1], 1, (1,1))
        self.layer_3 = nn.Conv2d(self.in_channels[2], 4, (1,1))
        self.layer_4 = nn.Conv2d(self.in_channels[3], 4, (1,1))

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(512, 256, (1,1))
        self.bn = nn.BatchNorm1d(21760, eps=1e-5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(400, num_classes)



    def forward(self, x):
        
        x0 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_1(x[0]))))))
        x1 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_2(x[1]))))))
        x2 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_3(x[2]))))))
        x3 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_4(x[3]))))))

        x = torch.cat([x0,x1,x2,x3], dim= 1)

        x = self.linear(x)
        
        return x


class MLP_head(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        
        self.layer_1 = nn.Conv2d(self.in_channels[0], 2, (1,1))
        self.layer_2 = nn.Conv2d(self.in_channels[1], 2, (1,1))
        self.layer_3 = nn.Conv2d(self.in_channels[2], 8, (1,1))
        self.layer_4 = nn.Conv2d(self.in_channels[3], 8, (1,1))

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm1d(256, eps=1e-5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(800, 256)
        self.out_layer = nn.Linear(256, 128)



    def forward(self, x):
        
        x0 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_1(x[0]))))))
        x1 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_2(x[1]))))))
        x2 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_3(x[2]))))))
        x3 = self.flatten(self.maxpool(self.maxpool(self.maxpool(self.relu(self.layer_4(x[3]))))))

        x = torch.cat([x0,x1,x2,x3], dim= 1)

        x = self.linear(x)
        x = self.relu(self.bn(x))
        x = self.out_layer(x)
        
        return x

