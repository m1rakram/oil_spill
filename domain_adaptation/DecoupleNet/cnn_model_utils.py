'''
This Scipt is intended to create seperate modules and networks

ClassHead and Spinal is for image classification
all others are for image segmentation


'''




import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
import models.resnet as resnet
import torch.nn.functional as F
import constants


path_to_resnet50 = "models/r50_2x_sk1.pth"
image_size = constants.IMAGE_SIZE





class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        )
    
    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)




#Deeplab v3

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):

        # for troubleshooting
        """         l = []
        for stage in self.stages.children():
            print(stage(x).shape)
            l.append(stage(x))
        h = torch.cat(l,dim=1)
        print(h.shape)
        return h """
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h




class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.ReLU())



class Deep_lab(nn.Sequential):

    def __init__(self, n_class):
        super(Deep_lab, self).__init__()

        #self.add_module("bn", nn.BatchNorm2d(4096))
        #self.add_module("resnet_backbone", _Backbone)
        self.add_module("aspp", _ASPP(2048, 256, [6, 12, 18]))
        concat_ch = 256*(3+2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))

        #self.add_module("sigmoid", nn.Sigmoid())
        

class deeplab_head(nn.Sequential):

    def __init__(self, n_class):
        super(deeplab_head, self).__init__()

        self.add_module("fc2", nn.Conv2d(256, n_class, kernel_size=1))
        self.add_module("upsamp", upsample())
        #self.add_module("sigmoid", nn.Sigmoid())



class upsample(nn.Module):

    def __init__(self):
        super(upsample, self).__init__()
        
    def forward(self, image):
        transform = nn.Upsample(size = image_size, mode = 'bilinear')
        image = transform(image)
        return image

