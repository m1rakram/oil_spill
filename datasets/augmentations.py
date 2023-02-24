from ctypes import resize
import torch.nn as nn
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import normalize
import constants


image_h = constants.IMAGE_H
image_w = constants.IMAGE_W
image_size = constants.IMAGE_SIZE





class CropImage(object):


    def __init__(self, output_size = image_h):
        self.output_size = output_size

    def __call__(self, image):
        
        # randomize operations
        rand_num = np.random.uniform(0.0,1.0)
        p = 0.3

        if(rand_num>p):
            image = transforms.Resize((self.output_size, self.output_size), transforms.InterpolationMode.BILINEAR)(image)

        elif(rand_num>0.6):
            posx = np.random.randint(0, 1024 - image_h)
            posy = np.random.randint(0, 1024 - image_w)
            image = F.crop(image, posx, posy, image_h, image_w)
        
        else:
            image = transforms.Resize((1024, 1024), transforms.InterpolationMode.BILINEAR)(image)
            image = transforms.RandomCrop((self.output_size, self.output_size))(image)

        
        
        p=0.5
        rand_num = np.random.uniform(0.0,1.0)
        if(rand_num>p):
            image = F.rotate(image, 90)
        
        rand_num = np.random.uniform(0.0,1.0)
        #random horizontal flip
        if(rand_num > p):
            image = F.hflip(image)

        rand_num = np.random.uniform(0.0,1.0)
        brightness = np.random.uniform(0.8,1.5)

        #random horizontal flip
        if(rand_num > p):
            image = F.adjust_brightness(image, brightness)



        return image




class Syn_Augmentation(object):


    def __init__(self, output_size = image_h):
        self.output_size = output_size

    def __call__(self, image, label):
        
        # randomize operations
        rand_num = np.random.uniform(0.0,1.0)
        p = 0.3

        
        if(rand_num>p):
            image = transforms.Resize((self.output_size, self.output_size), transforms.InterpolationMode.BILINEAR)(image)
            label = transforms.Resize((self.output_size, self.output_size), transforms.InterpolationMode.NEAREST)(label)
        
        elif(rand_num>0.6):
            posx = np.random.randint(0, 680 - image_h)
            posy = np.random.randint(0, 680 - image_w)
            image = F.crop(image, posx, posy, image_h, image_w)
            label = F.crop(label, posx, posy, image_h, image_w)

        else:
            angle  = np.random.randint(-30, 30)
            image  = F.center_crop(F.rotate(image, angle), [image_h, image_w])
            label  = F.center_crop(F.rotate(label, angle), [image_h, image_w])

        
        rand_num = np.random.uniform(0.0,1.0)
        p = 0.5
        #random horizontal flip
        if(rand_num > p):
            image = F.hflip(image)
            label = F.hflip(label)

        rand_num = np.random.uniform(0.0,1.0)
        brightness = np.random.uniform(0.9,1.5)

        rand_num = np.random.uniform(0.0,1.0)
        if(rand_num>p):
            image = F.rotate(image, 90)
            label = F.rotate(label, 90)

        #random horizontal flip
        if(rand_num > p):
            image = F.adjust_brightness(image, brightness)

        rand_num = np.random.uniform(0.0,1.0)

        if(rand_num > p):
            image = image + torch.randn(image.size()) * 2.0     # 1 is std and 0 is mean
        return image, label





class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img




def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target






def strongTransform(image, pseudo_label, mixmask):
    
    # add mix transform to image and label
    image, _= mix(mask = mixmask, data = image, target = None)
    pseudo_label, _= mix(mask = mixmask, data = pseudo_label, target = None)
    
    # add color jitter only to image
    jitter_rand = np.random.uniform(0.5,1.0)
    jittertransform = transforms.ColorJitter(jitter_rand, jitter_rand, jitter_rand, 0.2)
    image = jittertransform(image)
    
    # add gaussian nois to only image
    gauss_rand = random.randint(0, 1)
    if(gauss_rand):
        image = image + torch.randn(image.size()).cuda() * 1.0

    # flip image and label
    flip_rand = random.randint(0, 1)
    if(flip_rand):
        image = F.hflip(image)
        pseudo_label = F.hflip(pseudo_label)

    return image, pseudo_label


def weakTransform(image):
    image = F.hflip(image)
    return image
    


