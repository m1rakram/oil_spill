import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image 


import constants
import torch.optim as optim

from PIL import ImageFont
from PIL import ImageDraw

num_of_classes_seg = constants.NUM_CLASSES_SEG
image_size = constants.IMAGE_H    # as images are squared no need for tuple
image_h = constants.IMAGE_H
n_classes = constants.NUM_CLASSES_SEG
MAP_COLORS = [[0, 0, 255, 30],   # Sea 
                [0, 255, 0, 80],   # platform 
                [255, 0, 0, 80]]    # oil




def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	# if iter % lr_decay_iter or iter > max_iter:
	# 	return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	optimizer.param_groups[0]['lr'] = lr
	return lr
	# return lr

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter, learning_rate_D, num_steps, power):
    lr = lr_poly(learning_rate_D, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10






def colorize_image(image, label, image_name = "Image name not specified"):
    label = label.squeeze()

    colorized = np.zeros((image_size,image_size, 4)) 


    label = softmax_layer(label)
    for class_id in range(n_classes):

        colormask = MAP_COLORS[class_id]
        equality = np.equal(label, class_id)
        colorized[equality, :] = colormask
    
    #put image and label together to understand better
    PIL_image = Image.fromarray(colorized.astype('uint8'), 'RGBA')
    final1 = Image.new('RGBA', PIL_image.size)
    final1 = Image.alpha_composite(final1, image)
    final1 = Image.alpha_composite(final1, PIL_image)

    #draw = ImageDraw.Draw(final1)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    #font = ImageDraw.getfont()
    # draw.text((x, y),"Sample Text",(r,g,b))
    #draw.text((image_size/2-10, 0), image_name ,(255,255,255)) #, font=font)
    
    return final1

    


def softmax_layer(label):
    # this creates 1 channel label image with classes after softmax
    monochannel_label = np.zeros((image_size, image_size), dtype='int')
    
    #for simplicity change dimension orders
    label = label.permute(1, 2, 0)
    
    for h in range(image_size):
        for w in range(image_size):
            #apply softmax to each pixel, to define class id
            monochannel_label[h][w] = np.argmax(F.softmax(label[h][w], dim = 0))
    return monochannel_label 







class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device = 'cuda')

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        target = target.argmax(dim = 1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[:-1].mean().item()
        ious *= 100
        miou *= 100
        return np.round(ious.cpu().numpy(), 2), round(miou, 2)

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[:-1].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2), round(mf1, 2)

    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[:-1].mean().item()
        acc *= 100
        macc *= 100
        return np.round(acc.cpu().numpy(), 2), round(macc, 2)







def print_results(ious, miou, f1):
    # for i in range(len(ious)):
    #     if (math.isnan(ious[i])):
    #             ious[i] = 0

    # if (math.isnan(miou)):
    #     miou = 0   


    data = {'platforms': [ious[1]], 'oil': [ious[2]], 'miou': [miou]}
    df = pd.DataFrame(data = data)
    print("\n \n \n             ------- Results -------")
    print(df)
    print()
    print(data)
    print()
    print()


def metrics_correction(logits):
    for logit in range(len(logits)):
        for cl in range(len(logits[logit])):

            class_sum = np.sum(logits[logit][cl].cpu().detach().numpy())
            if(class_sum< 10*image_h*image_h/100):
                logits[logit][cl] = torch.zeros(logits[logit][cl].shape)
    return logits.cuda()



class new_metrics:
    def __init__(self, num_classes, ignore_label=0):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.iou = np.zeros(3, dtype=float)
        self.iou_count = np.zeros(3, dtype=int)
        self.image_count = 0
        self.mious = 0.0

    def update(self, preds, targets):
        
        for pred, target in zip(preds, targets):
            
            pred = pred.argmax(dim=0)
            target = target.argmax(dim = 0)
            iou_flag = 0
            iou = np.zeros(4, dtype=float)
            

            for cl in range(0,3):
                
                pred_logits = np.squeeze(np.asarray((pred == cl).cpu().detach().numpy(), dtype=int))
                label_logits =  np.squeeze(np.asarray((target == cl).cpu().detach().numpy(), dtype=int))
                                
                if(np.sum(label_logits)>0 or np.sum(pred_logits)>0):        
                    equality = np.sum(pred_logits*label_logits, dtype = float)
                    
                    non_equality = np.sum((label_logits - pred_logits) == 1, dtype = float)
                    if(non_equality+equality!=0):
                        iou[cl] = equality/(equality + non_equality)
                    
                    else:
                        iou[cl] = 0.0

                    self.iou[cl] += iou[cl]
                    self.iou_count[cl] +=1
                    iou_flag +=1
            
            if(iou_flag > 0):
                
                miou = np.sum(iou)/iou_flag
                self.mious += miou
                self.image_count += 1

                        

    
    def compute_iou(self):
        
        iou = self.iou/self.iou_count*100
        miou = self.mious/self.image_count*100

        return np.round(iou, 2), np.round(miou, 2)