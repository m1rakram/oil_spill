''' Script to train the Segmentation model'''

from re import A
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datasets.dataset_retrieval import synthetic_data, real_data
from models.segformer_utils.model import SegFormer, mit_b0, SegFormerHead
from datasets.augmentations import CropImage, Syn_Augmentation
from loss import DiceLoss
import numpy as np
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
import tqdm
import os
import torch
from utils import poly_lr_scheduler, new_metrics, colorize_image
import math
import sys
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

curr_epoch = 0
use_pretrained = False
num_of_classes_seg = 3      #5 classes including null
save_model_path = "checkpoints/"    #checkpoints to be stored
pth_name =  "synthetic_v2.pth"




def val(backbone, head, data_seg_val, loss_function, writer, epoch):
    


    data_iterator = enumerate(data_seg_val)     #take batches
    metric = new_metrics(num_of_classes_seg)
    with torch.no_grad():

        head.eval()    #switch model to evaluation mode
        backbone.eval()
        tq = tqdm.tqdm(total=len(data_seg_val))
        tq.set_description('Validation:')
        
        total_loss = 0

        for _, batch in data_iterator:
            #forward propagation
            image, label = batch
            pred = head(backbone(image.cuda()))
            loss = loss_function(pred.cuda(), label.cuda())
            pred = pred.softmax(dim=1)
            metric.update(pred.cuda(), label.cuda())

            total_loss += loss.item()
            tq.update(1)


    ious, miou = metric.compute_iou()
    #acc, macc = metric.compute_pixel_acc()

    
    writer.add_scalar("Validation mIoU", miou, epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_seg_val), epoch)


    tq.close()
    print("IoU",ious)
    print("mIoU",miou)

    return miou





def train(backbone, head, dataloader_seg, dataseg_val, optimizer, loss_func, max_num_epochs, current_epoch, learning_rate):
    
    
    writer = SummaryWriter(comment = "Segmentation")
    max_miou = 0
    val(backbone, head, dataseg_val, loss_func, writer, current_epoch)
    
    
    for epoch in range(current_epoch, max_num_epochs):
        
        
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=50)
        seg_iterator = enumerate(dataloader_seg)

        tq = tqdm.tqdm(total=len(dataloader_seg))
        #tq.set_description('epoch %d' % (epoch))
        tq.set_description('epoch %d, lr %f' % (epoch, lr)) # Print epoch and learning rate 
        

        running_loss = 0
        

        metrics = new_metrics(num_of_classes_seg, 4)
        backbone.train()
        head.train()
        
        for it, batch in seg_iterator:
            #image, label, eedt = batch
            optimizer.zero_grad()
            image, label = batch
            

            pred = head(backbone(image.cuda()))
            loss = loss_func(pred.cuda(), label.cuda())

                            
            pred = pred.softmax(dim=1)
            metrics.update(pred.cuda(), label.cuda())
            

            if(math.isnan(loss)):

                sys.exit("Error: Nan Loss, exiting training")
            
            #print(pred)
            #sys.exit("I said to exit")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            tq.update(1)

            tq.set_postfix(loss='%.6f' % (loss))
            
            
        
        ious, miou = metrics.compute_iou()

        writer.add_scalar("Loss", running_loss/len(dataloader_seg), epoch)
        writer.add_scalar("mIoU", miou, epoch)


        tq.close()

        print("IoU",ious)
        print("mIoU",miou)

            
        val_miou = val(backbone, head, dataseg_val, loss_func, writer, epoch)

    

        # Saving in the checkpoint also the epoch number and the optimizer to resume them
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict_head': head.state_dict(),
            'state_dict_backbone':backbone.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saving the model " + save_model_path)



    
    


def main():

    #sequence

    # use dataset from ai4mars
    
    syn_train = synthetic_data("train", transforms=Syn_Augmentation())
    real_train = real_data("train")
    syn_test = synthetic_data("val")


    #dataloaders for AI4Mars (segmentation) dataset 

    data_syn_train = DataLoader(
        syn_train,
        batch_size = 6,
        shuffle = True,
        num_workers = 2

    )

    print(len(data_syn_train))

    data_real_train = DataLoader(
        real_train,
        batch_size = 1,
        shuffle = True,
        num_workers = 1
    )
    

    data_syn_test = DataLoader(
        syn_test,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )



    backbone = mit_b0().cuda()
    head = SegFormerHead(in_channels=[32, 64, 160, 256], num_classes=3, embed_dim=512).cuda()


    curr_epoch=0
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr = learning_rate)

    if use_pretrained:
        print('load model from %s ...' % (save_model_path+pth_name))
        checkpoint = torch.load(save_model_path+pth_name)
        backbone.load_state_dict(checkpoint['state_dict_backbone'])
        head.load_state_dict(checkpoint['state_dict_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])


        print("epoch trained from checkpoint: " + str(checkpoint['epoch']))
        curr_epoch = checkpoint['epoch']
        print('Done!')
    #print(model.aspp.stages.c3.conv.weight)
    max_epochs = 50




    loss = DiceLoss()

    #change use prettrained
    train(backbone, head, data_syn_train, data_syn_test, optimizer, loss, max_epochs, curr_epoch, learning_rate)

if __name__=='__main__':
    main()
