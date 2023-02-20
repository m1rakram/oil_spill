import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from datasets.dataset_retrieval import real_data, synthetic_data
from datasets.dataset_retrieval import contrastive_data
from datasets.augmentations import GaussianBlur, CropImage, Syn_Augmentation, strongTransform, weakTransform

from models.segformer_utils.model import SegFormer, mit_b0, SegFormerHead
from models.segformer_utils.heads import MLP_head

from loss import DiceLoss
from domain_adaptation.SimCLR.info_nce import info_nce_loss
from domain_adaptation.SimCLR.ST_utils import define_mix_type, update_ema_variables, define_pixel_weight
from utils import poly_lr_scheduler, Metrics, print_results


import numpy as np
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
import tqdm
import os
import torch
import constants
import math
import sys
torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_of_classes_seg = constants.NUM_CLASSES_SEG
image_h = constants.IMAGE_H
image_tuple = constants.IMAGE_SIZE

save_model_path = "checkpoints/"    #checkpoints to be stored
init_name =  "pretrained.pth"
pth_name = "domain_adapted.pth"

curr_epoch = 0
use_pretrained = False


def upload_pretrained_weights(model, pretrained_model):
    
    for model_param, pretrained_param in zip(model.parameters(), pretrained_model.parameters()):
            model_param.data[:] = pretrained_param.data[:]
    return model

def val(backbone, model, data_seg_val, loss_function, writer, epoch):
    


    data_iterator = enumerate(data_seg_val)     #take batches
    metric = Metrics(num_of_classes_seg, 0)
    with torch.no_grad():

        model.eval()    #switch model to evaluation mode
        backbone.eval()
        tq = tqdm.tqdm(total=len(data_seg_val))
        tq.set_description('Validation:')
        
        total_loss = 0

        for _, batch in data_iterator:
            #forward propagation
            image, label = batch
            pred = model(backbone(image.cuda()))
            
            loss = loss_function(pred.cuda(), label.cuda())
            pred = pred.softmax(dim=1)
            metric.update(pred.cuda(), label.cuda())

            total_loss += loss.item()
            tq.update(1)


    ious, miou = metric.compute_iou()

    
    writer.add_scalar("Validation mIoU", miou, epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_seg_val), epoch)


    tq.close()
    print_results(ious, miou, 0)


    return miou



def train(backbone, mlp, head, mt_backbone, mt_head, syn_dataset, con_dataset, st_dataloader, ai_dataset, loss_dice, loss_CE, loss_CE_cons, optimizer, curr_epoch, max_epoch):
    
    
    writer = SummaryWriter(comment = "Customized")
    
    val(backbone, head, ai_dataset, loss_dice, writer, curr_epoch )
   
    

    for epoch in range(curr_epoch, max_epoch):
        
        backbone.train()
        mlp.train()
        head.train()
        
        
        metrics = Metrics(num_of_classes_seg, 0)

        consistency_weight = 1.0
        print("arrived")

        # iterator for contrastive learning
        con_iterator = enumerate(con_dataset)

        # iterator for synthetic dataset
        syn_iterator = enumerate(syn_dataset)

        # iterator for self-training
        st_iterator = enumerate(st_dataloader)
        
        tq = tqdm.tqdm(total=len(syn_dataset))
        tq.set_description('epoch %d' % (epoch))
        
        scaler = amp.GradScaler()

        for it, batch in syn_iterator:
            
            optimizer.zero_grad() 
            

            #supervised learning by using synthetic dataset
            image_syn, label_syn = batch

            with amp.autocast():
                pred_syn_head = head(backbone(image_syn.cuda()))
                loss_syn_sup = loss_dice(pred_syn_head.cuda(), label_syn.cuda())
            
            
            
            pred_syn_head = pred_syn_head.softmax(dim=1)
            metrics.update(pred_syn_head.cuda(), label_syn.cuda())

            try:
                _, batch = next(st_iterator)
            except:
                st_iterator = enumerate(st_dataloader)
                _, batch = next(st_iterator)


            # Self training starts:
            # code has been taken from https://github.com/WilhelmT/ClassMix

            image_st = batch

            image_mt = weakTransform(image_st)
            
            with amp.autocast():
                with torch.no_grad():  

                    mt_pred = mt_head(mt_backbone(image_mt.cuda()))

                mt_pred = weakTransform(mt_pred) 

                
                softmax_u_w = torch.softmax(mt_pred.detach(), dim=1)
                max_probs, argmax_u_w = torch.max(softmax_u_w, dim=1)

                # define mixmask
                MixMask = define_mix_type(image_st, max_probs, argmax_u_w, "class")

                # transform the label from teacher and image for student with the same transform
                image_s, prediction_pseudo = strongTransform(image_st.cuda(), softmax_u_w.cuda(), MixMask)  

                #inference on student
                pred_st = head(backbone(image_s.cuda()))

                max_probs, pseudo_label = torch.max(prediction_pseudo, dim=1)

                #pixel_weight = define_pixel_weight(max_probs, pseudo_label, None, epoch*len(syn_dataset) + it)

                loss_st = consistency_weight * loss_CE(pred_st, pseudo_label)
            
            #loss = loss_syn_sup + loss_st

                        

            # contrastive learning by using real datasets
            try:
                _, batch = next(con_iterator)
            except Exception as e:
                print(e)
                con_iterator = enumerate(con_dataset)
                _, batch = next(con_iterator)
            
            images = batch
            images = torch.cat(images, dim=0) 



            with amp.autocast():
                features = mlp(backbone(images.cuda()))
                logits, labels = info_nce_loss(features)
                loss_con = loss_CE_cons(logits, labels)
            
            #scaler.scale(loss_con).backward()
            
            loss = loss_syn_sup + loss_st + loss_con
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update ema model
            alpha_teacher = 0.99
            mt_backbone, mt_head = update_ema_variables( ema_backbone = mt_backbone,ema_head = mt_head, backbone = backbone, head =head, alpha_teacher=alpha_teacher, iteration=epoch*len(syn_dataset) + it)

            
            tq.update(1)
            tq.set_postfix(loss_st='%.6f, loss_seg = %.6f, loss_con = %.6f' % (loss_st.item(), loss_syn_sup.item(), loss_con.item()))

        tq.close()

        ious, miou = metrics.compute_iou()

        print_results(ious, miou, 0)

        val(backbone, head, ai_dataset, loss_dice, writer, epoch )


        checkpoint = {
            'epoch': epoch + 1,
            'state_dict_backbone': backbone.state_dict(),
            'state_dict_ema_backbone': mt_backbone.state_dict(),
            'state_dict_head': head.state_dict(),
            'state_dict_ema_head': mt_head.state_dict(),
            'state_dict_mlp': mlp.state_dict(),

            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)
        




def main():


    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms_con = transforms.Compose([transforms.RandomResizedCrop(size = image_h),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              GaussianBlur(kernel_size=int(0.1 * image_h)),
                                              transforms.ToTensor()
                                              ])


    syn_train = synthetic_data("train", transforms = Syn_Augmentation(image_h))
    con_train = contrastive_data(n_views = 2, transforms=data_transforms_con )
    real_train = real_data("train", transforms=CropImage())

    syn_val = synthetic_data("val")




    data_syn_train = DataLoader(
        syn_train,
        batch_size = 4,
        shuffle = True,
        num_workers=4
    )

    contrastive_dataloader = DataLoader(
        con_train,
        batch_size = 4,
        shuffle = True,
        num_workers = 2,
        drop_last=False
    )


    data_syn_val = DataLoader(
        syn_val,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    data_real_train = DataLoader(
        real_train,
        batch_size=2,
        num_workers=1,
        shuffle=True
    )




    backbone = mit_b0()
    head = SegFormerHead(in_channels=[32, 64, 160, 256], num_classes=3, embed_dim=512)
    mlp = MLP_head(in_channels=[32, 64, 160, 256])

    full_model = SegFormer(backbone, head)
    
    # mean teacher architecture
    mt_backbone = mit_b0()
    mt_head = SegFormerHead(in_channels=[32, 64, 160, 256], num_classes=3, embed_dim=512)

    for param in mt_backbone.parameters():
        param.detach_()
    for param in mt_head.parameters():
        param.detach_()
    
    
    curr_epoch=0
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(mlp.parameters()) + list(head.parameters()), lr = learning_rate)

    if use_pretrained:
        #if training should resume from checkpoint
        print('load model from %s ...' % (save_model_path+pth_name))
        checkpoint = torch.load(save_model_path+pth_name)

        backbone.load_state_dict(checkpoint['state_dict_backbone'])
        head.load_state_dict(checkpoint['state_dict_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        mt_backbone.load_state_dict(checkpoint['state_dict_ema_backbone'])
        mt_head.load_state_dict(checkpoint['state_dict_ema_head'])
        mlp.load_state_dict(checkpoint['state_dict_mlp'])

        print("epoch trained from checkpoint: " + str(checkpoint['epoch']))
        curr_epoch = checkpoint['epoch']
        print('Done!')

    else:
        #if training starts for the first time 

        print("initializing the weights")
        checkpoint = torch.load(save_model_path + init_name)
        head.load_state_dict(checkpoint['state_dict_head'])
        backbone.load_state_dict(checkpoint['state_dict_backbone'])
        
        full_model = SegFormer(backbone, head)

        
        #upload weights to the  parts
        head = upload_pretrained_weights(head, full_model.decode_head)
        backbone = upload_pretrained_weights(backbone, full_model.backbone)
        mt_head = upload_pretrained_weights(mt_head, full_model.decode_head)
        mt_backbone = upload_pretrained_weights(mt_backbone, full_model.backbone)
        del full_model

    if torch.cuda.is_available():
        backbone.cuda()
        mlp.cuda()
        head.cuda()
        mt_backbone.cuda()
        mt_head.cuda()
    max_epochs = 50



    
    loss_CE = nn.CrossEntropyLoss(ignore_index=0)
    loss_dice = DiceLoss()
    loss_CE_cons = nn.CrossEntropyLoss()

    train(backbone, mlp, head, mt_backbone, mt_head, 
          data_syn_train, contrastive_dataloader, data_real_train , data_syn_val, loss_dice, loss_CE, loss_CE_cons, optimizer, curr_epoch, max_epochs)
    

    

if __name__=='__main__':
    main()