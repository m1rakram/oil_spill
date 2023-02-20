import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from datasets.augmentations import CropImage

from models.segformer_utils.model import SegFormer, mit_b3, mit_b0, SegFormerHead
from datasets.dataset_retrieval import real_data, synthetic_data

from utils import colorize_image, metrics_correction, new_metrics
from torch.utils.data import DataLoader, ConcatDataset
from utils import Metrics, print_results

import numpy as np
import constants
import tqdm
import time
torch.cuda.empty_cache()

num_of_classes_seg = constants.NUM_CLASSES_SEG      #5 classes including null
save_model_path = "checkpoints/"    #checkpoints to be stored
pth_name =  "deeplab_decouplenet.pth"











def evaluate_model(args, model, data_test, dataset):

    
    model.eval()
    metric = new_metrics(num_of_classes_seg, 4)         #evaluation metrics for segmentation 
    total_f1 = 0                                    #storage for f1 score
    sig = nn.Sigmoid()                              #sigmoid function for classification


    # if task is segmentation task has 1 output
    if(args.task == "segmentation"):
        tq = tqdm.tqdm(total=len(dataset))
        tq.set_description('Segmentation validation on ' + args.dataset + ':')
        
        
        for _, batch in enumerate(data_test):
            image, label = batch
            with torch.no_grad():    
                            #get batches for segmentation datase

                pred_seg = model(image.cuda())      #inference
            pred_seg = pred_seg.softmax(dim=1)              #softmax laye
            
            metric.update(pred_seg.cuda(), label.cuda())    #update metric class for further estimation
            
            tq.update(1)

        tq.close()


    f1_score = total_f1*100/len(dataset)
    ious, miou = metric.compute_iou()
    
    print_results(ious, miou, f1_score)   #custom print function for table wise printing



def visualize_sample(args, model, dataset, image_id):
    model.eval()
    

    if(args.dataset != "merged"):
        image = dataset[image_id].unsqueeze(dim = 0)
        
        if(args.task == "segmentation"):
            with torch.no_grad():
                prediction_seg = model(image.cuda())
                prediction_seg = prediction_seg.softmax(dim=1)
                        



    complete_image = dataset.getimage(image_id)
    
    col = colorize_image(complete_image, prediction_seg.cpu().detach())
    
    col.show()
    








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = "segformer_b0_v2", help='trained model keyword')
    parser.add_argument('--dataset',  default = "real", help='dataset for validation')
    parser.add_argument('--model_path', default = "synthetic.pth", help='path name')
    parser.add_argument('--task', default = "segmentation", help='task type')
    parser.add_argument('--visual_flag', type = bool, default = True, help = 'To know if purpose is batch testing or visualisation')
    parser.add_argument('--image_id', type = int, default=201, help='ID of image')
    args = parser.parse_args()

    checkpoint = torch.load(save_model_path + args.model_path)
    

        


    if(args.model == "segformer_b0_v2"):
        backbone = mit_b0()
        head = SegFormerHead(in_channels=[32, 64, 160, 256], num_classes=num_of_classes_seg, embed_dim=512)
        
        backbone.load_state_dict(checkpoint['state_dict_backbone'])
        head.load_state_dict(checkpoint['state_dict_head'])
        model = nn.Sequential(backbone, head).cuda()
        print('Done! Model is ready!')
    

    
    if(args.dataset == "real"):
        dataset = real_data("val")
    
    elif(args.dataset == "synthetic"):
        dataset = synthetic_data("val")
    
        
    
    
    
    data_test = DataLoader(
            dataset,
            batch_size=1)

    if(args.visual_flag):
        visualize_sample(args, model, dataset, args.image_id)
    else:
        evaluate_model(args, model, data_test, dataset)


if __name__=='__main__':
    main()
            