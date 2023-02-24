import numpy as np
import torch
import torchvision.transforms as transforms
#import constants as cs
from PIL import Image 
from torch.utils.data import Dataset
import constants

synthetic_images_path = "datasets/synthetic_images/total/images/"
synthetic_labels_path = "datasets/synthetic_images/total/labels/"
synthetic_csv_path = "datasets/synthetic_images/total/label_txt.txt"

target_image_size = (constants.H, constants.W)

class synthetic_data(Dataset):
    def __init__(self, mode = "train", transforms = None, image_path = synthetic_images_path, label_path = synthetic_labels_path, txt =synthetic_csv_path):
        self.mode = mode 
        self.image_path = image_path
        self.label_path = label_path
        self.transforms = transforms
        self.txt = txt
        with open(txt) as f:
            self.image_list = [line.split(',') for line in f.readlines()]
        
        self.val_set = []
        self.train_set = []
        for i in range(len(self.image_list)):
            if (i%21==0):
                self.val_set.append(self.image_list[i][0].replace("\n",""))
            else:
                self.train_set.append(self.image_list[i][0].replace("\n",""))
        
        if(self.mode == "train"):
            self.images = self.train_set
        else:
            self.images = self.val_set

    

    def __getitem__(self, idx):

        image = Image.open(self.image_path + self.images[idx])
        label = Image.open(self.label_path + self.images[idx].replace("S", "L"))
        label = np.asarray(label)
        label_copy = np.copy(label)

        label = self.add_sea(label_copy)


        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        transform = transforms.Compose([transforms.PILToTensor()])
        label = transform(Image.fromarray(label))/255


        if(self.transforms):
            image, label = self.transforms(image, label)
        else:
            #resize to 512x512
            image, label = self.resize(image, label)


        return image, label

    def resize(self, image, label):
        #downsample image using nearest neighbors
        image = transforms.CenterCrop(target_image_size)(image) #transforms.InterpolationMode.BILINEAR)(image)
        label = transforms.CenterCrop(target_image_size )(label)#, transforms.InterpolationMode.NEAREST)(label)
        return image, label 
    
    def __len__(self):
        return len(self.images)
    

    def add_sea(self, label):
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if(sum(label[i][j])==0):
                    label[i][j][0] = 255
        return label





real_images_path = "datasets/real_images/"

real_csv_path = "datasets/real_images/label_txt.txt"



class real_data(Dataset):
    def __init__(self, mode = "train", transforms = None, image_path = real_images_path, txt = real_csv_path):
        self.mode = mode 
        self.image_path = image_path
        self.transforms = transforms
        self.txt = txt

        with open(txt) as f:
            self.image_list = [line.split(',') for line in f.readlines()]
        
        self.val_set = []
        self.train_set = []
        for i in range(len(self.image_list)):

            self.val_set.append(self.image_list[i][0].replace("\n",""))
            self.train_set.append(self.image_list[i][0].replace("\n",""))
        
        if(self.mode == "train"):
            self.images = self.train_set
        else:
            self.images = self.val_set

    

    def __getitem__(self, idx):

        image = Image.open(self.image_path + self.images[idx])


        if(self.transforms):
            image = self.transforms(image)
        else:
            #resize to 512x512
            image = self.resize(image)




        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)


        return image
    



    def resize(self, image):
        #downsample image using nearest neighbors
        image = transforms.CenterCrop([1024, 1024])(image)
        image = transforms.Resize(target_image_size)(image) #transforms.InterpolationMode.BILINEAR)(image)
        #label = transforms.CenterCrop(target_image_size)(label) #transforms.InterpolationMode.NEAREST)(label)
        return image
    
    
    
    def __len__(self):
        return len(self.images)
    
    def getimage(self, idx):

        image = Image.open(self.image_path + self.images[idx])

        
        image = self.resize(image)
        
        image = image.convert('RGBA')
        
        return image




class contrastive_data(Dataset):
    def __init__(self, mode = "train", n_views = None, transforms = None, image_path = real_images_path, txt = real_csv_path):
        self.mode = mode 
        self.image_path = image_path
        self.transforms = transforms
        self.txt = txt
        self.n_views = n_views

        with open(txt) as f:
            self.image_list = [line.split(',') for line in f.readlines()]
        
        self.val_set = []
        self.train_set = []
        for i in range(len(self.image_list)):

            self.val_set.append(self.image_list[i][0].replace("\n",""))
            self.train_set.append(self.image_list[i][0].replace("\n",""))
        
        if(self.mode == "train"):
            self.images = self.train_set
        else:
            self.images = self.val_set

    

    def __getitem__(self, idx):

        image = Image.open(self.image_path + self.images[idx])
        #image = transforms.ToTensor()(image)

        images = []
        if(self.transforms):
        
            for i in range(self.n_views):
                images.append( self.transforms(image) )

        else: 
            images = image
            images  = transforms.Resize(target_image_size, transforms.InterpolationMode.BILINEAR)(images)
            images = transforms.ToTensor()(images)



        return images

    def __len__(self):
        return len(self.images)



