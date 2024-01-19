# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:04:56 2021

@author: user
"""
import Automold as am
import Helpers as hp
import cv2
import random
import os
import numpy as np
from itertools import islice
from matplotlib import pyplot as plt
from PIL import Image
import glob
import torch
from torch.utils.data import TensorDataset, DataLoader


#%% AUGMENTATION OPTIONS:
    # 0-1: am.brighten, am.darken, add_snow?, add_fog, add_speed
    # ?: add_shadow?, add_rain, add_gravel?, add_sun_flare, add_manhole?
    #: random_flip
        
        # Average Time taken per augmentaion function to process 1 image(1280x720):
        # add_rain 0.02s
        # add_sun_flare 0.09s
        # add_fog 0.37s
        # add_speed 0.20s
        # random_brightness 0.05s
        # add_shadow 0.01s
        # random_flip 0.00s
        # add_manhole 0.01s
        # add_autumn 0.31s
        # add_gravel 0.04s
        # add_snow 0.06s
        # -----------------------
        # Total no. of augmented images created: 99
        # -----------------------
        # Total time taken to create  99  augmented images: 10.42s
#%%
def is_list(x):
    return ((type(x) is list)  or (len(x.shape)==4))

err_aug_type='wrong augmentation function is defined'
err_aug_list_type='aug_types should be a list of string function names'

##INPUTS (images,labels) MUST BE NUMPY ARRAY
#images: shuffled training images to be augmented
#aug_types: list of desired augmentations
#aug: list of the distribution of augemnations
#params: list of paramter values corresponding to each augmentation
def augBlock(images, labels, aug_types=["brighten", "darken","add_snow","random_flip","add_rain"], aug = [0.025,0.025,0.025,0.025,0.025], params = [0.5,0.5,0.5,None,None]): 
    output = []
    label_out = []
    aug_types_all=["brighten","darken","add_shadow","add_snow","add_rain","add_fog","add_gravel","add_sun_flare","add_speed","add_autumn","random_flip","add_manhole"]
    
    if not(is_list(aug_types)):
        raise Exception(err_aug_list_type)
        
    am.verify_image(images)
    for aug_type in aug_types:
        if not(aug_type in aug_types_all):
            raise Exception(err_aug_type)
            
    if(is_list(images)):
        #Select random subset of images based on aug
        # aug.append(1-sum(aug)) #unaffected images
        # print(np.dot(aug,len(images)))
        seclist = np.around(np.dot(aug,len(images)))
        # print(seclist)
        seclist = [ int(np.round(x)) for x in seclist ]
        # print(seclist)
        remaining = 1-(sum(seclist)/len(images))
        seclist.append(int(np.round(remaining*len(images))))
        # print(seclist)
        it = iter(images)
        it2 = iter(labels)
        image_splits = [sli for sli in (list(islice(it, 0, i)) for i in seclist)]
        label_splits = [sli for sli in (list(islice(it2, 0, i)) for i in seclist)]
        # print(len(image_splits))
        for count in range(len(image_splits)):
            image_list = image_splits[count]
            label_list = label_splits[count]
            if count == (len(aug_types)):
                for img in image_list:
                    output.append(img)
                for lbl in label_list:
                    label_out.append(lbl)
                continue
            selected_aug = aug_types[count]     
            p = params[count]
            # for aug_type in aug_types:
            for img in image_list:
                # r = random.randint(0,len(aug_types)-1)
                # selected_aug=aug_types[r]
                if selected_aug == "add_rain" or selected_aug == "random_flip":
                    command = 'am.'+selected_aug+'(img)'
                else:
                    command='am.'+selected_aug+'(img,' + str(p) + ')'
                output.append(eval(command))
            for lbl in label_list:
                if selected_aug == "random_flip":
                    command = 'am.'+selected_aug+'(lbl)'
                label_out.append(lbl)

    else:
        print(is_list(images))
        print("1 image")
        selected_aug=aug_types[random.randint(0,len(aug_types)-1)]
        command=selected_aug+'(images)'
        output=eval(command)
        
    output= np.array(output)
    # output = np.transpose(output, [0,3,1,2]) 
    label_out= np.array(label_out)
    tensor_x = torch.Tensor(output) # transform to torch tensor
    tensor_y = torch.Tensor(label_out)
    # tensor_y = tensor_y[:,None,:,:]
    # print(tensor_x.size())
    # print(tensor_y.size())
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size = len(images)) # create your dataloader
    return my_dataloader


#%% Test AugBlock
# images = []
# images_out = []
# in_Dir='../Images/'
# out_Dir='../out/'

# for filename in glob.glob(in_Dir+'/*.png'): #assuming gif
#     im=Image.open(filename)
#     images.append(np.asarray(im))
    
# images_aug = augBlock(images,aug_types=["brighten", "darken","add_snow","random_flip","add_rain"], aug = [0.1,0.1,0.1,0.1,0.1])
# count = 0
# for img in images_aug:
#     im = Image.fromarray(img)
#     # print(type(im))
#     images_out.append(im)
#     im.save(out_Dir+"img" + str(count) + ".png")
#     count = count+1
# #%%
# # images = np.random.randint(0, 255, size=(100, 720,1080,3), dtype = 'uint8')
# p = augBlock(images)
# f, axarr = plt.subplots(5,2, figsize=(12, 12))
# axarr[0,0].imshow(images[0])
# axarr[0,1].imshow(p[0])

# axarr[1,0].imshow(images[2])
# axarr[1,1].imshow(p[2])

# axarr[2,0].imshow(images[4])
# axarr[2,1].imshow(p[4])

# axarr[3,0].imshow(images[6])
# axarr[3,1].imshow(p[6])

# axarr[4,0].imshow(images[8])
# axarr[4,1].imshow(p[8])

# #%% Test functions
# for file in os.listdir(in_Dir):
#     image = cv2.imread(in_Dir + "/" + file)
#     coeff = random.uniform(0.7,0.8)
#     dark_images= am.darken(image, darkness_coeff=coeff) ## if brightness_coeff is undefined brightness is random in each image
#     # cv2.namedWindow('Test1',cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('Test1', 1500,1000)
#     # cv2.imshow("Test1", dark_images)
#     cv2.imshow("Test",(np.array(images[0].getdata())).reshape(images[0].size[0], images[0].size[1], 3))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     Kernels = [15,25]
#     blurK = random.choice(Kernels)
#     # fast_dark =cv2.blur(dark_images, (15, 15))
#     fast_dark =cv2.GaussianBlur(dark_images, (blurK, blurK),0)
#     # fast_dark = cv2.medianBlur(dark_images,15)
#     cv2.imwrite(out_Dir + "/" + file, fast_dark)

# # cv2.imshow("Test1", fast_dark)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
