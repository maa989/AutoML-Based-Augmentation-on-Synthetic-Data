import os
import sys
from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch


class CitySpaceseDataset(Cityscapes):
    def my_segmentation_transforms(self,image, target):
        image = F.resize(image, (256, 455))
        target = F.resize(target, (256, 455), interpolation=T.InterpolationMode.NEAREST)
        
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
    
    
    def label_correction(self,mask):
        ignore_index=255
        void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        class_map = {valid_class:idx for idx,valid_class in enumerate(valid_classes)}
        
        for _voidc in void_classes:
            mask[mask == _voidc] = ignore_index
        for _validc in valid_classes:
            mask[mask == _validc] = class_map[_validc]
        return mask
        
    def __getitem__(self, index):        
        image = Image.open(self.images[index]).convert('RGB')
        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        img,mask = self.my_segmentation_transforms(image,target)
        return img,self.label_correction(mask)
    
    def decode_segmap(self,temp):
        #convert gray scale to color
        ignore_index=255
        valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        colors = [   [  0,   0,   0],
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
            ]

        label_colors = {idx:color for idx,color in enumerate(colors)}
        n_classes=len(valid_classes)
        
        temp=temp.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            r[temp == l] = label_colors[l][0]
            g[temp == l] = label_colors[l][1]
            b[temp == l] = label_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
