from __future__ import print_function, division
import os
import time
import warnings
import sys
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Dice
# from torchsummary import summary

torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled

sys.path.append(os.path.abspath('./Automold--Road-Augmentation-Library'))
from network import UNet as Model
from Config import config as Config
from AugBlock import augBlock
from gtaV_helper import GTA5
from cityscapes_helper import CitySpaceseDataset

#%%
RESULT_FOLDER = '/Result'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'

def CEL(pred,target):
    criterion = nn.CrossEntropyLoss()
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    loss = criterion(pred,target)
    return loss


def MSE(pred,target):
    criterion = nn.MSELoss()
    loss = criterion(pred,target)
    return loss 



def train(model, train_loader, optimizer, scheduler, writer, cfg=None, test_loader = None ,aug=None,params=None):  #writer
    best_loss = 1e10
    best_val = 100
    count = 0

    # looping over given number of epochs
    for epoch in range(1):
        tic = time.time()
        # scheduler.step()
        model.train()

        for inputs, targets in train_loader:
            count += 1

            inputs = inputs.cpu().type(torch.float32)
            inputs = np.transpose(inputs, [0,3,1,2]) 
            # print(inputs.size())
            targets = targets.cpu().type(torch.float32)
            targets = targets[:,None,:,:]
#             print(targets.dtype, targets.size())
            
            
#             optimizer.zero_grad()
#             preds = model(inputs)
#             print(preds.dtype,preds.size())
#             loss = CEL(preds, targets)

#             loss.backward()
#             optimizer.step()

            # tensorboard logging
#             writer.add_scalar('Train/Loss', loss.item(), count)
            
            # Augmented data training
            if aug==None:
                aug_data = augBlock(inputs.cpu().numpy(), targets.cpu().numpy())
            else:
                aug_data = augBlock(inputs.cpu().numpy(), targets.cpu().numpy(),aug=aug,params=params)
            print(f'Time:{time.time()-tic}')
            tic = time.time()
            aug_inputs, aug_targets = next(iter(aug_data))
            
            aug_inputs = aug_inputs.cpu()
            aug_targets = aug_targets.cpu()

            optimizer.zero_grad()
            aug_preds = model(aug_inputs)
            loss = CEL(aug_preds, aug_targets)

            loss.backward()
            optimizer.step()

            # tensorboard logging
            writer.add_scalar('Train/Loss', loss.item(), count)

            print(f'Epoch:{epoch}, Step:{count}, Loss:{loss.item():.6f}, BestVal:{best_val:.6f}, Time:{time.time()-tic}')
            tic = time.time()
            
            if count == 1:
                break

    return loss.item()

def dice(pred, targets):
    criterion = Dice(average='micro')
    loss = 0
    print((len(pred)))
    for i in range(len(pred)):
        print(targets[i][0].size())
        print(pred[i][0].size())
        loss+=1
        #break
        #loss += criterion(torch.squeeze(pred[i][0].type(torch.int)),torch.squeeze(targets[i][0].type(torch.int)))
    return loss/len(pred)

def eval_model(model, test_loader,best_val=100, cfg=None):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0
    count = 0
    # check dataset type
    for inputs, targets in test_loader:
        count += 1
        inputs = inputs.cpu().type(torch.float32)
        # inputs = np.transpose(inputs, [0,3,1,2]) 
        # print(inputs.size())
        targets = targets.cpu().type(torch.float32)
        targets = targets[:,None,:,:]

        with torch.set_grad_enabled(False):
            preds = model(inputs)  
#             preds = torch.clip(preds, 0, 1)
            print(preds.dtype,preds.size())
            loss = dice(preds, targets) 
            # NMSE
            
            print(f'Loss:{loss.item():.6f}')
#             avg_loss += (loss.item() * inputs.shape[0])
#             n_samples += inputs.shape[0]
        print(count)
        if count == 1:
            break

#     model.train()
    
    return loss#.item()


def main(train_path,test_path):
    cfg = Config()
    
    os.makedirs(TENSORBOARD_PREFIX, exist_ok=True)
    os.makedirs(f'{RESULT_FOLDER}/{cfg.exp_name}', exist_ok=True)   

    writer = SummaryWriter(log_dir=f'{TENSORBOARD_PREFIX}/{cfg.exp_name}')
    
    data_train = GTA5(train_path)
    data_test = CitySpaceseDataset(test_path)
        
    train_loader =  DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))
    test_loader =  DataLoader(data_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, generator=torch.Generator(device='cpu'))
        
    # init model 
    model = Model()
    model.cpu()

    # init optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

    val_train = train(model, train_loader, optimizer, scheduler, writer, cfg=None, test_loader = None)

    print('[*] train ends... ')
    print(f'[*] Loss train: {val_train}')

    score_test = eval_model(model, test_loader,best_val=100, cfg=None)
    print('[*] Eval ends... ')
    print(f'[*] Score test: {score_test}')
    
    return score_test

#%%
# main(r"../GTA5/data", r"../Cityscapes/")