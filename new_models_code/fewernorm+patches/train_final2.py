#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from __future__ import print_function

import os
#os.chdir('..')
#print(f"Current directory: {os.getcwd()}")


# In[2]:


#import sys
#!{sys.executable} -m pip install numba


# In[3]:


import argparse
import random
import skimage
import skimage.io
import skimage.transform
import time
import math
import copy
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader_patches as DA

from models import *

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argoverse.data_loading.stereo_dataloader import ArgoverseStereoDataLoader
from argoverse.evaluation.stereo.eval import StereoEvaluator
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.camera_stats import RECTIFIED_STEREO_CAMERA_LIST

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]


# In[4]:


main_dir = "./"
data_dir = f"{main_dir}argoverse-stereo_v1.1/"


# In[5]:


stereo_data_loader_train = ArgoverseStereoDataLoader(data_dir, "train")
stereo_data_loader_val = ArgoverseStereoDataLoader(data_dir, "val")


# In[6]:


train_log_ids = os.listdir(f"{data_dir}/rectified_stereo_images_v1.1/train/")
test_log_ids = os.listdir(f"{data_dir}/rectified_stereo_images_v1.1/val/")
num_logs = 20


# In[10]:


parser = argparse.ArgumentParser(description='PSMNet')
maxdisp = 192
patch_size = 3
cv_patch_size = None
arg_model = 'basic'
epochs = 10
loadmodel = None
savemodel = "./"
no_cuda = False
seed = 1

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    
all_left_img = []
all_right_img = []
all_left_disp = []
for log_id in train_log_ids:
    # Loading the left rectified stereo image paths for the chosen log.
    all_left_img += stereo_data_loader_train.get_ordered_log_stereo_image_fpaths(
     log_id=log_id,
     camera_name=STEREO_FRONT_LEFT_RECT,
    )
    # Loading the right rectified stereo image paths for the chosen log.
    all_right_img += stereo_data_loader_train.get_ordered_log_stereo_image_fpaths(
     log_id=log_id,
     camera_name=STEREO_FRONT_RIGHT_RECT,
    )
    # Loading the disparity map paths for the chosen log.
    all_left_disp += stereo_data_loader_train.get_ordered_log_disparity_map_fpaths(
     log_id=log_id,
     disparity_name="stereo_front_left_rect_disparity",
    )

test_left_img = []
test_right_img = []
test_left_disp = []
for log_id in test_log_ids[:num_logs]:             
    # Loading the left rectified stereo image paths for the chosen log.
    test_left_img += stereo_data_loader_val.get_ordered_log_stereo_image_fpaths(
     log_id=log_id,
     camera_name=STEREO_FRONT_LEFT_RECT,
    )
    # Loading the right rectified stereo image paths for the chosen log.
    test_right_img += stereo_data_loader_val.get_ordered_log_stereo_image_fpaths(
     log_id=log_id,
     camera_name=STEREO_FRONT_RIGHT_RECT,
    )
    # Loading the disparity map paths for the chosen log.
    test_left_disp += stereo_data_loader_val.get_ordered_log_disparity_map_fpaths(
     log_id=log_id,
     disparity_name="stereo_front_left_rect_disparity",
    )

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 4, shuffle= True, num_workers= 4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 4, shuffle= False, num_workers= 4, drop_last=False)

if arg_model == 'stackhourglass':
    model = stackhourglass(maxdisp)
elif arg_model == 'basic':
    model = basic_final2(maxdisp, patch_size, cv_patch_size)
else:
    print('no model')

if cuda:
    model = nn.DataParallel(model)
    model.cuda()

if loadmodel is not None:
    if cuda:
        state_dict = torch.load(loadmodel)
    else:
        state_dict = torch.load(loadmodel, map_location='cpu')
        state_dict = {(k if 'module' not in k else k[7:]): v for k, v in state_dict['state_dict'].items()}
    model.load_state_dict(state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_true = Variable(torch.FloatTensor(disp_L))

        if cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        
        if arg_model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], reduction='mean') + F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean') 
        elif arg_model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')

        loss.backward()
        optimizer.step()
        
        if loss.data.ndim == 0:
            return loss.data.item()
        else:
            return loss.data[0]

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = copy.deepcopy(disp_true)
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(0, epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(tqdm(TrainImgLoader)):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            # if (batch_idx % 100 == 0):
                # print('Iter %d t raining loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            if np.isnan(loss):
                total_train_loss += 40
            else:
                total_train_loss += loss
            
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        savefilename = savemodel+'/final2_checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
    #total_test_loss = 0
    #for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        #test_loss = test(imgL,imgR, disp_L)
        #print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
        #total_test_loss += test_loss

    #print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------
    #SAVE test information
    #savefilename = savemodel+'testinformation.tar'
    #torch.save({
            #'test_loss': total_test_loss/len(TestImgLoader),
    #}, savefilename)


# In[12]:


if __name__ == '__main__':
    main()


# In[ ]:




