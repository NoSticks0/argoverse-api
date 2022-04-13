#!/usr/bin/env python
# coding: utf-8

# # PSM Net Baseline
# 
# *See for reference: https://github.com/JiaRenChang/PSMNet*

# ## 1 Setup
# 
# ----
# 
# Ensure you're in <...>/argoverse-api

# In[1]:


import os
#os.chdir('..')
#print(f"Current directory: {os.getcwd()}")


# In[2]:


import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argoverse.data_loading.stereo_dataloader import ArgoverseStereoDataLoader
from argoverse.evaluation.stereo.eval import StereoEvaluator
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.camera_stats import RECTIFIED_STEREO_CAMERA_LIST
import time

from models import *
from dataloader import KITTI_submission_loader as DA
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

STEREO_FRONT_LEFT_RECT = RECTIFIED_STEREO_CAMERA_LIST[0]
STEREO_FRONT_RIGHT_RECT = RECTIFIED_STEREO_CAMERA_LIST[1]


# In[3]:


main_dir = "./"
data_dir = f"{main_dir}argoverse-stereo_v1.1/"


# ## 2 Main Model
# 
# ----
# 
# Goal: Predict disparity map from pair of stereo images
# 
# ### 2.1 Training

# In[4]:


# Should create/tune a PSM net and save it as "model"
    # I'd say, given the time constraints, it's probably best to take one of the prebuilt ones from their
    # Github page and just tune it on the Argo data? If you want to try something else though go for it.

    # I think their code is input size invariant because the only operations they do are colvolution and
    # spatial pyramid pooling, neither of which should have any sizes hardcoded.
maxdisp = 192
loadmodel = "./checkpoint_9.tar"
model = basic(maxdisp)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
state_dict = torch.load(loadmodel)['state_dict']
#state_dict = {(k if 'module' not in k else k[7:]): v for k, v in state_dict['state_dict'].items()}
model.load_state_dict(state_dict)


# ### 2.2 Evaluation
# 
#  * Probably should add checkpoint code depending on how long this takes to run (e.g. save metrics after every iteration).

# In[6]:


def test(imgL,imgR):
    model.eval()

    #if args.cuda:
        #imgL = imgL.cuda()
        #imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output


# In[18]:


stereo_data_loader = ArgoverseStereoDataLoader(data_dir, "val")

metrics = []
lens = []
log_ids = [
    'f9fa3960-537f-3151-a1a3-37a9c0d6d7f7',
    '1d676737-4110-3f7e-bec0-0c90f74c248f',
    'da734d26-8229-383f-b685-8086e58d1e05',
    '6db21fda-80cd-3f85-b4a7-0aadeb14724d',
    '85bc130b-97ae-37fb-a129-4fc07c80cca7',
    '33737504-3373-3373-3373-633738571776',
    '033669d3-3d6b-3d3d-bd93-7985d86653ea',
    'f1008c18-e76e-3c24-adcc-da9858fac145',
    '5ab2697b-6e3e-3454-a36a-aba2c6f27818',
    'cb762bb1-7ce1-3ba5-b53d-13c159b532c8',
    '70d2aea5-dbeb-333d-b21e-76a7f2f1ba1c',
    '2d12da1d-5238-3870-bfbc-b281d5e8c1a1',
    '64724064-6472-6472-6472-764725145600',
    '00c561b9-2057-358d-82c6-5b06d76cebcf',
    'cb0cba51-dfaf-34e9-a0c2-d931404c3dd8',
    'e9a96218-365b-3ecd-a800-ed2c4c306c78',
    '39556000-3955-3955-3955-039557148672'
]
i = 0
for log_id in log_ids:
    i += 1
    print(f"Now evaluating log_id :: \t\t {i}/{len(log_ids)}")
    left_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
        log_id=log_id, 
        camera_name=STEREO_FRONT_LEFT_RECT)
    right_stereo_img_fpaths = stereo_data_loader.get_ordered_log_stereo_image_fpaths(
        log_id=log_id, 
        camera_name=STEREO_FRONT_RIGHT_RECT)
    disparity_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
        log_id=log_id,
        disparity_name="stereo_front_left_rect_disparity")
    disparity_obj_map_fpaths = stereo_data_loader.get_ordered_log_disparity_map_fpaths(
        log_id=log_id,
        disparity_name="stereo_front_left_rect_objects_disparity")
    lens += [len(left_stereo_img_fpaths)]
    
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    
    
    for inx in tqdm(range(len(left_stereo_img_fpaths))):

        imgL_o = Image.open(left_stereo_img_fpaths[inx]).convert('RGB')
        imgR_o = Image.open(right_stereo_img_fpaths[inx]).convert('RGB')
        
        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        if top_pad !=0:
            if right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            else:
                img = pred_disp[top_pad:, :]
        else:
            if right_pad != 0:
                img = pred_disp[:, :-right_pad]
            else:
                img = pred_disp
        
        # Load the testing image and corresponding disparity and foreground disparity maps
        #stereo_front_left_rect_image = stereo_data_loader.get_rectified_stereo_image(left_stereo_img_fpaths[idx])
        #stereo_front_right_rect_image = stereo_data_loader.get_rectified_stereo_image(right_stereo_img_fpaths[idx])
        stereo_front_left_rect_disparity = stereo_data_loader.get_disparity_map(disparity_map_fpaths[inx])
        stereo_front_left_rect_objects_disparity = stereo_data_loader.get_disparity_map(disparity_obj_map_fpaths[inx])
        
        #left_disparity_pred = np.uint16(left_disparity)
        left_disparity_pred = (img*256).astype('uint16')
        img = Image.fromarray(left_disparity_pred)
        #print(left_disparity_pred)
        timestamp = int(Path(disparity_map_fpaths[inx]).stem.split("_")[-1])
        save_dir_disp = f"{main_dir}707-files/results/psm/stereo_output/{log_id}"
        Path(save_dir_disp).mkdir(parents=True, exist_ok=True)
        filename = f"{save_dir_disp}/disparity_{timestamp}.png"
        img.save(filename)
        #cv2.imshow("image", left_disparity_pred)
        #if not cv2.imwrite(filename, left_disparity_pred):
            #raise Exception("Could not write image to " +filename)

    pred_dir = Path(save_dir_disp)
    gt_dir = Path(f"{data_dir}/disparity_maps_v1.1/val/{log_id}")
    save_figures_dir = Path(f"/tmp/results/psm/figures/{log_id}/")
    save_figures_dir.mkdir(parents=True, exist_ok=True)

    evaluator = StereoEvaluator(
        pred_dir,
        gt_dir,
        save_figures_dir,
    )
    metrics += [evaluator.evaluate()]


# In[19]:


compiled_metrics = { key : 0 for key in metrics[0] }
for i in range(0, len(metrics)):
    compiled_metrics = { key : compiled_metrics[key] + lens[i] * metrics[i][key] for key in compiled_metrics }

compiled_metrics = { key : compiled_metrics[key] / sum(lens) for key in compiled_metrics }


# In[ ]:


import json
print(f"{json.dumps(compiled_metrics, sort_keys=False, indent=4)}")

