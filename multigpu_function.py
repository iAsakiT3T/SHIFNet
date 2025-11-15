
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete

from tqdm import tqdm

import cfg
from conf import settings
from utils import *
import pickle

from PIL import Image
import torchvision.transforms.functional as TF


def train_sam(imgs, x, masks, modal, train_loss):

    data  = [imgs, x]
    seg_predictions= modal(data)
    predictions = nn.functional.interpolate(seg_predictions, size=masks.size()[1:],mode='bilinear')
    loss = train_loss(predictions,masks.squeeze(1).long())
    return loss


def evaluate_sam(args,modal, dataloader, device,val_loss,epoch):
    print('Evaluating...')
    modal.eval()
    loss_val= 0
    iter = 0
    if args.dataset == 'fmb' :
        metrics = Metrics(num_classes=14,device=device)
    elif args.dataset == 'pst':
        metrics = Metrics(num_classes=5,device=device)
    else:
        raise ValueError("unsupport dataset")
    sliding = False
    for pack in tqdm(dataloader):
        imgs = pack['image'].to(device)
        x = pack['x'].to(device)
        masks = pack['label'].to(device)

        with torch.no_grad(): 
            data  = [imgs, x]
            seg_predictions= modal(data)
            predictions = nn.functional.interpolate(seg_predictions, size=masks.size()[1:],mode='bilinear')#mode='nearest'//'bilinear'
            loss = val_loss(predictions,masks.squeeze(1).long())
            metrics.update(predictions.softmax(dim=1), masks.squeeze(1).long())
            loss_val += loss.item()
        iter += 1
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou,loss_val
