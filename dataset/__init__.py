import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
from torch import distributed as dist
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DistributedSampler, RandomSampler

from .augmentations import train_augmentation,val_augmentation
from .pst import PST
from .fmb import FMB_dataset
def get_dataloader(args):

    if args.dataset =='pst' :
        traintransform = train_augmentation([960,960], seg_fill=255)
        valtransform = val_augmentation([720,1280])
        trainset = PST(args.data_path, 'train', traintransform, ['rgb', 'thermal'] )
        validset = PST(args.data_path, 'val', valtransform, ['rgb', 'thermal']  )
    elif args.dataset =='fmb' :
        traintransform = train_augmentation([1024, 1024], seg_fill=255)
        valtransform = val_augmentation([1024, 1024])
        trainset = FMB_dataset(args.data_path, 'train', traintransform, ['rgb', 'thermal'] )#, 'object_mask'
        validset = FMB_dataset(args.data_path, 'val', valtransform, ['rgb', 'thermal']  )#, 'object_mask'
    else:
        print("the dataset is not supported now!!!")


    return trainset, validset
