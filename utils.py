
#----------schedules------------
import torch
import os
import torchvision
import logging
import time
from datetime import datetime
from abc import ABCMeta, abstractmethod
import torchvision.utils as vutils
import dateutil.tz
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Tuple
from PIL import Image

class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr




#-------------------loss------------------------
class Dice_CrossEntropy_Loss(nn.Module):
    def __init__(self,  smooth=1e-6):
        super(Dice_CrossEntropy_Loss, self).__init__()
        self.smooth = smooth
        #self.l1 = nn.L1Loss(reduction='none')
        self.criterion_G = nn.CrossEntropyLoss(ignore_index=255)
    def forward(self, predictions, rgb_labels):
        """
        计算Dice Loss，并忽略标签值为255的像素。
        
        参数:
        - predictions: 模型输出, 形状为 (b, num_classes, H, W)
        - labels: 真实标签, 形状为 (b, H, W)
        
        返回:
        - loss: Dice Loss
        """
        # 将预测值转换为概率分布
        #seg_loss
        
        CrossEntropyLoss = self.criterion_G(predictions, rgb_labels)

        return CrossEntropyLoss



#------------------------score------------------
class Metrics:
    def __init__(self, num_classes=19, ignore_label=255, device = "cuda") -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)#8*800*1088
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()]=0.
        miou = ious.mean().item()
        # miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()]=0.
        mf1 = f1.mean().item()
        # mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()]=0.
        macc = acc.mean().item()
        # macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)




#----------------------others-----------------------


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None, color_map=None):
    """
    Visualize segmentation results by combining original images with predicted masks and ground truth masks,
    and save the visualization to a file.

    Args:
        imgs (torch.Tensor): Original images with shape (b, c, h, w).
        pred_masks (torch.Tensor): Predicted masks with shape (b, c, h, w).
        gt_masks (torch.Tensor): Ground truth masks with shape (b, c, h, w).
        save_path (str): File path to save the visualization.
        reverse (bool, optional): Whether to reverse the masks. Defaults to False.
        points (torch.Tensor, optional): Points to highlight in the visualization. Defaults to None.
        color_map (dict, optional): Dictionary mapping class labels to RGB colors. Defaults to None.
    """
    color_map = {
    0: [0, 0, 0],       # unlabeled: Black
    1: [0, 0, 0],       # ego vehicle: Black
    2: [0, 0, 0],       # rectification border: Black
    3: [0, 0, 0],       # out of roi: Black
    4: [0, 0, 0],       # static: Black
    5: [111, 74, 0],    # dynamic: Brown
    6: [81, 0, 81],     # ground: Purple
    7: [128, 64, 128],  # road: Gray
    8: [244, 35, 232],  # sidewalk: Pink
    9: [250, 170, 160], # parking: Light Pink
    10: [230, 150, 140],# rail track: Light Salmon
    11: [70, 70, 70],   # building: Dark Gray
    12: [102, 102, 156],# wall: Light Slate Gray
    13: [190, 153, 153],# fence: Light Pink
    14: [180, 165, 180],# guard rail: Light Slate Gray
    15: [150, 100, 100],# bridge: Rosy Brown
    16: [150, 120, 90], # tunnel: Khaki
    17: [153, 153, 153],# pole: Dark Gray
    18: [153, 153, 153],# polegroup: Dark Gray
    19: [250, 170, 30], # traffic light: Gold
    20: [220, 220, 0],  # traffic sign: Yellow
    21: [107, 142, 35], # vegetation: Olive Drab
    22: [152, 251, 152],# terrain: Pale Green
    23: [70, 130, 180], # sky: Steel Blue
    24: [220, 20, 60],  # person: Crimson
    25: [255, 0, 0],    # rider: Red
    26: [0, 0, 142],    # car: Dark Blue
    27: [0, 0, 70],     # truck: Navy Blue
    28: [0, 60, 100],   # bus: Dark Cyan
    29: [0, 0, 90],     # caravan: Dark Blue
    30: [0, 0, 110],    # trailer: Dark Blue
    31: [0, 80, 100],   # train: Dark Cyan
    32: [0, 0, 230],    # motorcycle: Blue
    33: [119, 11, 32]   # bicycle: Maroon
}

    b, c, h, w = pred_masks.size()
    row_num = min(b, 4)
    
    preds = []
    gts = []
    _, predicted_classes = torch.max(pred_masks, dim=1)
    pred = predicted_classes.unsqueeze(1).expand(b,3,h,w)
    preds.append(pred)
    #print(predicted_classes.size(),gt_masks.size())
    gt = gt_masks.expand(b,3,h,w)
    gts.append(gt)
    imgs = torchvision.transforms.Resize((h,w))(imgs)
    tup = (imgs[:row_num,:,:,:],pred[:row_num,:,:,:], gt[:row_num,:,:,:])
    #print(pred[:row_num,:,:,:])
    #print(gt[:row_num,:,:,:])
    compose = torch.cat(tup,0)
    vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return  






def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger



def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict



def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))





def q_color(args):
    if args.dataset =='mf':
        classes = [ 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
        palette = torch.tensor([[64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]])
    elif args.dataset =='fmb':
        classes = ["Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign", "Vegetation", "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"]
        palette = torch.tensor([[70, 70, 70],
                [100, 40, 40],
                [55, 90, 80],
                [220, 20, 60],
                [153, 153, 153],
                [157, 234, 50],
                [128, 64, 128],
                [244, 35, 232],
                [107, 142, 35],
                [0, 0, 142],
                [102, 102, 156],
                [220, 220, 0],
                [70, 130, 180],
                [81, 0, 81],
                [150, 100, 100],
                ])
    elif args.dataset =='nyu':
        classes = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
        'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
        'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
        'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
        palette = torch.tensor([
            [174, 199, 232],  # wall
            [152, 223, 138],  # floor
            [31, 119, 180],   # cabinet
            [255, 187, 120],  # bed
            [188, 189, 34],   # chair
            [140, 86, 75],    # sofa
            [255, 152, 150],  # table
            [214, 39, 40],    # door
            [197, 176, 213],  # window
            [148, 103, 189],  # bookshelf
            [196, 156, 148],  # picture
            [23, 190, 207],   # counter
            [247, 182, 210],  # blinds
            [219, 219, 141],  # desk
            [255, 127, 14],   # shelves
            [158, 218, 229],  # curtain
            [44, 160, 44],    # dresser
            [112, 128, 144],  # pillow
            [227, 119, 194],  # mirror
            [82, 84, 163],    # floor mat
            [163, 99, 64],    # clothes
            [255, 168, 187],  # ceiling
            [255, 222, 23],   # books
            [0, 255, 0],      # refrigerator
            [1, 0, 103],      # television
            [148, 255, 181],  # paper
            [44, 160, 101],   # towel
            [112, 128, 144],  # shower curtain
            [156, 55, 196],   # box
            [255, 250, 200],  # whiteboard
            [0, 0, 255],      # person
            [255, 0, 0],      # night stand
            [0, 255, 255],    # toilet
            [255, 0, 255],    # sink
            [255, 255, 0],    # lamp
            [0, 0, 0],        # bathtub
            [255, 153, 0],    # bag
            [255, 102, 102],  # other structure
            [102, 204, 0],    # other furniture
            [204, 153, 255]   # other prop
        ])
    return classes, palette
def visualize_segmentation(args, label_tensor, pred_tensor, epoch, iter, visualize_file):
    classes, palette = q_color(args)
    label_img = label_tensor.squeeze().cpu().numpy()
    pred_img = pred_tensor.squeeze().cpu().numpy()
    save_dir = visualize_file
    h, w = label_img.shape
    output_label_img = np.zeros((h, w, 3), dtype=np.uint8)
    output_pred_img = np.zeros((h,w, 3), dtype=np.uint8)

    for idx in range(len(classes)):
        if idx == 0:  # unlabeled class
            continue

        mask_label = label_img == idx
        mask_pred = pred_img == idx
        output_label_img[mask_label] = palette[idx]
        output_pred_img[mask_pred] = palette[idx]
    #255label
    ig_label = label_img == 255
    output_pred_img[ig_label] = [0, 0, 0]
    # 保存标签图像
    label_filename = os.path.join(save_dir, f'label_{args.dataset}_{epoch}_{iter}.png')
    Image.fromarray(output_label_img).save(label_filename)

    # 保存预测图像
    pred_filename = os.path.join(save_dir, f'pred_{args.dataset}_{epoch}_{iter}.png')
    Image.fromarray(output_pred_img).save(pred_filename)




def ready_log_dir(args):
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    train_name = os.path.join('/root/autodl-tmp/sam2/output', f'result/train_learnable_{args.dataset}_{timestamp}.txt')
    val_name = os.path.join('/root/autodl-tmp/sam2/output', f'result/val_learnable_{args.dataset}_{timestamp}.txt')
    visualize_file = os.path.join('/root/autodl-tmp/sam2/output', f'visualize/test_adapt_{args.dataset}') + '_' + timestamp
    os.makedirs(visualize_file)
    with open(train_name,"a",encoding="utf-8") as file:
        file.write('start train'+"\n")
    with open(val_name,"a",encoding="utf-8") as file:
        file.write('strat val'+"\n")
    return train_name , val_name ,visualize_file
