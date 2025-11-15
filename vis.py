import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter.*")

warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import sys
import time
from datetime import datetime
import dateutil.tz

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
import cfg
from utils import *
import function

import torch.autograd 
from torch.cuda.amp import GradScaler, autocast
from sam2.modeling import build_vis_sam
from conf import settings
import numpy as np
import random
torch.cuda.empty_cache()
configs = {
    'b':
    {
    'scalp': 1,
    'embed_dim': 112,
    'num_heads': 2,
    'stages': (2, 3, 16, 3),
    'global_att_blocks': (12,16,20),
    'window_pos_embed_bkg_spatial_size': [14, 14],
    'window_spec': [8, 4, 14, 7],
    'num_pos_feats': 256,
    'normalize': True,
    'scale': None,
    'temperature': 10000,
    'd_model': 256,
    'backbone_channel_list': [896, 448, 224, 112],
    'fpn_top_down_levels': [2, 3],
    'fpn_interp_model': 'nearest',
    'ckpt_path' : '/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_base_plus.pt'
    },

    'l': 
    {
    'scalp': 0,
    'embed_dim': 144,
    'num_heads': 2,
    'stages': (2, 6, 36, 4),
    'global_att_blocks': (23, 33, 43),
    'window_pos_embed_bkg_spatial_size': [7, 7],
    'window_spec': [8, 4, 16, 8],
    'num_pos_feats': 256,
    'normalize': True,
    'scale': None,
    'temperature': 10000,
    'd_model': 256,
    'backbone_channel_list': [1152, 576, 288, 144],
    'fpn_top_down_levels': [2, 3],
    'fpn_interp_model': 'nearest',
    'ckpt_path' : '/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_large.pt'
    },

    's':
    {
    'scalp': 0,
    'embed_dim': 96,
    'num_heads': 1,
    'stages': (1, 2, 11, 2),
    'global_att_blocks': (7, 10, 13),
    'window_pos_embed_bkg_spatial_size': [7, 7],
    'window_spec': [8, 4, 14, 7],
    'num_pos_feats': 256,
    'normalize': True,
    'scale': None,
    'temperature': 10000,
    'd_model': 256,
    'backbone_channel_list': [768, 384, 192, 96],
    'fpn_top_down_levels': [2, 3],
    'fpn_interp_model': 'nearest',
    'ckpt_path' : '/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_b.pt'
    },

    't':
    {
    'scalp': 0,
    'embed_dim': 96,
    'num_heads': 1,
    'stages': (1, 2, 7, 2),
    'global_att_blocks': (5, 7, 9),
    'window_pos_embed_bkg_spatial_size': [7, 7],
    'window_spec': [8, 4, 14, 7],
    'num_pos_feats': 256,
    'normalize': True,
    'scale': None,
    'temperature': 10000,
    'd_model': 256,
    'backbone_channel_list': [768, 384, 192, 96],
    'fpn_top_down_levels': [2, 3],
    'fpn_interp_model': 'nearest',
    'ckpt_path' : '/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_b.pt'
    }, 
}


from dataset import *

def get_vis_dataloader(dataset, data_path):
    if dataset =='pst' :
        traintransform = train_augmentation([960,960], seg_fill=255)
        valtransform = val_augmentation([720,1280])
        trainset = PST(data_path, 'train', traintransform, ['rgb', 'thermal'] )#, 'object_mask'
        validset = PST(data_path, 'val', valtransform, ['rgb', 'thermal']  )#, 'object_mask'
        sampler = RandomSampler(trainset)
        sampler_val = None

        #nice_train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=True, sampler=sampler)
        nice_test_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, sampler=sampler_val)
        return nice_test_loader
    elif dataset =='fmb_new' :
        traintransform = train_augmentation([1024,1024], seg_fill=255)
        valtransform = val_augmentation([720,720])
        #trainset = FMB_dataset(data_path, 'train', traintransform, ['rgb', 'thermal'] )#, 'object_mask'
        validset = FMB_dataset(data_path, 'val', valtransform, ['rgb', 'thermal']  )#, 'object_mask'
        #sampler = RandomSampler(trainset)
        sampler_val = None

        #nice_train_loader = DataLoader(trainset, batch_size=1, num_workers=8, pin_memory=True, sampler=sampler)
        nice_test_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, sampler=sampler_val)
        return  nice_test_loader
    else:
        print("the dataset is not supported now!!!")


    return trainset, validset






def q_color(dataset):
    if dataset =='mf':
        classes = [ 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
        palette = torch.tensor([[64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192], [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]])
    elif dataset =='fmb_new':
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
    elif dataset =='pst':
        classes = ["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"]
        palette = torch.tensor([[70, 70, 70],
                [100, 40, 40],
                [55, 90, 80],
                [220, 20, 60],
                [153, 153, 153],
                ])
    elif dataset =='nyu':
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
def visualize_segmentation(dataset,label_tensor, pred_tensor, visualize_file, name):
    classes, palette = q_color(dataset)
    label_img = label_tensor.squeeze().cpu().numpy()
    pred_img = pred_tensor.squeeze().cpu().numpy()
    save_dir = visualize_file
    h, w = label_img.shape
    output_label_img = np.zeros((h, w, 3), dtype=np.uint8)
    output_pred_img = np.zeros((h,w, 3), dtype=np.uint8)

    for idx in range(len(classes)):
        # if idx == 0:  # unlabeled class
        #     continue

        mask_label = label_img == idx
        mask_pred = pred_img == idx
        output_label_img[mask_label] = palette[idx]
        output_pred_img[mask_pred] = palette[idx]
    #255label
    ig_label = label_img == 255
    output_pred_img[ig_label] = [0, 0, 0]
    # 保存标签图像
    label_filename = os.path.join(save_dir, f'label_{dataset}_{name}.png')
    Image.fromarray(output_label_img).save(label_filename)

    # 保存预测图像
    pred_filename = os.path.join(save_dir, f'pred_{dataset}_{name}.png')
    Image.fromarray(output_pred_img).save(pred_filename)



# def vis_cam(dataset, image_tensor, output, save_path, name):
#     global feature_maps, grads
#     # feature_maps.clear()
#     # grads.clear()

#     # 确保梯度计算
#     output.retain_grad()
    
#     # 选择目标类别（示例：类别2）
#     class_idx = 2
#     output_class = output[:, class_idx].sum()  # 标量
    
#     # 反向传播计算梯度
#     output_class.backward(retain_graph=True)
    
#     # 检查数据是否存在
#     if len(feature_maps) == 0 or len(grads) == 0:
#         print("Error: 未捕获到特征或梯度！")
#         return
    
#     # 计算 CAM
#     feature_map = feature_maps[0]  # (1, C, H, W)
#     grad = grads[0]                # (1, C, H, W)
    
#     weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
#     cam = (weights * feature_map).sum(dim=1, keepdim=True)  # (1, 1, H, W)
#     cam = F.relu(cam)
#     cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
#     cam = cam.squeeze().cpu().numpy()
#     cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)  # 归一化
    
#     # 反归一化图像
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(image_tensor.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(image_tensor.device)
#     image_denorm = image_tensor * std + mean
#     image_np = image_denorm.squeeze().cpu().numpy().transpose(1,2,0)
#     image_np = np.clip(image_np*255, 0, 255).astype(np.uint8)
    
#     # 叠加热力图
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_np)
#     plt.imshow(cam, cmap='jet', alpha=0.5)
#     plt.axis('off')
    
#     # 保存结果
#     save_file = os.path.join(save_path, f'cam_{name}.png')
#     plt.savefig(save_file, bbox_inches='tight')
#     plt.close()
def vis_cam(dataset, image_tensor, output, save_path, name, feature_maps, grads):
    # 选择目标类别（示例：类别2）
    class_idx = 3
    output_class = output[:, class_idx].sum()  # 标量
    
    # 反向传播计算梯度
    output_class.backward(retain_graph=True)
    
    if not feature_maps or not grads:
        print("Error: 未捕获到特征或梯度！")
        return
    
    # 计算 CAM
    feature_map = feature_maps[0]  # (1, C, H, W)
    grad = grads[0]                # (1, C, H, W)
    
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * feature_map).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)  # 归一化
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(image_tensor.device)
    image_denorm = image_tensor * std + mean
    image_np = image_denorm.squeeze().cpu().numpy().transpose(1,2,0)
    image_np = np.clip(image_np*255, 0, 255).astype(np.uint8)
    
    # 叠加热力图
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # 保存结果
    save_file = os.path.join(save_path, f'cam_{name}_classidx_{class_idx}.png')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()

def evaluate_sam(modal, dataloader, device,val_loss, dataset):
    print('Evaluating...')
    modal.eval()
    loss_val= 0
    iter = 0
    if dataset == 'fmb_new':
        metrics = Metrics(num_classes=14,device=device)
    elif dataset == 'pst':
        metrics = Metrics(num_classes=5,device=device)
    else:
        raise ValueError("unsupport dataset")
    sliding = False
    features = []
    gradients = []
    def hook_fn_forward(module, input, output):
        features.append(output)

    def hook_fn_backward(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    target_layer = model.decoder.head
    forward_hook = target_layer.register_forward_hook(hook_fn_forward)
    backward_hook = target_layer.register_full_backward_hook(hook_fn_backward)
    for pack in tqdm(dataloader):
        imgs = pack['image'].to(device)
        x = pack['x'].to(device)
        masks = pack['label'].to(device)
        name = pack['item_name']
        #with torch.no_grad():
        with torch.enable_grad():
            data  = [imgs, x]
            visualize_file = '/root/autodl-tmp/ssssam2/output/fmb_cam'  
            if not os.path.exists(visualize_file):
                os.makedirs(visualize_file)
            seg_predictions= modal(data)
            
                
            predictions = nn.functional.interpolate( seg_predictions, size=masks.size()[1:], mode='bilinear')  # mode='nearest'//'bilinear'
                
            loss = val_loss(predictions, masks.squeeze(1).long())
            metrics.update(predictions.softmax(dim=1), masks.squeeze(1).long())

 
            loss_val += loss.item()
            seg_predictions = seg_predictions.requires_grad_()

            #vis_cam(dataset, imgs, seg_predictions, visualize_file, name[0], features, gradients)
            modal.zero_grad()
            features.clear()  # 清理列表以便下一次迭代
            gradients.clear()
            x_label = masks
            x_pre = predictions.softmax(dim=1).argmax(dim=1)

            visualize_segmentation(dataset, x_label[0].unsqueeze(0), x_pre[0].unsqueeze(0), visualize_file, name[0])
            torch.cuda.empty_cache()
        iter += 1
    forward_hook.remove()
    backward_hook.remove()
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return  loss_val, miou


import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = cfg.parse_args()
GPUdevice = torch.device('cuda:0')
args.dataset = 'fmb_new'
ckp = '/root/autodl-tmp/sam2/best_ckpt/fmb_67.79/sam_sam_fmb_new_epoch85_67.79.pth'
model, criterion_G = build_vis_sam(configs,ckp,args)
model = model.to(GPUdevice)
model.eval()







# # 计算模型总参数量
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

nice_val_loader = get_vis_dataloader(args.dataset, '/root/autodl-tmp/sam2/data/FMB')

model.eval()
val_loss ,miou = evaluate_sam( model, nice_val_loader, GPUdevice, criterion_G, args.dataset)
print(miou)




# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# import torch
# import cfg
# from sam2.modeling import configs,build_mix_rgbx_text,build_mix_rgbx_seg,build_mix_rgbx_psp
# # 假设 model 是 PyTorch 模型，input 是一个示例输入
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# args = cfg.parse_args()
# model, criterion_G = build_mix_rgbx_seg(args,configs)
# model = model.to(device)
# model.eval()
# criterion_G = criterion_G.to(device)
# rgb = torch.randn(1,3,1024,1376).to(device)
# x = torch.randn(1,3,1024,1376).to(device)
# input = [rgb, x]
# flops = FlopCountAnalysis(model, input)
# print("FLOPs:", flops.total())
# print("Parameter count:", parameter_count_table(model))
# #python test_parm.py -ms b -dataset deliver -distributed "3" -type single-rgb -data_path /root/autodl-tmp/segment-anything-2/data/DELIVER -gpu_device 3 -b 1 -lr 1e-4 -use_text True -fuse_type fpg
# total = sum([param.nelement() for param in model.parameters()])
# print(' Number of params: %.2fM' % (total / 1e6))

# target_layer_name = "Adapter"  # 替换成你关注的层名称
# total_params = 0
# for name, param in model.named_parameters():
#     if target_layer_name in name:
#         layer_params = param.nelement()
#         total_params += layer_params
#         print(f"Layer: {name} | Parameters: {layer_params/1e6:.2f}M")

# print(f"\nTotal parameters in '{target_layer_name}' layers: {total_params/1e6:.2f}M")


# # python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=30000 multigpu_train.py -ms l -dataset nyu_new -distributed "1" -type concat -data_path /root/autodl-tmp/sam2/data/NYUDepth -gpu_device 1 -b 4 -lr 1e-5 -use_text True -fuse_type add1 -ddp True

# # python vis.py -ms l -dataset fmb_new -distributed "1" -type concat -data_path /root/autodl-tmp/sam2/data/FMB -gpu_device 1 -b 4 -lr 1e-5 -use_text True -fuse_type add1 -ddp True -ft_type seg














