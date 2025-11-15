import os
import time

import warnings
import socket

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import cfg
from multigpu_function import train_sam, evaluate_sam
from conf import settings
#from models.discriminatorlayer import discriminator
from utils import *
from tabulate import tabulate
import torch.autograd 
from multiptu_utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou, get_logger,\
                    update
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import FMB_dataset, train_augmentation, val_augmentation, get_dataloader
from pathlib import Path
from sam2.modeling import configs,build_mix_rgbx_text
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import random
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def main(args, gpu, save_dir):
    
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device('cuda')

    gpus = int(os.environ['WORLD_SIZE'])
    print(gpus)
    trainset , valset = get_dataloader(args)
    epochs = 200

    modal, criterion_G = build_mix_rgbx_text(args,configs)

    modal = modal.to(device)
    
    loss_function = criterion_G
    start_epoch = 0

    if args.ddp: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        modal = DDP(modal, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    if args.dataset == 'fmb':
        num_classes=["Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign", "Vegetation", "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"]
    elif args.dataset == 'pst':
        num_classes=["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"]
    else:
        raise ValueError("unsupport dataset")
    iters_per_epoch = len(trainset) // args.b // gpus
    nice_train_loader = DataLoader(trainset, batch_size=args.b, num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=args.b, num_workers=num_workers, pin_memory=False, sampler=sampler_val)

    optimizer = optim.Adam(modal.parameters(),
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#weight_decay=0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4)
    modal = update(modal)
    if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        #cal_flops(modal, ['img', 'thermal'], logger)
        logger.info('================== model structure =====================')
        logger.info(modal)
        logger.info('================== training config =====================')
        logger.info(args)

    for epoch in range(start_epoch, epochs):
        modal.train()
        if args.ddp: sampler.set_epoch(epoch)

        train_loss = 0.0        

        lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(enumerate(nice_train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (pack) in pbar:
            optimizer.zero_grad(set_to_none=True)
            imgs = pack['image'].to(device)
            x = pack['x'].to(device)
            masks = pack['label'].to(device)

            loss = train_sam(imgs, x, masks, modal, loss_function)


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
            writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        if args.dataset=='fmb':
            lower_iou = 67
        elif args.dataset=='pst':
            lower_iou = 87
        else:
            raise ValueError("unsupport dataset")
        if epoch>-1:
            if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
                acc, macc, _, _, ious, miou,val_loss = evaluate_sam(args, modal, valloader, device,loss_function, epoch)
                writer.add_scalar('val/mIoU', miou, epoch)
                scheduler.step(val_loss)
                if miou > best_mIoU and miou > lower_iou:
                    prev_best_ckp = save_dir / f"sam_{args.dataset}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"sam_{args.dataset}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"sam_{args.dataset}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"sam_{args.dataset}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(modal.module.state_dict() if args.ddp else modal.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': modal.module.state_dict() if args.ddp else modal.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc,num_classes))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

    if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    args = cfg.parse_args()

    seed = random.randint(0, 2**32 - 1)

    fix_seeds(seed)
    setup_cudnn()
    gpu = setup_ddp()


    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    exp_name = f'train_{args.dataset}_{timestamp}'
    save_dir = Path('output', exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')

    main(args, gpu, save_dir)
    print(f"随机生成的初始种子：{seed}")
    cleanup_ddp()
