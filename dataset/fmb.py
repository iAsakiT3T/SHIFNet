import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import torch.nn as nn
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler

from torchvision.transforms import Normalize, Resize, ToTensor
from PIL import Image
import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor
class FMB_dataset(Dataset):
    """
    num_classes: 14
    """
    CLASSES = ["Road", "Sidewalk", "Building", "Traffic Light", "Traffic Sign", "Vegetation", "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"]

    PALETTE = torch.tensor([[70, 70, 70],
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
    
    def __init__(self, root: str = '/root/autodl-tmp/segment-anything-2/data/FMB', split: str = 'train', transform = None, modals = ['rgb', 'thermal'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.to_tensor = ToTensor()
        if split == 'val':
            split = 'test'
        self.files = sorted(glob.glob(os.path.join(*[root, split, 'Visible', '*.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        # if not self.files:
        #     raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = self.files[index].split("/")[-1].split(".")[0]
        rgb = str(self.files[index])
        thermal = rgb.replace('/Visible', '/Infrared')
        lbl_path = rgb.replace('/Visible', '/Label')
        text_path = rgb.replace('/Visible', '/text_embeddings')
        text_path = os.path.splitext(text_path)[0] + '.pt'

        sample = {}
        sample['rgb'] = self._open_image(rgb)
        H, W = sample['rgb'].shape[1:]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_image(thermal,is_x=True)

        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        label[label==255] = 0
        label -= 1
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        #text_embeddings = torch.load(text_path)
        return {
            'image':sample[0],#torch.Size([3, 1024, 1024])
            'x' : sample[1],
            'label' : label,#torch.Size([1024, 1024])
            'text_path': text_path,
            'item_name' :item_name
        }

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def _open_image(self, file, is_x=False):
        image = Image.open(file)
        
        if is_x:
            # 对于红外图像，保持原始的单通道图像数据
            image = image.convert('L')  # 确保它是单通道的
            image = np.array(image)  # 转换为 numpy 数组，形状为 (H, W)
            # 复制单通道图像成3通道
            image = np.stack((image,) * 3, axis=-1)  # 现在的形状是 (H, W, 3
        else:
            image = np.array(image.convert("RGB"))  # 转换为RGB格式

        return self.to_tensor(image)


    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        img, mask = sample['img'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample

if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = Compose([ 
                Resize((1024,1024)),
                Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

    trainset = FMB(transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

    for i, sample in enumerate(trainloader):
        print("Unique values in 'image':", torch.unique(sample['image']))
        print("Unique values in 'x':", torch.unique(sample['x']))
        print("Unique values in 'label':", torch.unique(sample['label']))