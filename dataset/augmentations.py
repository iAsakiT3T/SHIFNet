import torchvision.transforms.functional as TF 
import random
from typing import Tuple, List, Union, Tuple, Optional
import torch.nn.functional as F
from torchvision.transforms import  Normalize
from PIL import Image
import math

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        img, mask = sample['rgb'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['rgb'] = TF.adjust_brightness(sample['rgb'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['rgb'] = TF.adjust_contrast(sample['rgb'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['rgb'] = TF.adjust_saturation(sample['rgb'], self.saturation)
        return sample
    



class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['rgb'] = TF.gaussian_blur(sample['rgb'], self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            for k, v in sample.items():
                if k != 'object_mask':
                    sample[k] = TF.hflip(v)
                else:
                    for id, m in v.items():
                        sample['object_mask'][id] = TF.hflip(m)
            return sample
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        H, W = sample['rgb'].shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            elif k == 'object_mask':
                for id, m in v.items():
                    sample['object_mask'][id] = TF.resize(m[None], (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['rgb'].shape[1] - tH, 0)
        margin_w = max(sample['rgb'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            if k != 'object_mask':
                sample[k] = v[:, y1:y2, x1:x2]
            else:
                for id, m in v.items():
                    sample['object_mask'][id] = m[:, y1:y2, x1:x2]

        # pad the image
        if sample['rgb'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['rgb'].shape[2], tH - sample['rgb'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                elif k == 'object_mask':
                    for id, m in v.items():
                        sample['object_mask'][id] = TF.pad(m, padding, fill=0)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample
    



class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size.
                If size is a sequence, the output size will be matched to this.
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, sample: list) -> list:
        H, W = sample["rgb"].shape[1:]

        # scale the image
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        for k, v in sample.items():
            if k == "mask":
                sample[k] = TF.resize(
                    v.unsqueeze(0), (nH, nW), TF.InterpolationMode.NEAREST
                ).squeeze(0)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
        # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32

        for k, v in sample.items():
            if k == "mask":
                sample[k] = TF.resize(
                    v.unsqueeze(0), (alignH, alignW), TF.InterpolationMode.NEAREST
                ).squeeze(0)
            else:
                sample[k] = TF.resize(
                    v, (alignH, alignW), TF.InterpolationMode.BILINEAR
                )
        # img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return sample
class normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.normalize = Normalize(self.mean, self.std)
    def __call__(self, sample: list) -> list:
        for k, v in sample.items():
            if k == 'rgb':
                sample[k] = sample[k].float()
                sample[k]  = self.normalize(sample[k])
            else:
                continue

        return sample

# class Normalize:
#     def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample: list) -> list:
#         for k, v in sample.items():
#             if k == 'mask':
#                 continue
#             elif k == 'rgb':
#                 sample[k] = sample[k].float()
#                 sample[k] /= 255
#                 sample[k] = TF.normalize(sample[k], self.mean, self.std)
#             else:
#                 sample[k] = sample[k].float()
#                 sample[k] /= 255
        
#         return sample   


def train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        RandomColorJitter(p=0.2), # 
        RandomHorizontalFlip(p=0.5), #
        RandomGaussianBlur((3, 3), p=0.2), #
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill), 
        #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def val_augmentation(size: Union[int, Tuple[int], List[int]]):#[0.449, 0.449, 0.449], [0.226, 0.226, 0.226]
    return Compose([
        Resize(size),
        #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])