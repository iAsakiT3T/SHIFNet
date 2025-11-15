import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import  Tensor
import numbers
import copy
from einops import rearrange

from mmcv.cnn import ConvModule
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor



class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x




class add1Fusion(nn.Module):
    def __init__(self, layer = CXBlock(256), num_layers = 2):
        super(add1Fusion, self).__init__()
        self.layers = get_clones(layer, num_layers)
        self.out_proj = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.rgb_proj = nn.Sequential(
            nn.Linear(256,768),
            nn.ReLU(),
            nn.Linear(768,768)
        )
        self.x_proj = nn.Sequential(
            nn.Linear(256,768),
            nn.ReLU(),
            nn.Linear(768,768)
        )

    def forward(self, rgb, x, label_feature):
        #out = self.fuse_blocks(rgb + x)
        b, c, h, w =rgb.shape
        text_features = label_feature.mean(dim=0, keepdim=True)
        # 展开特征图
        rgb_features = rgb.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        x_features = x.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)

        rgb_features = self.rgb_proj(rgb_features)
        x_features = self.x_proj(x_features)

        rgb_features = rgb_features / rgb_features.norm(dim=-1, keepdim=True)
        x_features = x_features / x_features.norm(dim=-1, keepdim=True)
        rgb_weight = rgb_features @ text_features.t()
        x_weight = x_features @ text_features.t()
        # 调整形状
        rgb_weight = rgb_weight.view(b, h, w, -1).permute(0, 3, 1, 2)
        x_weight = x_weight.view(b, h, w, -1).permute(0, 3, 1, 2)
        weight = torch.cat([rgb_weight,x_weight],dim=1)
        weight = torch.softmax(weight, dim=1)
        weight_rgb, weight_depth = weight.chunk(2, dim=1)
        
        out = rgb*weight_rgb + x* weight_depth
        for layer in self.layers:
            out = layer(out)
        out = self.out_proj(out)

        return out

if __name__ == "__main__":
    # 假设输入维度为 B=2, C=64, H=128, W=128
    B, C, H, W = 2, 64, 128, 128
    input_tensor = torch.randn(B, C, H, W)  # 随机生成一个输入张量

    # 定义池化层和卷积核大小
    pool_sizes = [2, 4, 8]
    model = Conv_fuse(in_channels=C)

    # 前向传播
    output = model(input_tensor)

    # 输出结果
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output.shape)


