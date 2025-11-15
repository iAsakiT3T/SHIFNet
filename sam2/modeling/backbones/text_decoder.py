import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmcv.cnn import ConvModule
import warnings
import numpy as np
import torch.nn.init as init
from torch import  Tensor
import torchvision
torchvision.models.inception_v3
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            #size=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

#-------------------------------------+++++++++++++++++++--------------------------------
from typing import Any, Callable, List, Optional, Tuple
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionC(nn.Module):
    def __init__(
        self, in_channels: int, conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        c7 = 128
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 64, kernel_size=(1, 7), padding=(0, 3))

        self.final_conv = conv_block(64, 8, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        outputs = branch1x1 + branch7x7 + branch7x7dbl

        outputs = self.final_conv(outputs)
        return outputs


class GlobalFeatureConcat2D(nn.Module):
    def __init__(self, channels=256):
        super(GlobalFeatureConcat2D, self).__init__()
        self.channels = channels

        self.project = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1), #nn.Conv2d(768, 512, 1),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
        )
    def forward(self, x, contextFeature):

        # 3. 在通道维度上拼接全局特征
        x = torch.cat((x, contextFeature), dim=1)  # [batch_size, channels * 2, height, width]
        
        x = self.project(x)
        return x



class ProjectReadout(nn.Module):
    def __init__(self, in_features):
        super(ProjectReadout, self).__init__()

        self.project = nn.Sequential(MLP(in_features, in_features//2))#, nn.GELU())

    def forward(self, x):
        out = self.project(x)
        
        return out

class LearnedUpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnedUpsamplingModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x



class ContextModule(nn.Module):
    def __init__(self, input_channels):
        super(ContextModule, self).__init__()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))
        self.conv1_1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.inception = InceptionC(64)

        self.upsample = LearnedUpsamplingModule(8, 256)

    def forward(self, x):
        b, c, h ,w =x.size()

        x = self.adaptive_pool(x)
        x = self.conv1_1(x)

        x = self.inception(x)

        x = self.upsample(x)

        return x


class Text_decoder(nn.Module):
    def __init__(self, args, in_chan=256):
        super(Text_decoder, self).__init__()

        self.ContextModule = nn.ModuleList()
        self.GlobalFeatureConcat2D = nn.ModuleList()
        self.postprocess1 = nn.ModuleList()
        self.postprocess2 = nn.ModuleList()
        n = len(args.channels_list)
        for i in range(n):
            current_GlobalFeatureConcat2D = GlobalFeatureConcat2D(channels=in_chan)
            postprocess1 = ProjectReadout(in_chan)
            contextModule = ContextModule(in_chan)
            postprocess2 = Interpolate(scale_factor=2**i, mode='bilinear', align_corners=False)#Postprocess2(i)#(args.d_model, i )
            
            self.GlobalFeatureConcat2D.append(current_GlobalFeatureConcat2D)
            self.postprocess1.append(postprocess1)
            self.postprocess2.append(postprocess2)
            self.ContextModule.append(contextModule)
        
        self.linear_fuse = nn.Conv2d(args.d_model*2, args.d_model, kernel_size=1)
        self.out_c =768
        self.head = nn.Conv2d(256, self.out_c, kernel_size=1)

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.output_conv =  nn.Sequential(
            Interpolate(scale_factor=4, mode="bilinear", align_corners=True),
        )

        self.dropout = nn.Dropout2d(0.1)
    def forward(self, fpn_fusion ,label_feature):
        n, _, h, w = fpn_fusion[0].shape
        
        out = []
        post1_out = []
        post2_out = []
        contextFeature = []
        
        for i in range(len(fpn_fusion)):
            contextFeature.append(self.ContextModule[i](fpn_fusion[i]))
        for i in range(len(fpn_fusion)):
            out.append(self.GlobalFeatureConcat2D[i](fpn_fusion[i], contextFeature[i]))#h*c*h*w
        for i in range(len(fpn_fusion)):
            post1_out.append(self.postprocess1[i](out[i]).permute(0,2,1).reshape(n, -1, out[i].shape[2], out[i].shape[3]))
        for i in range(len(fpn_fusion)):
            post2_out.append( self.postprocess2[i](post1_out[i]))
        path_1 = self.linear_fuse(torch.cat([post2_out[3], post2_out[2], post2_out[1], post2_out[0]], dim=1))

        logit_scale = self.logit_scale.exp()

        image_features = self.head(path_1)
        image_features = self.dropout(image_features)
        text_features = label_feature

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)#57600*512

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = logit_scale * image_features @ text_features.t()#logit_scale * image_features.half() @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)#相似性得分

        out = self.output_conv(out)#14*480*480

        return out
