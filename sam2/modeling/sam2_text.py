import torch
from torch import nn
import torch.nn.functional as F
import logging
from torch import distributed as dist
from .backbones import ImageEncoder, FpnNeck, Hiera, Text_decoder, add1Fusion
from .position_encoding import PositionEmbeddingSine
import argparse
from encoding.nn import SegmentationLosses

from .memory_attention import MemoryAttention ,MemoryAttentionLayer 
from .sam.transformer import RoPEAttention,Attention 

class Mix_RGBX(nn.Module):
    def __init__(
        self,
        args,
        configs,
        image_encoder: ImageEncoder,
        decoder : Text_decoder,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.
        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.args = args
        self.configs =configs
        self.image_encoder = image_encoder

        self.fuse_block = nn.ModuleList(add1Fusion() for i in range(4))
        self.decoder = decoder
        if dist.is_initialized():
            # Get the current process rank
            rank = dist.get_rank()
            # Set the device to the corresponding GPU based on the rank
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            # Default to a specific GPU if not in distributed training
            device = torch.device(f'cuda:{args.gpu_device}')
        if args.dataset == 'fmb' or args.dataset == 'pst':
            self.label_feature = torch.load(args.label_path).to(device)
        else:
            raise KeyError('Dont support dataset')

    @property
    def device(self):
        return self.image_pixel_mean.device

    def forward(self, batched_input):#
        batched_input = torch.stack([x for x in batched_input], dim=0)#.unsqueeze(1)#.squeeze(0)#.unsqueeze(0) # train .squeeze(0) # 0730 
        m,b,c,h,w = batched_input.size()
        batched_input = batched_input.reshape(m*b,c,h,w)
        backbone_out = self.image_encoder(batched_input)

        feat_s0 = backbone_out['backbone_fpn'][0] #+ backbone_out['vision_pos_enc'][0]
        feat_s1 = backbone_out['backbone_fpn'][1] #+ backbone_out['vision_pos_enc'][1]
        feat_s2 = backbone_out['backbone_fpn'][2] #+ backbone_out['vision_pos_enc'][2]
        src  = backbone_out['backbone_fpn'][3] #+ backbone_out['vision_pos_enc'][3]
        
        b,c,_,_ = src.shape
        modal_type = 2
        b = int(b/modal_type)
        src = src.view(modal_type,-1,src.size(1),src.size(2),src.size(3))
        feat_s0 = feat_s0.view(modal_type,-1,feat_s0.size(1),feat_s0.size(2),feat_s0.size(3))
        feat_s1 = feat_s1.view(modal_type,-1,feat_s1.size(1),feat_s1.size(2),feat_s1.size(3))
        feat_s2 = feat_s2.view(modal_type,-1,feat_s2.size(1),feat_s2.size(2),feat_s2.size(3))
        src_result = torch.zeros((src[0].size(0), src[0].size(1), src[0].size(2), src[0].size(3)), dtype=src[0].dtype, device=src[0].device)
        feat_s0_result = torch.zeros((feat_s0[0].size(0), feat_s0[0].size(1), feat_s0[0].size(2), feat_s0[0].size(3)), dtype=feat_s0[0].dtype, device=feat_s0[0].device)
        feat_s1_result = torch.zeros((feat_s1[0].size(0), feat_s1[0].size(1), feat_s1[0].size(2), feat_s1[0].size(3)), dtype=feat_s1[0].dtype, device=feat_s1[0].device)
        feat_s2_result = torch.zeros((feat_s2[0].size(0), feat_s2[0].size(1), feat_s2[0].size(2), feat_s2[0].size(3)), dtype=feat_s2[0].dtype, device=feat_s2[0].device)

        for i in range(b):
            src_result[i] = self.fuse_block[3] (src[0, i].unsqueeze(0), src[1, i].unsqueeze(0),self.label_feature)
            feat_s2_result[i] = self.fuse_block[2] (feat_s2[0, i].unsqueeze(0),  feat_s2[1, i].unsqueeze(0),self.label_feature)
            feat_s1_result[i] = self.fuse_block[1] (feat_s1[0, i].unsqueeze(0),  feat_s1[1, i].unsqueeze(0),self.label_feature)
            feat_s0_result[i] = self.fuse_block[0] (feat_s0[0, i].unsqueeze(0),  feat_s0[1, i].unsqueeze(0),self.label_feature)

            
        fuse_feats = [feat_s0_result, feat_s1_result, feat_s2_result, src_result]
        seg_predictions = self.decoder(fuse_feats, self.label_feature)

        return seg_predictions


def build_mix_rgbx_text(args, configs):
        if args.dataset == 'fmb':
            num_classes = 14
        elif args.dataset == 'pst':
            num_classes = 5
        else:
            raise ValueError("unsupport dataset")
        args.channels_list = configs[args.ms]['backbone_channel_list']
        args.d_model = configs[args.ms]['d_model']
        model = Mix_RGBX(
                args,
                configs,
                image_encoder=ImageEncoder(
                scalp=0,
                trunk = Hiera(args,
                              embed_dim=configs[args.ms]['embed_dim'],
                              num_heads=configs[args.ms]['num_heads'],
                              stages= configs[args.ms]['stages'],
                              global_att_blocks= configs[args.ms]['global_att_blocks'],
                              window_pos_embed_bkg_spatial_size= configs[args.ms]['window_pos_embed_bkg_spatial_size'],
                              window_spec= configs[args.ms]['window_spec']
                              ),
                neck =FpnNeck(position_encoding=PositionEmbeddingSine(
                                                num_pos_feats=configs[args.ms]['num_pos_feats'], 
                                                normalize=configs[args.ms]['normalize'], 
                                                scale=configs[args.ms]['scale'], 
                                                temperature= configs[args.ms]['temperature']),
                              d_model = configs[args.ms]['d_model'],
                              backbone_channel_list=configs[args.ms]['backbone_channel_list'],
                              fpn_top_down_levels= configs[args.ms]['fpn_top_down_levels'],
                              fpn_interp_model = configs[args.ms]['fpn_interp_model'],
                              ),
                ),
                decoder=Text_decoder(args),
                )
        ckpt_path = args.ckpt
        if ckpt_path is not None:
            state_dict = model.state_dict()
            sd = torch.load(ckpt_path, map_location="cpu")["model"]
            for key,value in sd.items():
                if "image_encoder" in key:
                    #if "mlp" not in key:
                    state_dict[key] = value
                    if "patch_embed.proj.weight" in key:
                        state_dict["image_encoder.trunk.x_patch_embed.proj.weight"] = value
                    if "patch_embed.proj.bias" in key:
                        state_dict["image_encoder.trunk.x_patch_embed.proj.bias"] = value
                    if "layers.0.weight" in key:
                        block_index = key.split('blocks.')[1].split('.')[0]
                        for layer in [0, 1]:
                            for j in range(3):  # 专家的索引 0, 1, 和 2
                                state_dict[f'image_encoder.trunk.blocks.{block_index}.moe_layers.{layer}.layers.0.experts.{j}.linear.weight'] = value
                    elif "layers.1.weight" in key:
                        block_index = key.split('blocks.')[1].split('.')[0]
                        for layer in [0, 1]:
                            for j in range(3):  # 专家的索引 0, 1, 和 2
                                state_dict[f'image_encoder.trunk.blocks.{block_index}.moe_layers.{layer}.layers.1.experts.{j}.linear.weight'] = value
                    elif "layers.0.bias" in key:
                        block_index = key.split('blocks.')[1].split('.')[0]
                        for layer in [0, 1]:
                            for j in range(3):  # 专家的索引 0, 1, 和 2
                                state_dict[f'image_encoder.trunk.blocks.{block_index}.moe_layers.{layer}.layers.0.experts.{j}.linear.bias'] = value
                    elif "layers.1.bias" in key:
                        block_index = key.split('blocks.')[1].split('.')[0]
                        for layer in [0, 1]:
                            for j in range(3):  # 专家的索引 0, 1, 和 2
                                state_dict[f'image_encoder.trunk.blocks.{block_index}.moe_layers.{layer}.layers.1.experts.{j}.linear.bias'] = value
            model.load_state_dict(state_dict, strict=False)
            logging.info("Loaded checkpoint sucessfully")
        if  True :
            id = 255
            criterion = SegmentationLosses(
                            se_loss=False, 
                            cos_loss=False,
                            aux=False, 
                            nclass=num_classes, 
                            se_weight=0.2, 
                            aux_weight=0.2, 
                            ignore_index=id, 
                        )
            print('use text drive semantic segmantation')
        return model, criterion






def build_vis_sam(configs,ckp, args):
        if args.dataset == 'fmb':
            num_classes = 14
        elif args.dataset == 'pst':
            num_classes = 5
        else:
            raise ValueError("unsupport dataset")
        args.channels_list = configs['l']['backbone_channel_list']
        args.d_model = configs['l']['d_model']
        model = Mix_RGBX(
                args,
                configs,
                image_encoder=ImageEncoder(
                scalp=0,
                trunk = Hiera(args,
                              embed_dim=configs['l']['embed_dim'],
                              num_heads=configs['l']['num_heads'],
                              stages= configs['l']['stages'],
                              global_att_blocks= configs['l']['global_att_blocks'],
                              window_pos_embed_bkg_spatial_size= configs['l']['window_pos_embed_bkg_spatial_size'],
                              window_spec= configs['l']['window_spec']
                              ),
                neck =FpnNeck(position_encoding=PositionEmbeddingSine(
                                                num_pos_feats=configs['l']['num_pos_feats'], 
                                                normalize=configs['l']['normalize'], 
                                                scale=configs['l']['scale'], 
                                                temperature= configs['l']['temperature']),
                              d_model = configs['l']['d_model'],
                              backbone_channel_list=configs['l']['backbone_channel_list'],
                              fpn_top_down_levels= configs['l']['fpn_top_down_levels'],
                              fpn_interp_model = configs['l']['fpn_interp_model'],
                              ),
                ),
                decoder=Text_decoder(args),
                )
        
        sd = torch.load(ckp, map_location="cpu")#["model_state_dict"]
        state_dict = model.state_dict()
        print(sd.keys())
        print("----------------------------")
        model.load_state_dict(sd, strict=True)
        
        if True:
            criterion = SegmentationLosses(
                            se_loss=False, 
                            cos_loss=False,
                            aux=False, 
                            nclass=num_classes, 
                            se_weight=0.2, 
                            aux_weight=0.2, 
                            ignore_index=255, 
                        )
            print('use text drive semantic segmantation')
        else:
            criterion = Dice_CrossEntropy_Loss()
        return model, criterion