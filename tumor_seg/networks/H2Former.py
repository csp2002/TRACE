import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform
from .H2Former_utils import *
from .vit_seg_modeling import TRACE


            
class Res34_Swin_MS(nn.Module):    ###  low resolution + Multi-scale
    def __init__(self,image_size, block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(Res34_Swin_MS, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.swin_layers = nn.ModuleList()
        embed_dim = 64
        self.num_layers = 4
        self.image_size = image_size
        depths=[2, 2, 2, 2]
        num_heads=[2, 4, 8, 16]
        window_size = self.image_size// 16
        self.mlp_ratio = 4.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size//2,self.image_size//2]
        
        patch_size=[2, 4, 8, 16]
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        self.MS2 = PatchMerging(64)
        self.MS3 = PatchMerging(128)
        self.MS4 = PatchMerging(256)
        
        for i_layer in range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)
            
        channels = [64,128,256,512]
        self.decode4 = Decoder(channels[3],channels[2])
        self.decode3 = Decoder(channels[2],channels[1])
        self.decode2 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels[0], num_classes, kernel_size=1,bias=False))

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        encoder = []
        ms1 = self.patch_embed(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        #print(x.shape,ms1.shape)
        x = x.flatten(2).transpose(1, 2)
        x = x+ms1
        
        x = self.swin_layers[0] (x)
        B, L, C = x.shape
        ms2 = self.MS2(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        encoder.append(x)
        
        x = self.layer2(x)
        x = x+ms2
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[1] (x)
        B, L, C = x.shape
        ms3 = self.MS3(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        encoder.append(x)
        
        x = self.layer3(x)
        x = x+ms3
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[2]  (x)
        B, L, C = x.shape
        ms4 = self.MS4(x)
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        encoder.append(x)
        
        x = self.layer4(x)
        x = x+ms4
        x = x.flatten(2).transpose(1, 2)
        x = self.swin_layers[3]  (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        encoder.append(x)
        
        d4 = self.decode4(encoder[3], encoder[2]) 
        d3 = self.decode3(d4, encoder[1]) 
        d2 = self.decode2(d3, encoder[0]) 
        out = self.decode0(d2)    
        return out
    
def res34_swin_MS(image_size, num_class) :
    return Res34_Swin_MS(image_size, BasicBlock, [3, 4, 6, 3],num_classes = num_class)


class H2Former_ours(nn.Module):  
    def __init__(self, img_size=224, num_classes=2, refine_iters = 3, in_chan=1,
            detach_between_iters = True):
        super(H2Former_ours, self).__init__()
        self.H2Former =Res34_Swin_MS(img_size, BasicBlock, [3, 4, 6, 3],num_classes = num_classes)
        self.refinement_module = TRACE()
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # self.refinement_module.load_from(weights=np.load(config_small.pretrained_path))

    def forward(self, x,  x_ref, ref_gt):  #x: bs,1,224,224
        
        logits = self.H2Former(x)  #bs,2,224,224
        logits_ref = self.H2Former(x_ref)  #bs,2,224,224

        probs = torch.softmax(logits, dim=1)  # (bs, 2, 224, 224)
        ori_mask = probs[:, 1:2, :, :]        # 取出“前景”类别的概率，仍保持 shape 为 (bs, 1, 224, 224)

        probs_ref = torch.softmax(logits_ref, dim=1)
        ref_mask = probs_ref[:, 1:2, :, :]  # (bs, 1, 224, 224)

        # ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        # ref_mask = torch.argmax(logits_ref, dim=1).unsqueeze(1)  #bs,1,224,224

        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        # new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        target_image = torch.cat([x, ori_mask], dim=1)  #bs,2,224,224
        ref_image = torch.cat([x_ref, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        preds_all = []  
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        preds_all.append(final_pred)

        for it in range(1, max(1, self.refine_iters) ):
        # 把上一轮的输出变成下一轮的 target prediction
            next_mask = torch.softmax(final_pred, dim=1)[:, 1:2]   

            if self.detach_between_iters:
                next_mask = next_mask.detach()

            target_2ch = torch.cat([x, next_mask], dim=1)      # (B,2,H,W)
            # reference 一般保持不变；如果你希望同步更新 ref_pred，也可以把 ref_mask 改成 logits_ref 的迭代版
            # 这里保持 ref_3ch 不变：
            # ref_3ch = torch.cat([ref_image_raw, ref_mask, ref_gt], dim=1)

            final_pred = self.refinement_module(target_2ch, ref_image)
            preds_all.append(final_pred)
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return {"final": final_pred, "iters": preds_all}