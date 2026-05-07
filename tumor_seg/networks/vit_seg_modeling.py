from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2, ResNetV3, ResNetV4, ResNetV5, DualEncoderWithInteraction
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math
logger = logging.getLogger(__name__)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            
            grid_size = config.patches["grid"]
            # print('config:',config)
            # print('grid_size:',grid_size)
            # print('img_size:',img_size)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        # raise Exception('Please set the grid size in the config file.')
        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        
        out_channels = decoder_channels
        

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
        # print('in_channels:',in_channels)
        # print('out_channels:',out_channels)
        # print('skip_channels:',skip_channels)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # print('hidden_states:',hidden_states.shape)    #24,196,768
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        # print('x1:',x.shape)
        x = x.contiguous().view(B, hidden, h, w)  #24,768,14,14
        # print('x2:',x.shape)
        x = self.conv_more(x)  #24,512,14,14
        # print('x3:',x.shape)
        #features[0]:24,512,28,28; features[1]:24,256,56,56; features[2]:24,64,112,112
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            #x.shape:24,256,28,28; 24,64,56,56; 24,16,112,112; 24,16,224,224
        #     print('d_x'+str(i),x.shape)
        # print('final_x:',x.shape)
        return x
class VisionTransformer(nn.Module):   #original in TransUNet
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
class LinearAlign(nn.Module):
    """1x1 Conv + BN（无激活），轻量通道对齐"""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
    def forward(self, x):
        return self.bn(self.conv(x))
def compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True):
    """
    ref_mask: (B,1,224,224)  概率（前景通道）
    ref_gt:   (B,1,224,224)  {0,1}
    return: (conf_112, conf_56, conf_28)  三个尺度
    """
    eps = 1e-6
    p = ref_mask.clamp(eps, 1 - eps)
    # 与GT一致性
    conf_gt = 1.0 - (p - ref_gt).abs()                    # [0,1]
    # 熵置信度（1 - 归一化熵）
    H = -(p*torch.log(p) + (1-p)*torch.log(1-p))          # [0, log2]
    conf_ent = 1.0 - (H / (torch.log(torch.tensor(2.0, device=p.device))))
    conf = torch.clamp(w_g * conf_gt + (1 - w_g) * conf_ent, 0.0, 1.0)  # (B,1,224,224)
    if detach:
        conf = conf.detach()

    conf_112 = F.interpolate(conf, size=(112,112), mode='bilinear', align_corners=True)
    conf_56  = F.interpolate(conf, size=(56,56),   mode='bilinear', align_corners=True)
    conf_28  = F.interpolate(conf, size=(28,28),   mode='bilinear', align_corners=True)
    return conf_112, conf_56, conf_28
class Decoder_AlignPlusConf(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 轻量对齐
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)                        # (B,512,14,14)

        x = self.up3(x)                                  # (B,512,28,28)
        t3a = self.t_align3(t3); r3a = self.r_align3(r3) # (B,512,28,28)
        r3_use = r3a * conf_28
        x = self.dec3(torch.cat([x, t3a, r3_use], dim=1))# -> (B,256,28,28)

        x = self.up2(x)                                  # (B,256,56,56)
        t2a = self.t_align2(t2); r2a = self.r_align2(r2)
        r2_use = r2a * conf_56
        x = self.dec2(torch.cat([x, t2a, r2_use], dim=1))# -> (B,128,56,56)

        x = self.up1(x)                                  # (B,128,112,112)
        t1a = self.t_align1(t1); r1a = self.r_align1(r1)
        r1_use = r1a * conf_112
        x = self.dec1(torch.cat([x, t1a, r1_use], dim=1))# -> (B,64,112,112)

        x = self.final_up(x)                             # (B,64,224,224)
        return self.out_conv(x)
from timm import create_model
class InputAdapter(nn.Module):
    """把 n 通道的 [pred, image(, gt)] 映射到 3 通道，以便完整复用 ImageNet 预训练。
       结构：Conv3x3 → BN → GELU → Conv1x1 （轻量、稳定）
    """
    def __init__(self, in_ch, mid_ch=16, out_ch=3, use_bn=True, use_act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn: layers += [nn.BatchNorm2d(mid_ch)]
        if use_act: layers += [nn.GELU()]
        layers += [nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class ResNetEncoderAligned(nn.Module):
    """
    用 timm 的 ResNet（此处改成 resnet18）做 backbone，
    并投影成你原 decoder 需要的接口：
      x        = (B, 1024, 14, 14)   ← 来自 layer3(14x14) 投到 1024
      features = [ (B,512,28,28), (B,256,56,56), (B,64,112,112) ]
                 ↑    layer2         ↑ layer1         ↑ stem 上采样
    """
    def __init__(self, variant='resnet18', in_chans=3, pretrained=True, keep_rgb_weights=None):
        super().__init__()
        if keep_rgb_weights is None:
            keep_rgb_weights = (pretrained and in_chans != 3)
        self.input_proj = InputAdapter(in_chans, out_ch=3) if keep_rgb_weights else None
        backbone_in = 3 if keep_rgb_weights else in_chans

        # 关键：这里把 variant 设为 resnet18
        self.backbone = create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # [112, 56, 28, 14]
            in_chans=backbone_in
        )
        chs = [fi['num_chs'] for fi in self.backbone.feature_info]  # resnet18: [64, 64, 128, 256]

        # 对齐到你 decoder 期望的通道
        self.to_064_112 = nn.Conv2d(chs[0],  64, 1, bias=False)    # stem(112) → 64
        self.to_256_56  = nn.Conv2d(chs[1], 256, 1, bias=False)    # layer1(56) → 256
        self.to_512_28  = nn.Conv2d(chs[2], 512, 1, bias=False)    # layer2(28) → 512
        self.to_1024_14 = nn.Conv2d(chs[3], 1024,1, bias=False)    # layer3(14) → 1024

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)           # n→3（仅在需要复用3ch预训练时）
        f112, f56, f28, f14 = self.backbone(x)  # timm 顺序: [112,56,28,14]

        # 112：stem → 64
        f112 = self.to_064_112(f112)
        # 56：layer1 → 256
        f56  = self.to_256_56(f56)
        # 28：layer2 → 512
        f28  = self.to_512_28(f28)
        # 14：layer3 → 1024（作为瓶颈 x）
        x14  = self.to_1024_14(f14)

        # 你的 decoder 期望顺序：[512@28, 256@56, 64@112]
        features = [f28, f56, f112]
        return x14, features
class target_encoder_resnet(nn.Module):
    def __init__(self, config, img_size=224, in_chans=2, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=True)
    def forward(self, x): return self.enc(x)
class reference_encoder_resnet(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=False)
    def forward(self, x): return self.enc(x)
class TRACE(nn.Module):   #
    def __init__(self,  img_size=224, num_classes=2, pretrained=True):
        super(TRACE, self).__init__()
        self.num_classes = num_classes
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_resnet(img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        conf_112, conf_56, conf_28 = compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True)
        logits = self.decoder(
            features_tar[2], features_tar[1], features_tar[0], x_tar,
            features_ref[2], features_ref[1], features_ref[0], x_ref,
            confs=(conf_112, conf_56, conf_28)
)

        
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
#         logits = self.decoder(
#     features_tar[2], features_tar[1], features_tar[0], x_tar,
#     features_ref[2], features_ref[1], features_ref[0], x_ref
# )

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits
class TransUNet_ours(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(TransUNet_ours, self).__init__()
        self.num_classes = num_classes   #csp: seems no use
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.refinement_module = TRACE(config_small)
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # self.refinement_module.load_from(weights=np.load(config_small.pretrained_path))

    def forward(self, x,  x_ref, ref_gt):  #x: bs,1,224,224
        # print('x1:',x.shape)
        ori_image = x
        ref_image = x_ref
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if x_ref.size()[1] == 1:
            x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        x_ref, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        x_ref = self.decoder(x_ref, features_ref)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        logits_ref = self.segmentation_head(x_ref)  #bs,2,224,224

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
        target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        preds_all = []  
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        preds_all.append(final_pred)

        for it in range(1, max(1, self.refine_iters) ):
        # 把上一轮的输出变成下一轮的 target prediction
            next_mask = torch.softmax(final_pred, dim=1)[:, 1:2]   

            if self.detach_between_iters:
                next_mask = next_mask.detach()

            target_2ch = torch.cat([ori_image, next_mask], dim=1)      # (B,2,H,W)
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

        # return final_pred

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
from math import pi
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
    'R18-ViT-S_16': configs.get_r18_s16_config(),
}
