# coding=utf-8
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



class Embeddings2(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=5):  #csp,change 3 to 5
        super(Embeddings2, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
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

        if self.hybrid:
            self.hybrid_model = ResNetV3(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
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

class Embeddings3(nn.Module):   #used in the smaller segmentatio, no vit
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=5):  #csp,change 3 to 5
        super(Embeddings3, self).__init__()
        # self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        # if config.patches.get("grid") is not None:   # ResNet
        #     grid_size = config.patches["grid"]
        #     # print('grid_size:',grid_size)
        #     # print('img_size:',img_size)
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        #     n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        # if self.hybrid:
        self.hybrid_model = ResNetV3(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        # if self.hybrid:
        x, features = self.hybrid_model(x)
        # else:
        #     features = None
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return x, features

class target_encoder(nn.Module):   
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=2):  
        super(target_encoder, self).__init__()
        # self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        # if config.patches.get("grid") is not None:   # ResNet
        #     grid_size = config.patches["grid"]
        #     # print('grid_size:',grid_size)
        #     # print('img_size:',img_size)
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        #     n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        # if self.hybrid:
        self.hybrid_model = ResNetV4(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        # if self.hybrid:
        x, features = self.hybrid_model(x)
        # else:
        #     features = None
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return x, features


class reference_encoder(nn.Module):   
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):  
        super(reference_encoder, self).__init__()
        # self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        # if config.patches.get("grid") is not None:   # ResNet
        #     grid_size = config.patches["grid"]
        #     # print('grid_size:',grid_size)
        #     # print('img_size:',img_size)
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        #     n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        # if self.hybrid:
        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        # if self.hybrid:
        x, features = self.hybrid_model(x)
        # else:
        #     features = None
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return x, features


class shared_encoder(nn.Module):   
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=1):  
        super(shared_encoder, self).__init__()
        # self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        # if config.patches.get("grid") is not None:   # ResNet
        #     grid_size = config.patches["grid"]
        #     # print('grid_size:',grid_size)
        #     # print('img_size:',img_size)
        #     patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        #     patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        #     n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #     self.hybrid = False

        # if self.hybrid:
        self.hybrid_model = ResNetV5(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            # in_channels = self.hybrid_model.width * 16
        # self.patch_embeddings = Conv2d(in_channels=in_channels,
        #                                out_channels=config.hidden_size,
        #                                kernel_size=patch_size,
        #                                stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        # if self.hybrid:
        x, features = self.hybrid_model(x)
        # else:
        #     features = None
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x = x.flatten(2)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        # embeddings = self.dropout(embeddings)
        return x, features



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

class Transformer2(nn.Module):  #used in the smaller segmentation
    def __init__(self, config, img_size, vis):
        super(Transformer2, self).__init__()
        self.embeddings = Embeddings2(config, img_size=img_size)
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


class DecoderCup_no_skip(nn.Module):
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
        skip_channels = [0] * len(decoder_channels)  # 全部为0，表示无skip connection
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # print('hidden_states:',hidden_states.shape)    #24,196,384
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        # print('x1:',x.shape)  #24,384,14,14
        x = self.conv_more(x)
        # print('x2:',x.shape)  #24,512,14,14
        # raise Exception
        for decoder_block in self.blocks:
            x = decoder_block(x, skip=None)
        
        return x
    
class DecoderCup_no_skip_no_vit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            in_channels=1024,
            out_channels=head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = [0] * len(decoder_channels)  # 全部为0，表示无skip connection
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features=None):
        # B, n_patch, hidden = hidden_states.size()
        # # print('hidden_states:',hidden_states.shape)    #24,196,384
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        # print('x1:',x.shape)  #24,384,14,14
        x = self.conv_more(x)
        # print('x2:',x.shape)  #24,512,14,14
        # raise Exception
        for decoder_block in self.blocks:
            x = decoder_block(x, skip=None)
        
        return x


class DecoderCup_no_vit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels =512
        self.conv_more = Conv2dReLU(
            in_channels=1024,
            out_channels=head_channels,
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

    def forward(self, x, features=None):
        # print('hidden_states:',hidden_states.shape)    #24,196,768
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # # print('x1:',x.shape)
        # x = x.contiguous().view(B, hidden, h, w)  #24,768,14,14
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


# Sinusoidal Positional Embedding
def get_sinusoidal_embedding(seq_len, dim):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len,1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, dim)

# Transformer-based Cross Attention with positional encoding
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, seq_len=784):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # 采用Sinusoidal位置编码初始化
        pos_embed = get_sinusoidal_embedding(seq_len, dim).unsqueeze(0)  # (1,seq_len,dim)
        self.pos_embed_q = nn.Parameter(pos_embed, requires_grad=True)
        self.pos_embed_kv = nn.Parameter(pos_embed.clone(), requires_grad=True)

    def forward(self, query_feat, key_feat):
        bs, c, h, w = query_feat.size()

        q = self.q_proj(query_feat.flatten(2).permute(0,2,1)) + self.pos_embed_q  # (bs,784,128)
        k = self.k_proj(key_feat.flatten(2).permute(0,2,1)) + self.pos_embed_kv   # (bs,784,128)
        v = self.v_proj(key_feat.flatten(2).permute(0,2,1)) + self.pos_embed_kv   # (bs,784,128)

        attn_output, _ = self.attn(q, k, v)  # (bs,784,128)

        attn_output = attn_output.permute(0,2,1).view(bs,c,h,w)  # (bs,128,28,28)

        return attn_output

# Channel Attention (SE block)
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x  # (bs,channels,H,W)

class ModuleM1(nn.Module):
    def __init__(self):
        super(ModuleM1, self).__init__()

        self.reduce_f0 = nn.Conv2d(512, 256, kernel_size=1)

        self.cross_attn_f0 = CrossAttention(dim=256, num_heads=4, seq_len=784)

        self.offset_conv_f1 = nn.Conv2d(768, 18, kernel_size=3, padding=1)
        self.offset_conv_f2 = nn.Conv2d(192, 18, kernel_size=3, padding=1)

        self.channel_attn_f1 = ChannelAttention(512)
        self.channel_attn_f2 = ChannelAttention(128)

        self.conv_fuse_f1 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.conv_fuse_f2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, f0_o, f1_o, f2_o, f0_r, f1_r, f2_r):
        # f0降维
        f0_o_reduced = self.reduce_f0(f0_o)  # (bs,256,28,28)
        f0_r_reduced = self.reduce_f0(f0_r)  # (bs,256,28,28)

        # Transformer-based S0
        S0 = self.cross_attn_f0(f0_o_reduced, f0_r_reduced)  # (bs,256,28,28)

        # f0→f1
        S0_up = F.interpolate(S0, scale_factor=2, mode='bilinear')  # (bs,256,56,56)
        offset_f1 = self.offset_conv_f1(torch.cat([S0_up, f1_o, f1_r], dim=1))  # (bs,18,56,56)
        aligned_f1_r = deform_conv2d(f1_r, offset_f1, weight=torch.randn(256,256,3,3).to(f1_r.device), padding=1)

        fused_f1 = self.conv_fuse_f1(self.channel_attn_f1(torch.cat([aligned_f1_r, f1_o], dim=1)))  # (bs,64,56,56)

        # f1→f2
        fused_f1_up = F.interpolate(fused_f1, scale_factor=2, mode='bilinear')  # (bs,64,112,112)
        offset_f2 = self.offset_conv_f2(torch.cat([fused_f1_up, f2_o, f2_r], dim=1))  # (bs,18,112,112)
        aligned_f2_r = deform_conv2d(f2_r, offset_f2, weight=torch.randn(64,64,3,3).to(f2_r.device), padding=1)

        fused_f2 = self.conv_fuse_f2(self.channel_attn_f2(torch.cat([aligned_f2_r, f2_o], dim=1)))  # (bs,64,112,112)

        # guidance
        guidance = F.interpolate(fused_f2, scale_factor=2, mode='bilinear')  # (bs,64,224,224)

        return guidance

class ModuleM1_v2(nn.Module):
    def __init__(self):
        super(ModuleM1_v2, self).__init__()

        self.reduce_f0 = nn.Conv2d(512, 256, kernel_size=1)

        self.cross_attn_f0 = CrossAttention(dim=256, num_heads=4, seq_len=784)

        self.offset_conv_f1 = nn.Conv2d(768, 18, kernel_size=3, padding=1)
        self.offset_conv_f2 = nn.Conv2d(192, 18, kernel_size=3, padding=1)

        self.channel_attn_f1 = ChannelAttention(512)
        self.channel_attn_f2 = ChannelAttention(128)

        self.conv_fuse_f1 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.conv_fuse_f2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, f0_o, f1_o, f2_o, f0_r, f1_r, f2_r):
        # f0降维
        f0_o_reduced = self.reduce_f0(f0_o)  # (bs,256,28,28)
        f0_r_reduced = self.reduce_f0(f0_r)  # (bs,256,28,28)

        # Transformer-based S0
        S0 = self.cross_attn_f0(f0_o_reduced, f0_r_reduced)  # (bs,256,28,28)

        # f0→f1
        S0_up = F.interpolate(S0, scale_factor=2, mode='bilinear')  # (bs,256,56,56)
        offset_f1 = self.offset_conv_f1(torch.cat([S0_up, f1_o, f1_r], dim=1))  # (bs,18,56,56)
        aligned_f1_r = deform_conv2d(f1_r, offset_f1, weight=torch.randn(256,256,3,3).to(f1_r.device), padding=1)

        fused_f1 = self.conv_fuse_f1(self.channel_attn_f1(torch.cat([aligned_f1_r, f1_o], dim=1)))  # (bs,64,56,56)

        # f1→f2
        fused_f1_up = F.interpolate(fused_f1, scale_factor=2, mode='bilinear')  # (bs,64,112,112)
        offset_f2 = self.offset_conv_f2(torch.cat([fused_f1_up, f2_o, f2_r], dim=1))  # (bs,18,112,112)
        aligned_f2_r = deform_conv2d(f2_r, offset_f2, weight=torch.randn(64,64,3,3).to(f2_r.device), padding=1)

        fused_f2 = self.conv_fuse_f2(self.channel_attn_f2(torch.cat([aligned_f2_r, f2_o], dim=1)))  # (bs,64,112,112)

        # guidance
        fused_f2_up = F.interpolate(fused_f2, scale_factor=2, mode='bilinear')  # (bs,64,224,224)

        return S0_up, fused_f1_up, fused_f2_up



# 标准Spatial Attention (CBAM的实现方式)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (bs,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (bs,1,H,W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (bs,2,H,W)
        attention_map = self.sigmoid(self.conv(concat))  # (bs,1,H,W)
        return x * attention_map  # 空间加权后的输出特征

class ModuleM2(nn.Module):
    def __init__(self):
        super(ModuleM2, self).__init__()

        # 特征扩展层（mask→feature）
        self.expand_original_mask = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.expand_reference_mask = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # 标准Spatial Attention层
        self.spatial_attn_original = SpatialAttention(kernel_size=7)
        self.spatial_attn_reference = SpatialAttention(kernel_size=7)

        # Confidence预测模块
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Refined mask预测模块
        self.refined_mask_predictor = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, original_mask, reference_mask, guidance):
        # original_mask：(bs,2,224,224)
        # reference_mask：(bs,1,224,224)
        # guidance：(bs,64,224,224)

        # Step 1: 特征扩展
        feat_original_mask = self.expand_original_mask(original_mask)      # (bs,32,224,224)
        feat_reference_mask = self.expand_reference_mask(reference_mask)  # (bs,32,224,224)

        # Step 2: 标准Spatial Attention融合 (使用guidance与mask特征concat后再做attention)
        feat_og = torch.cat([feat_original_mask, guidance], dim=1)  # (bs,96,224,224)
        feat_og = self.spatial_attn_original(feat_og)               # (bs,96,224,224)，空间加权

        feat_rg = torch.cat([feat_reference_mask, guidance], dim=1) # (bs,96,224,224)
        feat_rg = self.spatial_attn_reference(feat_rg)              # (bs,96,224,224)，空间加权

        # Step 3: Confidence预测
        confidence_input = torch.cat([feat_og, feat_rg], dim=1)     # (bs,192,224,224)
        confidence_map = self.confidence_predictor(confidence_input) # (bs,1,224,224)

        # Step 4: Refined mask预测
        refined_input = torch.cat([guidance, reference_mask], dim=1)  # (bs,65,224,224)
        refined_mask = self.refined_mask_predictor(refined_input)     # (bs,2,224,224)

        # Step 5: 动态融合
        final_mask = confidence_map * original_mask + (1 - confidence_map) * refined_mask  # (bs,2,224,224)

        return final_mask, confidence_map, refined_mask


class ModuleM2_no_confidence(nn.Module):
    def __init__(self):
        super(ModuleM2_no_confidence, self).__init__()

        # 特征扩展层（mask→feature）
        self.expand_original_mask = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.expand_reference_mask = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # 标准Spatial Attention层
        self.spatial_attn_original = SpatialAttention(kernel_size=7)
        self.spatial_attn_reference = SpatialAttention(kernel_size=7)

        # Confidence预测模块
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
            # nn.Sigmoid()
        )

        # Refined mask预测模块
        self.refined_mask_predictor = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, original_mask, reference_mask, guidance):
        # original_mask：(bs,2,224,224)
        # reference_mask：(bs,1,224,224)
        # guidance：(bs,64,224,224)

        # Step 1: 特征扩展
        feat_original_mask = self.expand_original_mask(original_mask)      # (bs,32,224,224)
        feat_reference_mask = self.expand_reference_mask(reference_mask)  # (bs,32,224,224)

        # Step 2: 标准Spatial Attention融合 (使用guidance与mask特征concat后再做attention)
        feat_og = torch.cat([feat_original_mask, guidance], dim=1)  # (bs,96,224,224)
        feat_og = self.spatial_attn_original(feat_og)               # (bs,96,224,224)，空间加权

        feat_rg = torch.cat([feat_reference_mask, guidance], dim=1) # (bs,96,224,224)
        feat_rg = self.spatial_attn_reference(feat_rg)              # (bs,96,224,224)，空间加权

        # Step 3: Confidence预测
        confidence_input = torch.cat([feat_og, feat_rg], dim=1)     # (bs,192,224,224)
        confidence_map = self.confidence_predictor(confidence_input) # (bs,1,224,224)

        # Step 4: Refined mask预测
        # refined_input = torch.cat([guidance, reference_mask], dim=1)  # (bs,65,224,224)
        # refined_mask = self.refined_mask_predictor(refined_input)     # (bs,2,224,224)

        # Step 5: 动态融合
        # final_mask = confidence_map * original_mask + (1 - confidence_map) * refined_mask  # (bs,2,224,224)

        return confidence_map








# CBAM标准Spatial Attention模块
class SpatialAttention_v2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_v2, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (bs,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (bs,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)      # (bs,2,H,W)
        x = self.conv(x)                              # (bs,1,H,W)
        return self.sigmoid(x)                        # (bs,1,H,W)

# 修正后的M2模块（CBAM标准Spatial Attention）
class ModuleM2_v2(nn.Module):
    def __init__(self):
        super(ModuleM2_v2, self).__init__()

        # Mask特征扩展 (2→32通道)
        self.expand_original_mask = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.expand_reference_mask = nn.Conv2d(2, 32, kernel_size=3, padding=1)

        # 标准CBAM Spatial Attention
        self.spatial_attention = SpatialAttention_v2(kernel_size=7)

        # Guidance特征对齐卷积（通道数调整）
        self.guidance_f0_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.guidance_f1_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.guidance_f2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 动态gate生成网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        # 最终预测网络
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, original_mask, reference_mask, guidance_f0, guidance_f1, guidance_f2):
        bs, _, H, W = original_mask.size()

        # Step 1: 统一reference_mask格式
        foreground_prob = reference_mask.float()                          # (bs,1,H,W)
        background_prob = 1 - foreground_prob                             # (bs,1,H,W)
        reference_mask_prob = torch.cat([background_prob, foreground_prob], dim=1)  # (bs,2,H,W)

        # Step 2: Mask特征扩展
        ori_feat = self.expand_original_mask(original_mask)               # (bs,32,H,W)
        ref_feat = self.expand_reference_mask(reference_mask_prob)        # (bs,32,H,W)

        # Step 3: 使用CBAM标准Spatial Attention
        concat_feat = torch.cat([ori_feat, ref_feat], dim=1)              # (bs,64,H,W)
        spatial_attn = self.spatial_attention(concat_feat)                # (bs,1,H,W)，CBAM标准
        guided_feat = ori_feat * spatial_attn                             # (bs,32,H,W)，逐位置加权

        # Step 4: Residual连接
        fused_mask_feat = guided_feat + ori_feat                          # (bs,32,H,W)

        # Step 5: 动态多尺度Guidance融合
        g0 = F.interpolate(self.guidance_f0_conv(guidance_f0), size=(H,W), mode='bilinear', align_corners=False)  # (bs,64,H,W)
        g1 = F.interpolate(self.guidance_f1_conv(guidance_f1), size=(H,W), mode='bilinear', align_corners=False)  # (bs,64,H,W)
        g2 = self.guidance_f2_conv(guidance_f2)                           # 已为(bs,64,H,W)

        gates = self.gate_conv(fused_mask_feat)                           # (bs,3,H,W)
        gates = F.softmax(gates, dim=1)

        fused_guidance = gates[:,0:1]*g0 + gates[:,1:2]*g1 + gates[:,2:]*g2 # (bs,64,H,W)

        # Step 6: 最终Refined Mask预测
        final_input = torch.cat([fused_mask_feat, fused_guidance], dim=1) # (bs,96,H,W)
        refined_mask = self.final_conv(final_input)                       # (bs,2,H,W)

        return refined_mask





class SimpleRefinementModule(nn.Module):
    def __init__(self):
        super(SimpleRefinementModule, self).__init__()

        # 只保留CNN精细调整部分
        self.refine_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, original_mask, pseudo_mask):
        # original_mask：(bs,2,224,224)，原网络输出概率
        # pseudo_mask：(bs,1,224,224)，伪标注mask，值为0或1

        # Step 1: 非常简单直接的pseudo mask扩展
        foreground = pseudo_mask.float()                  # (bs,1,224,224)
        background = 1.0 - foreground                     # (bs,1,224,224)
        pseudo_feat = torch.cat([background, foreground], dim=1)  # (bs,2,224,224)

        # Step 2: 拼接original_mask和pseudo_feat
        combined = torch.cat([original_mask, pseudo_feat], dim=1)  # (bs,4,224,224)

        # Step 3: CNN精细调整
        delta_mask = self.refine_conv(combined)           # (bs,2,224,224)

        # Step 4: 残差连接
        refined_mask = original_mask + delta_mask         # (bs,2,224,224)

        return refined_mask


class AdaptiveMaskRefinement(nn.Module):
    def __init__(self):
        super(AdaptiveMaskRefinement, self).__init__()

        # Mask特征扩展到32通道
        self.ori_feat_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pseudo_feat_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)

        # multi-scale特征对齐与融合
        self.f0_conv = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.f1_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.f2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fuse_multi_scale = nn.Conv2d(192, 64, kernel_size=3, padding=1)

        # 动态gate生成网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终refinement预测层
        self.refine_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, original_mask, pseudo_mask, f0, f1, f2):
        bs, _, H, W = original_mask.size()

        # Step 1: pseudo mask格式统一扩展
        pseudo_feat = torch.cat([1-pseudo_mask, pseudo_mask], dim=1)  # (bs,2,H,W)

        # mask特征初始提取
        ori_feat = F.relu(self.ori_feat_conv(original_mask))          # (bs,32,H,W)
        pseudo_feat = F.relu(self.pseudo_feat_conv(pseudo_feat))      # (bs,32,H,W)

        # Step 2: multi-scale特征对齐与融合
        f0_up = F.interpolate(self.f0_conv(f0), size=(56,56), mode='bilinear', align_corners=False) # (bs,64,56,56)
        f1_up = self.f1_conv(f1)                                                                    # (bs,64,56,56)
        f2_down = F.interpolate(self.f2_conv(f2), size=(56,56), mode='bilinear', align_corners=False)# (bs,64,56,56)
        multi_scale_feat = torch.cat([f0_up, f1_up, f2_down], dim=1)                                # (bs,192,56,56)
        multi_scale_feat = F.relu(self.fuse_multi_scale(multi_scale_feat))                          # (bs,64,56,56)
        multi_scale_feat = F.interpolate(multi_scale_feat, size=(H,W), mode='bilinear', align_corners=False) # (bs,64,H,W)

        # Step 3: 动态gate生成
        gate = self.gate_conv(multi_scale_feat)  # (bs,1,H,W), pseudo重要性权重

        # Step 4: 动态融合两个mask特征
        fused_feat = gate * pseudo_feat + (1 - gate) * ori_feat       # (bs,32,H,W)

        # Step 5: 残差预测最终mask
        delta_mask = self.refine_conv(fused_feat)                     # (bs,2,H,W)
        refined_mask = original_mask + delta_mask                     # (bs,2,H,W)

        return refined_mask


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

class My_VisionTransformer2(nn.Module):   #version 1, no confidence
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer2, self).__init__()
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
        self.fusion_module = ModuleM1()
        self.refinement_module = ModuleM2_no_confidence()

    def forward(self, x, x_ref, mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if x_ref.size()[1] == 1:
            x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        confidence_map = self.refinement_module(logits, mask_ref, guidance)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return confidence_map

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


class My_VisionTransformer(nn.Module):  #version 1
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer, self).__init__()
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
        self.fusion_module = ModuleM1()
        self.refinement_module = ModuleM2()

    def forward(self, x, x_ref, mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if x_ref.size()[1] == 1:
            x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        final_pred, confidence_map, refined_mask = self.refinement_module(logits, mask_ref, guidance)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred, refined_mask

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



class My_VisionTransformer_v2(nn.Module):  #version 2
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v2, self).__init__()
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
        self.fusion_module = ModuleM1_v2()
        self.refinement_module = ModuleM2_v2()

    def forward(self, x, x_ref, mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        if x_ref.size()[1] == 1:
            x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        guid_0, guid_1, guid_2 = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        final_pred = self.refinement_module(logits, mask_ref, guid_0, guid_1, guid_2)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class My_VisionTransformer_v3(nn.Module):  #version 1
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v3, self).__init__()
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
        self.refinement_module = SimpleRefinementModule()

    def forward(self, x,  mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # if x_ref.size()[1] == 1:
        #     x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        # _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        final_pred = self.refinement_module(logits, mask_ref)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class My_VisionTransformer_v4(nn.Module):  #version 1
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v4, self).__init__()
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
        self.refinement_module = AdaptiveMaskRefinement()

    def forward(self, x,  mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # if x_ref.size()[1] == 1:
        #     x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        # _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        final_pred = self.refinement_module(logits, mask_ref,f0_o, f1_o, f2_o)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v3(nn.Module):
    def __init__(self):
        super(refinement_v3, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # -------- Encoder --------
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = conv_block(256, 512)

        # -------- Decoder --------
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)

        # -------- Output --------
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)             # (bs, 64, 224, 224)
        e2 = self.enc2(self.pool1(e1))# (bs, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))# (bs, 256, 56, 56)

        # ----- Bottleneck -----
        b = self.bottleneck(self.pool3(e3))  # (bs, 512, 28, 28)

        # ----- Decoder -----
        d3 = self.upconv3(b)              # (bs, 256, 56, 56)
        d3 = torch.cat([e3, d3], dim=1)   # (bs, 512, 56, 56)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)             # (bs, 128, 112, 112)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)             # (bs, 64, 224, 224)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)              # (bs, 2, 224, 224)
        return out

class My_VisionTransformer_v5(nn.Module):  #v3o1
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v5, self).__init__()
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
        self.refinement_module = refinement_v3()

    def forward(self, x,  mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        ori_image = x
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # if x_ref.size()[1] == 1:
        #     x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        # _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        new_image = torch.cat([ori_image, mask_ref, ori_mask], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

class ModulationNet(nn.Module):
    def __init__(self):
        super(ModulationNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Conv2d(32, 1, 3, padding=1)

        self.mask_proj = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, original_mask, pseudo_mask):
        x = torch.cat([original_mask, pseudo_mask], dim=1)  # (bs,2,224,224)

        features = self.feature_extractor(x)  # (bs,32,224,224)

        foreground_mask = torch.max(original_mask, pseudo_mask)  # (bs,1,224,224)

        mask_bias = self.mask_proj(foreground_mask)  # (bs,1,224,224)

        out = self.bottleneck(features)
        out = out + mask_bias
        modulation_map = torch.sigmoid(out)

        return modulation_map, foreground_mask

class refinement_v4(nn.Module):
    def __init__(self):
        super(refinement_v4, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # -------- Encoder --------
        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = conv_block(256, 512)

        # -------- Decoder --------
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)

        # -------- Output --------
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)             # (bs, 64, 224, 224)
        e2 = self.enc2(self.pool1(e1))# (bs, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))# (bs, 256, 56, 56)

        # ----- Bottleneck -----
        b = self.bottleneck(self.pool3(e3))  # (bs, 512, 28, 28)

        # ----- Decoder -----
        d3 = self.upconv3(b)              # (bs, 256, 56, 56)
        d3 = torch.cat([e3, d3], dim=1)   # (bs, 512, 56, 56)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)             # (bs, 128, 112, 112)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)             # (bs, 64, 224, 224)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)              # (bs, 2, 224, 224)
        return out
    
class My_VisionTransformer_v6(nn.Module):  #v3o2
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6, self).__init__()
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
        self.modulation_net = ModulationNet()
        self.refinement_module = refinement_v4()

    def forward(self, x,  mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        
        ori_image = x
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # if x_ref.size()[1] == 1:
        #     x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        # _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        modulation_map, foreground_mask = self.modulation_net(ori_mask, mask_ref)

        enhanced_image = ori_image * modulation_map  # (bs,1,224,224)

        
        final_pred = self.refinement_module(enhanced_image)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred, foreground_mask, modulation_map, mask_ref

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





# ---------------------------------------
# Spatial Transformer (Flow-based warping)
# ---------------------------------------
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create a mesh grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')  # 注意PyTorch 2.x后需加indexing
        grid = torch.stack(grids)  # (2, H, W), [y,x]顺序
        grid = grid.unsqueeze(0).float()  # (1, 2, H, W)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        src: (bs, C, H, W) -> ori_mask
        flow: (bs, 2, H, W) -> x, y displacement
        """
        new_locs = self.grid + flow  # (bs, 2, H, W)
        shape = flow.shape[2:]  # (H, W)

        # Normalize new_locs to [-1,1] for grid_sample
        new_locs[:, 0, :, :] = 2.0 * (new_locs[:, 0, :, :] / (shape[1] - 1) - 0.5)  # x direction
        new_locs[:, 1, :, :] = 2.0 * (new_locs[:, 1, :, :] / (shape[0] - 1) - 0.5)  # y direction

        # Rearrange to (bs, H, W, 2) for grid_sample
        new_locs = new_locs.permute(0, 2, 3, 1)  # (bs, H, W, 2)

        # Perform warping
        warped = F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True)

        return warped

# ---------------------------------------
# Flow Predictor Network
# ---------------------------------------
class FlowPredictor(nn.Module):
    def __init__(self, in_channels=2):  # ori_image(1) + ref_mask(1)
        super(FlowPredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 56x56
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),   # 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)             # output flow: 2 channels (dx, dy)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # (bs, 2, 224, 224)

# ---------------------------------------
# Full Refinement Module
# ---------------------------------------
class refinement_v5(nn.Module):
    def __init__(self, size=(224, 224)):
        super(refinement_v5, self).__init__()
        self.flow_predictor = FlowPredictor(in_channels=2)  # ori_image + ref_mask
        self.spatial_transformer = SpatialTransformer(size=size, mode='bilinear')

    def forward(self, ori_image, ref_mask, ori_mask):
        """
        ori_image: (bs, 1, 224, 224)
        ref_mask: (bs, 1, 224, 224)
        ori_mask: (bs, 2, 224, 224)  # softmax probabilities
        """

        # Step 1: Predict flow from (ori_image + ref_mask)
        flow_input = torch.cat([ori_image, ref_mask], dim=1)  # (bs, 2, 224, 224)
        flow = self.flow_predictor(flow_input)  # (bs, 2, 224, 224)

        # Step 2: Warp ori_mask using predicted flow
        refined_mask = self.spatial_transformer(ori_mask, flow)  # (bs, 2, 224, 224)

        return refined_mask, flow




class My_VisionTransformer_v7(nn.Module):  #v3o2
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v7, self).__init__()
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
        self.modulation_net = ModulationNet()
        self.refinement_module = refinement_v5()

    def forward(self, x,  mask_ref):  #x: bs,1,224,224
        # print('x1:',x.shape)
        
        ori_image = x
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # if x_ref.size()[1] == 1:
        #     x_ref = x_ref.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden) bs,196,768
        # _, _, features_ref = self.transformer(x_ref)  # (B, n_patch, hidden) bs,196,768
        f0_o, f1_o, f2_o = features[0], features[1], features[2]
        # f0_r, f1_r, f2_r = features_ref[0], features_ref[1], features_ref[2]
        # guidance = self.fusion_module(f0_o, f1_o, f2_o, f0_r, f1_r, f2_r)  # (bs,64,224,224)
        # print("x2:",x.shape)
        # print('len:',len(features))
        # print('feature0:',features[0].shape)
        # print('feature1:',features[1].shape)
        # print('feature2:',features[2].shape)

        x = self.decoder(x, features)  #bs,16,224,224
        # print('x3:',x.shape)
        logits = self.segmentation_head(x)  #bs,2,224,224
        # ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        # # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # # raise Exception
        # modulation_map, foreground_mask = self.modulation_net(ori_mask, mask_ref)

        # enhanced_image = ori_image * modulation_map  # (bs,1,224,224)

        
        final_pred, flow = self.refinement_module(ori_image, mask_ref, logits)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred, mask_ref, flow

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





class refinement_v6(nn.Module):  #used in v5
    def __init__(self):
        super(refinement_v6, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def up_conv(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # -------- Encoder --------
        self.enc1 = conv_block(5, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = conv_block(256, 512)

        # -------- Decoder --------
        self.upconv3 = up_conv(512, 256)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64)
        self.dec1 = conv_block(128, 64)

        # -------- Output --------
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)             # (bs, 64, 224, 224)
        e2 = self.enc2(self.pool1(e1))# (bs, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))# (bs, 256, 56, 56)

        # ----- Bottleneck -----
        b = self.bottleneck(self.pool3(e3))  # (bs, 512, 28, 28)

        # ----- Decoder -----
        d3 = self.upconv3(b)              # (bs, 256, 56, 56)
        d3 = torch.cat([e3, d3], dim=1)   # (bs, 512, 56, 56)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)             # (bs, 128, 112, 112)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)             # (bs, 64, 224, 224)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)              # (bs, 2, 224, 224)
        return out



class My_VisionTransformer_v8(nn.Module):  #v5
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v8, self).__init__()
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
        self.refinement_module = refinement_v6()

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
        ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        ref_mask = torch.argmax(logits_ref, dim=1).unsqueeze(1)  #bs,1,224,224

        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v7(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v7, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer2(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    # def load_from(self, weights):
    #     with torch.no_grad():

    #         res_weight = weights
    #         self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
    #         self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

    #         self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
    #         self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

    #         posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

    #         posemb_new = self.transformer.embeddings.position_embeddings
    #         if posemb.size() == posemb_new.size():
    #             self.transformer.embeddings.position_embeddings.copy_(posemb)
    #         elif posemb.size()[1]-1 == posemb_new.size()[1]:
    #             posemb = posemb[:, 1:]
    #             self.transformer.embeddings.position_embeddings.copy_(posemb)
    #         else:
    #             logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
    #             ntok_new = posemb_new.size(1)
    #             if self.classifier == "seg":
    #                 _, posemb_grid = posemb[:, :1], posemb[0, 1:]
    #             gs_old = int(np.sqrt(len(posemb_grid)))
    #             gs_new = int(np.sqrt(ntok_new))
    #             print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
    #             posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    #             zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    #             posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
    #             posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    #             posemb = posemb_grid
    #             self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

    #         # Encoder whole
    #         for bname, block in self.transformer.encoder.named_children():
    #             for uname, unit in block.named_children():
    #                 unit.load_from(weights, n_block=uname)

    #         if self.transformer.embeddings.hybrid:
    #             self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
    #             gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
    #             gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
    #             self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
    #             self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

    #             for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
    #                 for uname, unit in block.named_children():
    #                     unit.load_from(res_weight, n_block=bname, n_unit=uname)


class My_VisionTransformer_v9(nn.Module):  #v6, use transunet as the smaller segmentation model
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v9, self).__init__()
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
        self.refinement_module = refinement_v7(config_small)
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
        ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        ref_mask = torch.argmax(logits_ref, dim=1).unsqueeze(1)  #bs,1,224,224

        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class My_VisionTransformer_v10(nn.Module):  #v6.1, use transunet as the smaller segmentation model, continuous 
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v10, self).__init__()
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
        self.refinement_module = refinement_v7(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v8(nn.Module):   #smaller  transunet, no skip-connection
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v8, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer2(config, img_size, vis)
        self.decoder = DecoderCup_no_skip(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits





class My_VisionTransformer_v651(nn.Module):  #v6.5.1, use transunet as the smaller segmentation model, continuous, no skip-connection
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v651, self).__init__()
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
        self.refinement_module = refinement_v8(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v9(nn.Module):   #smaller  transunet, no skip-connection and no vit
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v9, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        self.embeddings = Embeddings3(config, img_size)
        self.decoder = DecoderCup_no_skip_no_vit(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        print('x:',x.shape)
        x, features = self.embeddings(x)  
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        # raise Exception
        # print('x:',x.shape)
        raise Exception
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v652(nn.Module):  #v6.5.2, use transunet as the smaller segmentation model, continuous, no skip-connection and no vit
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v652, self).__init__()
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
        self.refinement_module = refinement_v9(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v10(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v10, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        self.embeddings = Embeddings3(config, img_size)
        self.decoder = DecoderCup_no_vit(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x, features = self.embeddings(x)
        
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v653(nn.Module):  #v6.1, use transunet as the smaller segmentation model, continuous, no vit
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v653, self).__init__()
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
        self.refinement_module = refinement_v10(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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






class CSA_Lite_Residual(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(CSA_Lite_Residual, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 将 decoder 的输入 x 投影到 out_channels，方便 residual 相加
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.scale = out_channels ** -0.5

    def forward(self, x, skip):
        """
        x:    (B, C1, H, W) ← decoder feature
        skip: (B, C2, H, W) ← encoder skip feature
        """
        B, _, H, W = x.shape

        # Query 来自 decoder feature
        q = self.query_conv(x)              # (B, C, H, W)
        q = q.flatten(2).transpose(1, 2)    # (B, HW, C)

        # K, V 来自 encoder skip feature，先做全局池化
        k = self.key_conv(skip)             # (B, C, H, W)
        v = self.value_conv(skip)           # (B, C, H, W)
        k = F.adaptive_avg_pool2d(k, 1).squeeze(-1).squeeze(-1)  # (B, C)
        v = F.adaptive_avg_pool2d(v, 1).squeeze(-1).squeeze(-1)  # (B, C)
        k = k.unsqueeze(1)  # (B, 1, C)
        v = v.unsqueeze(1)  # (B, 1, C)

        # Attention: Q @ K^T
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, HW, 1)
        attn = torch.softmax(attn, dim=1)

        # 加权聚合 V
        out = attn * v  # (B, HW, C)
        out = out.transpose(1, 2).view(B, -1, H, W)  # (B, C, H, W)
        out = self.out_conv(out)  # (B, C, H, W)

        # 残差连接
        x_proj = self.res_conv(x)  # (B, C, H, W)
        return out + x_proj



class CSA_Lite_Residual_Decoder(nn.Module):
    def __init__(self):
        super(CSA_Lite_Residual_Decoder, self).__init__()

        # 上采样 + Attention block（包含残差）
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.csa3 = CSA_Lite_Residual(in_channels=1024, skip_channels=512, out_channels=512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.csa2 = CSA_Lite_Residual(in_channels=512, skip_channels=256, out_channels=256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.csa1 = CSA_Lite_Residual(in_channels=256, skip_channels=64, out_channels=64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)  # 输出通道数为2类分割

    def forward(self, f1, f2, f3, f4):
        """
        f4: (B, 1024, 14, 14)
        f3: (B, 512, 28, 28)
        f2: (B, 256, 56, 56)
        f1: (B, 64, 112, 112)
        """
        d4 = f4

        d4_up = self.up4(d4)         # (B, 1024, 28, 28)
        d3 = self.csa3(d4_up, f3)    # → (B, 512, 28, 28)

        d3_up = self.up3(d3)         # (B, 512, 56, 56)
        d2 = self.csa2(d3_up, f2)    # → (B, 256, 56, 56)

        d2_up = self.up2(d2)         # (B, 256, 112, 112)
        d1 = self.csa1(d2_up, f1)    # → (B, 64, 112, 112)

        d1_up = self.up1(d1)         # (B, 64, 224, 224)
        out = self.out_conv(d1_up)   # → (B, 2, 224, 224)

        return out



class refinement_v11(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v11, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        self.embeddings = Embeddings3(config, img_size)
        self.decoder = CSA_Lite_Residual_Decoder()
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x, features = self.embeddings(x)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        
        logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v654(nn.Module):  #v6.1, use transunet as the smaller segmentation model, continuous, no vit
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v654, self).__init__()
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
        self.refinement_module = refinement_v11(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class CMIBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, embed_dim=64):
        super(CMIBlock, self).__init__()
        self.embed_x = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.embed_skip = nn.Conv2d(skip_channels, embed_dim, kernel_size=1)

        self.residual_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.eps = 1e-6

    def forward(self, x, skip):
        """
        x:    decoder feature (B, C1, H, W)
        skip: encoder feature (B, C2, H, W)
        """
        B, _, H, W = x.shape

        # 1. Project embeddings
        x_embed = self.embed_x(x)      # (B, d, H, W)
        skip_embed = self.embed_skip(skip)  # (B, d, H, W)

        # 2. Normalize along channel dimension
        x_norm = F.normalize(x_embed, p=2, dim=1)      # (B, d, H, W)
        skip_norm = F.normalize(skip_embed, p=2, dim=1)  # (B, d, H, W)

        # 3. Compute cosine similarity for each spatial position
        sim_map = torch.sum(x_norm * skip_norm, dim=1, keepdim=True)  # (B, 1, H, W)
        sim_map = torch.relu(sim_map)  # ReLU optional

        # 4. Project skip into residual and inject with matching weight
        skip_proj = self.residual_proj(skip)  # (B, out_C, H, W)
        residual = sim_map * skip_proj        # element-wise enhancement

        # 5. Project x to out_channels and add residual
        x_proj = self.output_proj(x)          # (B, out_C, H, W)
        out = x_proj + residual               # fusion

        return out


class CMI_Decoder(nn.Module):
    def __init__(self):
        super(CMI_Decoder, self).__init__()

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cmi3 = CMIBlock(in_channels=1024, skip_channels=512, out_channels=512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cmi2 = CMIBlock(in_channels=512, skip_channels=256, out_channels=256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cmi1 = CMIBlock(in_channels=256, skip_channels=64, out_channels=64)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, f1, f2, f3, f4):
        """
        f4: (B, 1024, 14, 14)
        f3: (B, 512, 28, 28)
        f2: (B, 256, 56, 56)
        f1: (B, 64, 112, 112)
        """
        d4 = f4  # starting from deepest feature

        d4_up = self.up4(d4)        # (B, 1024, 28, 28)
        d3 = self.cmi3(d4_up, f3)   # → (B, 512, 28, 28)

        d3_up = self.up3(d3)        # (B, 512, 56, 56)
        d2 = self.cmi2(d3_up, f2)   # → (B, 256, 56, 56)

        d2_up = self.up2(d2)        # (B, 256, 112, 112)
        d1 = self.cmi1(d2_up, f1)   # → (B, 64, 112, 112)

        d1_up = self.up1(d1)        # (B, 64, 224, 224)
        out = self.out_conv(d1_up) # → (B, 2, 224, 224)

        return out



class refinement_v12(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v12, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        self.embeddings = Embeddings3(config, img_size)
        self.decoder = CMI_Decoder()
        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x, features = self.embeddings(x)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        
        logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v655(nn.Module):  #v6.1, use transunet as the smaller segmentation model, continuous, no vit
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v655, self).__init__()
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
        self.refinement_module = refinement_v12(config_small)
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
        new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
        final_pred = self.refinement_module(new_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class SimpleFusionDecoder(nn.Module):
    def __init__(self):
        super(SimpleFusionDecoder, self).__init__()

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(3072, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14)
          t3, t2, t1: (B, 512, 28, 28), (B, 256, 56, 56), (B, 64, 112, 112)
          r3, r2, r1: 对应的 reference encoder features
        """
        x = torch.cat([x_tar, x_ref], dim=1)  # (B, 2048, 14, 14)
        x = self.up4(x)  # → (B, 2048, 28, 28)

        x = torch.cat([x, torch.cat([t3, r3], dim=1)], dim=1)  # (B, 2048 + 1024, 28, 28)
        x = self.fuse3(x)  # → (B, 512, 28, 28)

        x = self.up3(x)  # → (B, 512, 56, 56)
        x = torch.cat([x, torch.cat([t2, r2], dim=1)], dim=1)
        x = self.fuse2(x)  # → (B, 256, 56, 56)

        x = self.up2(x)  # → (B, 256, 112, 112)
        x = torch.cat([x, torch.cat([t1, r1], dim=1)], dim=1)
        x = self.fuse1(x)  # → (B, 64, 112, 112)

        x = self.up1(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)
        return out



class refinement_v13(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v13, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = SimpleFusionDecoder()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v656(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v656, self).__init__()
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
        self.refinement_module = refinement_v13(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class DPSEDecoder(nn.Module):
    def __init__(self):
        super(DPSEDecoder, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3 + ref3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 + ref2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 + ref1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v14(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v14, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6511(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6511, self).__init__()
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
        self.refinement_module = refinement_v14(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class RGFFDecoder(nn.Module):
    def __init__(self):
        super(RGFFDecoder, self).__init__()

        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference GT encoder
        self.ref_gt_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)     # -> 224x224
        self.ref_gt_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # -> 112x112
        self.ref_gt_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # -> 56x56

        # gates: 通道投影 + Sigmoid
        self.gate1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.Sigmoid())   # for 112x112
        self.gate2 = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1), nn.Sigmoid())   # for 56x56
        self.gate3 = nn.Sequential(nn.Conv2d(128, 512, 3, padding=1), nn.Sigmoid())  # for 28x28

        # decode path
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref, ref_gt):
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # ref_gt encoding (with float conversion)
        g = ref_gt.float()
        g1 = F.max_pool2d(F.relu(self.ref_gt_conv1(g)), 2)     # (B, 32, 112, 112)
        g2 = F.max_pool2d(F.relu(self.ref_gt_conv2(g1)), 2)    # (B, 64, 56, 56)
        g3 = F.max_pool2d(F.relu(self.ref_gt_conv3(g2)), 2)    # (B, 128, 28, 28)

        # gating
        attn1 = self.gate1(g1)  # (B, 128, 112, 112)
        attn2 = self.gate2(g2)  # (B, 256, 56, 56)
        attn3 = self.gate3(g3)  # (B, 512, 28, 28)

        # decode stage 1
        x = self.up3(x)  # → 28x28
        f3 = t3 + r3
        x = self.dec3(torch.cat([x, f3 * attn3], dim=1))  # (B, 1024, 28, 28)

        # decode stage 2
        x = self.up2(x)  # → 56x56
        f2 = t2 + r2
        x = self.dec2(torch.cat([x, f2 * attn2], dim=1))  # (B, 512, 56, 56)

        # decode stage 3
        x = self.up1(x)  # → 112x112
        # print('x:',x.shape)
        f1 = t1 + r1
        # print('f1:',f1.shape)
        # print('attn1:',attn1.shape)
        x = self.dec1(torch.cat([x, f1 * attn1], dim=1))  # (B, 256, 112, 112)

        x = self.final_up(x)  # → 224x224
        out = self.out_conv(x)
        return out




class refinement_v15(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v15, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.share_encoder = target_encoder(config, img_size)
        # self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = RGFFDecoder()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        reference_image = torch.cat([ref_image, ref_mask], dim=1)  #bs,2,224,224
        x_tar, features_tar = self.share_encoder(target_image)
        x_ref, features_ref = self.share_encoder(reference_image)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        ref_gt = ref_gt.float()
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref,
    ref_gt
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits




class My_VisionTransformer_v657(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v657, self).__init__()
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
        self.refinement_module = refinement_v15(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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














class refinement_v16(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v16, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.share_encoder = target_encoder(config, img_size)
        self.ref_gt_encoder = shared_encoder(config, img_size)

        # Gate modules for GT attention
        self.gate1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Sigmoid())     # 112x112
        self.gate2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())   # 56x56
        self.gate3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.Sigmoid())   # 28x28

        # Decoder
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        target_image = torch.cat([ori_image, ori_mask], dim=1)  # (B,2,224,224)
        reference_image = torch.cat([ref_image, ref_mask], dim=1)  # (B,2,224,224)
        ref_gt = ref_gt.float()
        x_tar, features_tar = self.share_encoder(target_image)   # features: [64@112x112, 256@56x56, 512@28x28]
        x_ref, features_ref = self.share_encoder(reference_image)
        x_gt, features_gt = self.ref_gt_encoder(ref_gt)          # features: [64, 256, 512]

        # Attention from GT features (correct order)
        attn3 = self.gate3(features_gt[0])   # (B,512,28,28)
        attn2 = self.gate2(features_gt[1])   # (B,256,56,56)
        attn1 = self.gate1(features_gt[2])   # (B,64,112,112)

        # Decoder
        x = self.init_fuse(x_tar)           # (B,512,14,14)

        x = self.up3(x)                     # → (B,512,28,28)
        f3 = torch.abs(features_tar[0] - features_ref[0])
        x = self.dec3(torch.cat([x, f3 * attn3], dim=1))  # (B,1024,28,28) → (B,256,28,28)

        x = self.up2(x)                     # → (B,256,56,56)
        f2 = torch.abs(features_tar[1] - features_ref[1])
        x = self.dec2(torch.cat([x, f2 * attn2], dim=1))  # (B,512,56,56) → (B,128,56,56)

        x = self.up1(x)                     # → (B,128,112,112)
        f1 = torch.abs(features_tar[2] - features_ref[2])
        x = self.dec1(torch.cat([x, f1 * attn1], dim=1))  # (B,192,112,112) → (B,64,112,112)

        x = self.final_up(x)                # → (B,64,224,224)
        logits = self.out_conv(x)           # → (B,num_classes,224,224)

        return logits




class My_VisionTransformer_v658(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v658, self).__init__()
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
        self.refinement_module = refinement_v16(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

class refinement_v17(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v17, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.share_encoder = target_encoder(config, img_size)
        self.ref_gt_encoder = shared_encoder(config, img_size)

        # Gate modules for GT attention
        self.gate1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Sigmoid())     # 112x112
        self.gate2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())   # 56x56
        self.gate3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.Sigmoid())   # 28x28

        # Decoder
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        target_image = torch.cat([ori_image, ori_mask], dim=1)  # (B,2,224,224)
        reference_image = torch.cat([torch.abs(ori_image - ref_image), torch.abs(ori_mask - ref_mask)], dim=1)  # (B,2,224,224)
        ref_gt = ref_gt.float()
        x_tar, features_tar = self.share_encoder(target_image)   # features: [64@112x112, 256@56x56, 512@28x28]
        x_ref, features_ref = self.share_encoder(reference_image)
        x_gt, features_gt = self.ref_gt_encoder(ref_gt)          # features: [64, 256, 512]

        # Attention from GT features (correct order)
        attn3 = self.gate3(features_gt[0])   # (B,512,28,28)
        attn2 = self.gate2(features_gt[1])   # (B,256,56,56)
        attn1 = self.gate1(features_gt[2])   # (B,64,112,112)

        # Decoder
        x = self.init_fuse(x_tar)           # (B,512,14,14)

        x = self.up3(x)                     # → (B,512,28,28)
        f3 = features_tar[0] + features_ref[0]
        x = self.dec3(torch.cat([x, f3 * attn3], dim=1))  # (B,1024,28,28) → (B,256,28,28)

        x = self.up2(x)                     # → (B,256,56,56)
        f2 = features_tar[1] + features_ref[1]
        x = self.dec2(torch.cat([x, f2 * attn2], dim=1))  # (B,512,56,56) → (B,128,56,56)

        x = self.up1(x)                     # → (B,128,112,112)
        f1 = features_tar[2] + features_ref[2]
        x = self.dec1(torch.cat([x, f1 * attn1], dim=1))  # (B,192,112,112) → (B,64,112,112)

        x = self.final_up(x)                # → (B,64,224,224)
        logits = self.out_conv(x)           # → (B,num_classes,224,224)

        return logits


class My_VisionTransformer_v659(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v659, self).__init__()
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
        self.refinement_module = refinement_v17(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v18(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v18, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.share_encoder = target_encoder(config, img_size)
        self.ref_gt_encoder = shared_encoder(config, img_size)

        # Gate modules for GT attention
        # self.gate1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Sigmoid())     # 112x112
        # self.gate2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())   # 56x56
        # self.gate3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.Sigmoid())   # 28x28

        # Decoder
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        target_image = torch.cat([ori_image, ori_mask], dim=1)  # (B,2,224,224)
        reference_image = torch.cat([ref_image, ref_mask], dim=1)  # (B,2,224,224)
        ref_gt = ref_gt.float()
        x_tar, features_tar = self.share_encoder(target_image)   # features: [64@112x112, 256@56x56, 512@28x28]
        x_ref, features_ref = self.share_encoder(reference_image)
        x_gt, features_gt = self.ref_gt_encoder(ref_gt)          # features: [64, 256, 512]

        # Attention from GT features (correct order)
        # attn3 = self.gate3(features_gt[0])   # (B,512,28,28)
        # attn2 = self.gate2(features_gt[1])   # (B,256,56,56)
        # attn1 = self.gate1(features_gt[2])   # (B,64,112,112)

        # Decoder
        x = self.init_fuse(x_tar)           # (B,512,14,14)

        x = self.up3(x)                     # → (B,512,28,28)
        f3 = torch.abs(features_ref[0] - features_gt[0])
        x = self.dec3(torch.cat([x, f3 ], dim=1))  # (B,1024,28,28) → (B,256,28,28)

        x = self.up2(x)                     # → (B,256,56,56)
        f2 = torch.abs(features_ref[1] - features_gt[1])
        x = self.dec2(torch.cat([x, f2 ], dim=1))  # (B,512,56,56) → (B,128,56,56)

        x = self.up1(x)                     # → (B,128,112,112)
        f1 = torch.abs(features_ref[2] - features_gt[2])
        x = self.dec1(torch.cat([x, f1 ], dim=1))  # (B,192,112,112) → (B,64,112,112)

        x = self.final_up(x)                # → (B,64,224,224)
        logits = self.out_conv(x)           # → (B,num_classes,224,224)

        return logits


class My_VisionTransformer_v6510(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6510, self).__init__()
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
        self.refinement_module = refinement_v18(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

# class refinement_v18(nn.Module):   #
#     def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
#         super(refinement_v18, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.classifier = config.classifier
#         # self.transformer = Transformer2(config, img_size, vis)
#         # self.embeddings = Embeddings3(config, img_size)
#         self.encoder = shared_encoder(config, img_size)
#         # self.reference_encoder = reference_encoder(config, img_size)
#         # self.decoder = 

#         # self.segmentation_head = SegmentationHead(
#         #     in_channels=config['decoder_channels'][-1],
#         #     out_channels=config['n_classes'],
#         #     kernel_size=3,
#         # )
#         self.config = config

#     def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
#         # if x.size()[1] == 1:
#         #     x = x.repeat(1,5,1,1)
#         # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
#         # print('x:',x.shape)
#         ori_i, ori_i_f = self.encoder(ori_image)
#         ref_i, ref_i_f = self.encoder(ref_image)
#         ori_i_m, ref_i_f = self.encoder(ori_mask)
#         ref_i_m, ref_i_f = self.encoder(ref_mask)
#         ref_gt, ref_gt_f = self.encoder(ref_gt)

#         logits = self.decoder(
#     features_tar[2], features_tar[1], features_tar[0], x_tar,
#     features_ref[2], features_ref[1], features_ref[0], x_ref,
#     ref_gt
# )

#         # logits = self.decoder(features[2], features[1], features[0], x)
#         # logits = self.segmentation_head(x)
#         return logits


class DPSEDecoder_2(nn.Module):
    def __init__(self):
        super(DPSEDecoder_2, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3 - r3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 - r2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 - r1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v19(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v19, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_2()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6513(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6513, self).__init__()
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
        self.refinement_module = refinement_v19(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class DPSEDecoder_3(nn.Module):
    def __init__(self):
        super(DPSEDecoder_3, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3 + ref3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 + ref2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 + ref1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v20(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v20, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_3()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6512(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6512, self).__init__()
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
        self.refinement_module = refinement_v20(config_small)
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
        ref_image = torch.cat([ref_image, ref_mask, torch.abs(ref_gt-ref_mask)], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v21(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v21, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.share_encoder = target_encoder(config, img_size)
        self.ref_gt_encoder = shared_encoder(config, img_size)

        # Gate modules for GT attention
        self.gate1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Sigmoid())     # 112x112
        self.gate2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())   # 56x56
        self.gate3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.Sigmoid())   # 28x28

        # Decoder
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        target_image = torch.cat([ori_image, ori_mask], dim=1)  # (B,2,224,224)
        reference_image = torch.cat([ref_image, ref_mask], dim=1)  # (B,2,224,224)
        ref_gt = ref_gt.float()
        x_tar, features_tar = self.share_encoder(target_image)   # features: [64@112x112, 256@56x56, 512@28x28]
        x_ref, features_ref = self.share_encoder(reference_image)
        x_gt, features_gt = self.ref_gt_encoder(ref_gt)          # features: [64, 256, 512]

        # Attention from GT features (correct order)
        # attn3 = self.gate3(features_gt[0])   # (B,512,28,28)
        # attn2 = self.gate2(features_gt[1])   # (B,256,56,56)
        # attn1 = self.gate1(features_gt[2])   # (B,64,112,112)

        # Decoder
        x = self.init_fuse(x_tar)           # (B,512,14,14)

        x = self.up3(x)                     # → (B,512,28,28)
        f3 = features_ref[0] + features_gt[0]
        x = self.dec3(torch.cat([x, f3 ], dim=1))  # (B,1024,28,28) → (B,256,28,28)

        x = self.up2(x)                     # → (B,256,56,56)
        f2 = features_ref[1] + features_gt[1]
        x = self.dec2(torch.cat([x, f2 ], dim=1))  # (B,512,56,56) → (B,128,56,56)

        x = self.up1(x)                     # → (B,128,112,112)
        f1 = features_ref[2] + features_gt[2]
        x = self.dec1(torch.cat([x, f1 ], dim=1))  # (B,192,112,112) → (B,64,112,112)

        x = self.final_up(x)                # → (B,64,224,224)
        logits = self.out_conv(x)           # → (B,num_classes,224,224)

        return logits


class My_VisionTransformer_v6514(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6514, self).__init__()
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
        self.refinement_module = refinement_v21(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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





class DPSEDecoder_4(nn.Module):
    def __init__(self):
        super(DPSEDecoder_4, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x,  r3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x,  r2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x,  r1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v22(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v22, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_4()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6515(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6515, self).__init__()
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
        self.refinement_module = refinement_v22(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class DPSEDecoder_5(nn.Module):
    def __init__(self):
        super(DPSEDecoder_5, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3, r3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 , r2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 , r1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v23(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v23, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_5()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6516(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6516, self).__init__()
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
        self.refinement_module = refinement_v23(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v24(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v24, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.share_encoder = target_encoder(config, img_size)
        self.ref_gt_encoder = shared_encoder(config, img_size)

        # Gate modules for GT attention
        self.gate1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.Sigmoid())     # 112x112
        self.gate2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Sigmoid())   # 56x56
        self.gate3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.Sigmoid())   # 28x28

        # Decoder
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.config = config

    def forward(self, ori_image, ori_mask, ref_image, ref_mask, ref_gt):
        target_image = torch.cat([ori_image, ori_mask], dim=1)  # (B,2,224,224)
        reference_image = torch.cat([ref_image, ref_mask], dim=1)  # (B,2,224,224)
        ref_gt = ref_gt.float()
        x_tar, features_tar = self.share_encoder(target_image)   # features: [64@112x112, 256@56x56, 512@28x28]
        x_ref, features_ref = self.share_encoder(reference_image)
        x_gt, features_gt = self.ref_gt_encoder(ref_gt)          # features: [64, 256, 512]

        # Attention from GT features (correct order)
        attn3 = self.gate3(features_gt[0])   # (B,512,28,28)
        attn2 = self.gate2(features_gt[1])   # (B,256,56,56)
        attn1 = self.gate1(features_gt[2])   # (B,64,112,112)

        # Decoder
        x = self.init_fuse(x_tar)           # (B,512,14,14)

        x = self.up3(x)                     # → (B,512,28,28)
        f3 = features_tar[0] + features_ref[0]
        x = self.dec3(torch.cat([x, f3 * attn3], dim=1))  # (B,1024,28,28) → (B,256,28,28)

        x = self.up2(x)                     # → (B,256,56,56)
        f2 = features_tar[1] + features_ref[1]
        x = self.dec2(torch.cat([x, f2 * attn2], dim=1))  # (B,512,56,56) → (B,128,56,56)

        x = self.up1(x)                     # → (B,128,112,112)
        f1 = features_tar[2] + features_ref[2]
        x = self.dec1(torch.cat([x, f1 * attn1], dim=1))  # (B,192,112,112) → (B,64,112,112)

        x = self.final_up(x)                # → (B,64,224,224)
        logits = self.out_conv(x)           # → (B,num_classes,224,224)

        return logits




class My_VisionTransformer_v6517(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6517, self).__init__()
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
        self.refinement_module = refinement_v24(config_small)
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
        # target_image = torch.cat([ori_image, ori_mask], dim=1)  #bs,2,224,224
        # ref_image = torch.cat([ref_image, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(ori_image, ori_mask, ref_image, ref_mask, ref_gt)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class DPSEDecoder_6(nn.Module):
    def __init__(self):
        super(DPSEDecoder_6, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1024, 256, kernel_size=3, padding=1)  # 512+512

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(512, 128, kernel_size=3, padding=1)   # 256+256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(192, 64, kernel_size=3, padding=1)    # 128+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3 + r3], dim=1))  # 512 + 512 → 256

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 + r2], dim=1))  # 256 + 256 → 128

        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 + r1], dim=1))  # 64 + 128 → 64

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v25(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v25, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_6()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6518(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6518, self).__init__()
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
        self.refinement_module = refinement_v25(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

class DPSEDecoder_7(nn.Module):
    def __init__(self):
        super(DPSEDecoder_7, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512+512
        self.dec3_2 = Conv2dReLU(256, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)   # 256+256
        self.dec2_2 = Conv2dReLU(128, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)    # 128+64
        self.dec1_2 = Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3, r3], dim=1))  # 512 + 512 → 256
        x = self.dec3_2(x)

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 , r2], dim=1))  # 256 + 256 → 128
        x = self.dec2_2(x)
        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 , r1], dim=1))  # 64 + 128 → 64
        x = self.dec1_2(x)

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v26(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v26, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_7()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6519(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6519, self).__init__()
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
        self.refinement_module = refinement_v26(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class DPSEDecoder_8(nn.Module):
    def __init__(self):
        super(DPSEDecoder_8, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        # self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        # self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        # self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512+512
        self.dec3_2 = Conv2dReLU(256, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)   # 256+256
        self.dec2_2 = Conv2dReLU(128, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)    # 128+64
        self.dec1_2 = Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        # ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3, r3], dim=1))  # 512 + 512 → 256
        x = self.dec3_2(x)

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        # ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 , r2], dim=1))  # 256 + 256 → 128
        x = self.dec2_2(x)
        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        # ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 , r1], dim=1))  # 64 + 128 → 64
        x = self.dec1_2(x)

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v27(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v27, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_8()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6520(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6520, self).__init__()
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
        self.refinement_module = refinement_v27(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class DPSEDecoder_9(nn.Module):
    def __init__(self):
        super(DPSEDecoder_9, self).__init__()

        # 主干起点（仅 target）
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # reference skip 投影层
        self.ref_proj3 = Conv2dReLU(512, 512, kernel_size=1, padding=0)
        self.ref_proj2 = Conv2dReLU(256, 256, kernel_size=1, padding=0)
        self.ref_proj1 = Conv2dReLU(64, 64, kernel_size=1, padding=0)

        # 上采样 + 解码层
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512+512
        self.dec3_2 = Conv2dReLU(256, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)   # 256+256
        self.dec2_2 = Conv2dReLU(128, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)    # 128+64
        self.dec1_2 = Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        """
        Inputs:
          x_tar: (B, 1024, 14, 14) — target 主干输出
          t3, t2, t1: target encoder features (B, C, H, W)
          r3, r2, r1: reference encoder features (B, C, H, W)
        """
        x = self.init_fuse(x_tar)  # (B, 512, 14, 14)

        # decode 1
        x = self.up3(x)  # → (B, 512, 28, 28)
        ref3 = self.ref_proj3(r3)
        x = self.dec3(torch.cat([x, t3, ref3], dim=1))  # 512 + 512 → 256
        x = self.dec3_2(x)

        # decode 2
        x = self.up2(x)  # → (B, 256, 56, 56)
        ref2 = self.ref_proj2(r2)
        x = self.dec2(torch.cat([x, t2 , ref2], dim=1))  # 256 + 256 → 128
        x = self.dec2_2(x)
        # decode 3
        x = self.up1(x)  # → (B, 128, 112, 112)
        ref1 = self.ref_proj1(r1)
        x = self.dec1(torch.cat([x, t1 , ref1], dim=1))  # 64 + 128 → 64
        x = self.dec1_2(x)

        x = self.final_up(x)  # → (B, 64, 224, 224)
        out = self.out_conv(x)  # → (B, 2, 224, 224)

        return out


class refinement_v28(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v28, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DPSEDecoder_9()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6521(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6521, self).__init__()
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
        self.refinement_module = refinement_v28(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


#cost volume
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# --- 基础块 ---
# class Conv2dReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1):
#         super().__init__(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

class Projector(nn.Module):
    """1x1 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = Conv2dReLU(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x):  # (B, Cin, H, W) -> (B, Cout, H, W)
        return self.proj(x)

# --- 局部余弦 cost volume + 加权对齐 ---
class CostVolumeAggregator(nn.Module):
    """
    对 (t, r) 做局部位移窗口内的余弦相似度，softmax 得权重，对 r 做加权求和得到 r_aligned。
    radius=1 => 3x3窗口，radius=2 => 5x5窗口。
    """
    def __init__(self, radius=2, use_softmax=True):
        super().__init__()
        self.radius = radius
        self.use_softmax = use_softmax

    def forward(self, t, r):
        """
        t, r: (B, C, H, W)
        return:
          r_aligned: (B, C, H, W)
          corr_map : (B, K, H, W), K=(2r+1)^2  仅用于可视化/ablation
        """
        B, C, H, W = t.shape
        # 归一化用于余弦相似度
        t_n = F.normalize(t, dim=1, eps=1e-6)
        r_n = F.normalize(r, dim=1, eps=1e-6)

        corr_list = []
        r_shift_vals = []
        shifts = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                r_n_shift = torch.roll(r_n, shifts=(dy, dx), dims=(2, 3))
                r_shift = torch.roll(r,    shifts=(dy, dx), dims=(2, 3))
                # 余弦相似度：通道求和
                sim = (t_n * r_n_shift).sum(dim=1, keepdim=True)  # (B,1,H,W)
                corr_list.append(sim)
                r_shift_vals.append(r_shift)
                shifts.append((dy, dx))

        corr = torch.cat(corr_list, dim=1)  # (B,K,H,W)
        if self.use_softmax:
            weights = F.softmax(corr, dim=1)  # (B,K,H,W)
        else:
            weights = torch.sigmoid(corr)

        # 加权还原 r_aligned
        r_aligned = torch.zeros_like(r)
        for k in range(len(r_shift_vals)):
            r_aligned = r_aligned + r_shift_vals[k] * weights[:, k:k+1, :, :]

        return r_aligned, corr

# --- 融合块（包含残差回退） ---
class FusionBlockCV(nn.Module):
    """
    输入：x_up(上一层上采样输出), t(当前image分支特征), r(当前mask分支特征)
    1) 可选的投影（外部先做或内部做）
    2) CV 对 r 做对齐得到 r_align
    3) 主路径：concat [x_up, t, r_align] -> 3x3 -> 3x3 -> out_ch
    4) 残差路径：concat [x_up, t, r]     -> 1x1 -> out_ch
    输出：out (B, out_ch, H, W)
    """
    def __init__(self, in_x, c_t, c_r, out_ch, radius=2):
        super().__init__()
        self.cv = CostVolumeAggregator(radius=radius, use_softmax=True)

        self.main = nn.Sequential(
            Conv2dReLU(in_x + c_t + c_r, out_ch, kernel_size=3, padding=1),
            Conv2dReLU(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.resid = nn.Conv2d(in_x + c_t + c_r, out_ch, kernel_size=1, padding=0, bias=False)

    def forward(self, x_up, t, r):
        # r 对齐（余弦-based CV）
        r_align, _ = self.cv(t, r)  # (B, c_r, H, W), (B,K,H,W)

        # 主路径：对齐后的 r
        main_in = torch.cat([x_up, t, r_align], dim=1)
        y_main = self.main(main_in)

        # 残差路径：未对齐的 r（作为回退）
        res_in = torch.cat([x_up, t, r], dim=1)
        y_res = self.resid(res_in)

        return y_main + y_res  # residual add

# --- 纯 concat 融合（当某层不做CV时使用） ---
class FusionBlockConcat(nn.Module):
    def __init__(self, in_x, c_t, c_r, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            Conv2dReLU(in_x + c_t + c_r, out_ch, kernel_size=3, padding=1),
            Conv2dReLU(out_ch, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x_up, t, r):
        return self.fuse(torch.cat([x_up, t, r], dim=1))

class DecoderCV_v1_MaskProj_AllScales(nn.Module):
    def __init__(self, radius=2):
        super().__init__()
        # 起点
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 仅 mask 分支投影
        self.proj_r3 = Projector(512, 512)
        self.proj_r2 = Projector(256, 256)
        self.proj_r1 = Projector(64,  64)

        # 上采样 + 融合（全尺度CV）
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = FusionBlockCV(in_x=512, c_t=512, c_r=512, out_ch=256, radius=radius)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = FusionBlockCV(in_x=256, c_t=256, c_r=256, out_ch=128, radius=radius)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = FusionBlockCV(in_x=128, c_t=64,  c_r=64,  out_ch=64,  radius=radius)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)         # (B,512,14,14)

        x = self.up3(x)                   # (B,512,28,28)
        r3p = self.proj_r3(r3)
        x = self.dec3(x, t3, r3p)         # -> (B,256,28,28)

        x = self.up2(x)                   # (B,256,56,56)
        r2p = self.proj_r2(r2)
        x = self.dec2(x, t2, r2p)         # -> (B,128,56,56)

        x = self.up1(x)                   # (B,128,112,112)
        r1p = self.proj_r1(r1)
        x = self.dec1(x, t1, r1p)         # -> (B,64,112,112)

        x = self.final_up(x)              # (B,64,224,224)
        return self.out_conv(x)           # (B,2,224,224)

class DecoderCV_v2_BothProj_AllScales(nn.Module):
    def __init__(self, radius=2):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 双分支投影
        self.proj_t3 = Projector(512, 512)
        self.proj_t2 = Projector(256, 256)
        self.proj_t1 = Projector(64,  64)

        self.proj_r3 = Projector(512, 512)
        self.proj_r2 = Projector(256, 256)
        self.proj_r1 = Projector(64,  64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = FusionBlockCV(in_x=512, c_t=512, c_r=512, out_ch=256, radius=radius)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = FusionBlockCV(in_x=256, c_t=256, c_r=256, out_ch=128, radius=radius)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = FusionBlockCV(in_x=128, c_t=64,  c_r=64,  out_ch=64,  radius=radius)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)  # (B,512,14,14)

        x = self.up3(x)            # (B,512,28,28)
        t3p, r3p = self.proj_t3(t3), self.proj_r3(r3)
        x = self.dec3(x, t3p, r3p) # -> (B,256,28,28)

        x = self.up2(x)            # (B,256,56,56)
        t2p, r2p = self.proj_t2(t2), self.proj_r2(r2)
        x = self.dec2(x, t2p, r2p) # -> (B,128,56,56)

        x = self.up1(x)            # (B,128,112,112)
        t1p, r1p = self.proj_t1(t1), self.proj_r1(r1)
        x = self.dec1(x, t1p, r1p) # -> (B,64,112,112)

        x = self.final_up(x)       # (B,64,224,224)
        return self.out_conv(x)

class DecoderCV_v3_BothProj_HighScales(nn.Module):
    def __init__(self, radius=2):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.proj_t3 = Projector(512, 512)
        self.proj_t2 = Projector(256, 256)
        self.proj_t1 = Projector(64,  64)

        self.proj_r3 = Projector(512, 512)
        self.proj_r2 = Projector(256, 256)
        self.proj_r1 = Projector(64,  64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cv3  = FusionBlockCV(in_x=512, c_t=512, c_r=512, out_ch=256, radius=radius)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cv2  = FusionBlockCV(in_x=256, c_t=256, c_r=256, out_ch=128, radius=radius)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cat1 = FusionBlockConcat(in_x=128, c_t=64, c_r=64, out_ch=64)  # 低尺度不用CV

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)       # (B,512,14,14)

        x = self.up3(x)                 # (B,512,28,28)
        t3p, r3p = self.proj_t3(t3), self.proj_r3(r3)
        x = self.cv3(x, t3p, r3p)       # -> (B,256,28,28)

        x = self.up2(x)                 # (B,256,56,56)
        t2p, r2p = self.proj_t2(t2), self.proj_r2(r2)
        x = self.cv2(x, t2p, r2p)       # -> (B,128,56,56)

        x = self.up1(x)                 # (B,128,112,112)
        t1p, r1p = self.proj_t1(t1), self.proj_r1(r1)
        x = self.cat1(x, t1p, r1p)      # -> (B,64,112,112)

        x = self.final_up(x)            # (B,64,224,224)
        return self.out_conv(x)

class refinement_v29(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v29, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DecoderCV_v1_MaskProj_AllScales()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6522(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6522, self).__init__()
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
        self.refinement_module = refinement_v29(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v30(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v30, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DecoderCV_v2_BothProj_AllScales()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6523(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6523, self).__init__()
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
        self.refinement_module = refinement_v30(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v31(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v31, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DecoderCV_v3_BothProj_AllScales()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6524(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6524, self).__init__()
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
        self.refinement_module = refinement_v28(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ---------- 基础块 ----------
# class Conv2dReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1):
#         super().__init__(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

# ---------- 动态卷积注入（Depthwise，带可选门控/降通道/零初始化） ----------
class DynDepthwiseInject(nn.Module):
    """
    mask特征 r 生成位置自适应的 depthwise k×k 卷积核，作用于 image特征 t；再 residual 回到 t。
    形状:
      t, r: (B, C, H, W) -> 输出: (B, C, H, W)
    """
    def __init__(self, c, k=3, reduce=128, mid=64, use_gate=True, kernel_softmax=False):
        super().__init__()
        assert k in (3, 5), "k建议取3或5"
        self.k = k
        self.use_gate = use_gate
        self.kernel_softmax = kernel_softmax

        # 可选降通道（对t做），降低FLOPs
        self.reduce_in  = nn.Conv2d(c, reduce, 1, bias=False) if reduce and reduce < c else None
        self.expand_out = nn.Conv2d(reduce, c, 1, bias=False) if self.reduce_in is not None else None
        c_eff = reduce if self.reduce_in is not None else c

        # r -> kernel 参数 (每通道一核)
        self.gen = nn.Sequential(
            nn.Conv2d(c, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c_eff * k * k, 1, bias=True)
        )
        # 初始化使初态≈恒等
        nn.init.zeros_(self.gen[-1].weight)
        if self.gen[-1].bias is not None:
            nn.init.zeros_(self.gen[-1].bias)

        # 门控（可选）
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, t, r):
        """
        t, r: (B,C,H,W)
        """
        B, C, H, W = t.shape

        # 降通道
        if self.reduce_in is not None:
            t0 = self.reduce_in(t)        # (B,C',H,W)
            Ceff = t0.size(1)
        else:
            t0 = t
            Ceff = C

        # 生成核 (B, Ceff*k*k, H, W) -> (B,Ceff,k,k,H,W)
        ker = self.gen(r).view(B, Ceff, self.k, self.k, H, W)
        if self.kernel_softmax:
            # 让每个位置的k×k权重做softmax，更稳（可选）
            ker = ker.view(B, Ceff, self.k * self.k, H, W)
            ker = F.softmax(ker, dim=2).view(B, Ceff, self.k, self.k, H, W)

        # unfold t 的局部块: (B, Ceff*k*k, H*W) -> (B,Ceff,k,k,H,W)
        pad = self.k // 2
        t_pad = F.pad(t0, [pad, pad, pad, pad], mode='reflect')
        patches = F.unfold(t_pad, kernel_size=self.k)      # (B, Ceff*k*k, H*W)
        patches = patches.view(B, Ceff, self.k, self.k, H, W)

        # 动态卷积逐点乘加和
        y = (patches * ker).sum(dim=(2, 3))                # (B,Ceff,H,W)

        # 恢复通道
        if self.expand_out is not None:
            y = self.expand_out(y)                         # (B,C,H,W)

        # 门控 + 残差
        if self.use_gate:
            g = self.gate(torch.cat([t, r], dim=1))        # (B,C,H,W)
            y = g * y
        return t + y

# ---------- 解码器 ①：中高层放核（dec3/dec2），无门控 ----------
class DynDecoder_HighOnly_NoGate(nn.Module):
    """
    dec3(28x28)、dec2(56x56) 用动态卷积，无门控；dec1(112x112) 保持普通concat。
    """
    def __init__(self, k=3, reduce=128):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # dec3: DynInject(t3,r3) -> cat([x_up, t3_dyn, r3]) -> 256
        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn3  = DynDepthwiseInject(c=512, k=k, reduce=reduce, use_gate=False)
        self.dec3  = Conv2dReLU(512+512+512, 256, kernel_size=3, padding=1)
        self.dec3_2= Conv2dReLU(256, 256, kernel_size=3, padding=1)

        # dec2
        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn2  = DynDepthwiseInject(c=256, k=k, reduce=reduce, use_gate=False)
        self.dec2  = Conv2dReLU(256+256+256, 128, kernel_size=3, padding=1)
        self.dec2_2= Conv2dReLU(128, 128, kernel_size=3, padding=1)

        # dec1: no dynamic, plain concat
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1  = Conv2dReLU(128+64+64, 64, kernel_size=3, padding=1)
        self.dec1_2= Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)                 # (B,512,14,14)

        x = self.up3(x)                           # (B,512,28,28)
        t3_dyn = self.dyn3(t3, r3)                # (B,512,28,28)
        x = torch.cat([x, t3_dyn, r3], dim=1)     # (B,1536,28,28)
        x = self.dec3(x)                          # (B,256,28,28)
        x = self.dec3_2(x)                        # (B,256,28,28)

        x = self.up2(x)                           # (B,256,56,56)
        t2_dyn = self.dyn2(t2, r2)                # (B,256,56,56)
        x = torch.cat([x, t2_dyn, r2], dim=1)     # (B,768,56,56)
        x = self.dec2(x)                          # (B,128,56,56)
        x = self.dec2_2(x)                        # (B,128,56,56)

        x = self.up1(x)                           # (B,128,112,112)
        x = torch.cat([x, t1, r1], dim=1)         # (B,256,112,112)
        x = self.dec1(x)                          # (B,64,112,112)
        x = self.dec1_2(x)                        # (B,64,112,112)

        x = self.final_up(x)                      # (B,64,224,224)
        return self.out_conv(x)                   # (B,2,224,224)

# ---------- 解码器 ②：所有层放核，无门控 ----------
class DynDecoder_All_NoGate(nn.Module):
    def __init__(self, k=3, reduce=64):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn3  = DynDepthwiseInject(c=512, k=k, reduce=reduce, use_gate=False)
        self.dec3  = Conv2dReLU(512+512+512, 256, kernel_size=3, padding=1)
        self.dec3_2= Conv2dReLU(256, 256, kernel_size=3, padding=1)

        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn2  = DynDepthwiseInject(c=256, k=k, reduce=reduce, use_gate=False)
        self.dec2  = Conv2dReLU(256+256+256, 128, kernel_size=3, padding=1)
        self.dec2_2= Conv2dReLU(128, 128, kernel_size=3, padding=1)

        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn1  = DynDepthwiseInject(c=64,  k=k, reduce=reduce, use_gate=False)
        self.dec1  = Conv2dReLU(128+64+64, 64, kernel_size=3, padding=1)
        self.dec1_2= Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)

        x = self.up3(x)
        t3_dyn = self.dyn3(t3, r3)
        x = torch.cat([x, t3_dyn, r3], dim=1)
        x = self.dec3(x); x = self.dec3_2(x)

        x = self.up2(x)
        t2_dyn = self.dyn2(t2, r2)
        x = torch.cat([x, t2_dyn, r2], dim=1)
        x = self.dec2(x); x = self.dec2_2(x)

        x = self.up1(x)
        t1_dyn = self.dyn1(t1, r1)
        x = torch.cat([x, t1_dyn, r1], dim=1)
        x = self.dec1(x); x = self.dec1_2(x)

        x = self.final_up(x)
        return self.out_conv(x)

# ---------- 解码器 ③：所有层放核，有门控 ----------
class DynDecoder_All_Gated(nn.Module):
    def __init__(self, k=3, reduce=64):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn3  = DynDepthwiseInject(c=512, k=k, reduce=reduce, use_gate=True)
        self.dec3  = Conv2dReLU(512+512+512, 256, kernel_size=3, padding=1)
        self.dec3_2= Conv2dReLU(256, 256, kernel_size=3, padding=1)

        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn2  = DynDepthwiseInject(c=256, k=k, reduce=reduce, use_gate=True)
        self.dec2  = Conv2dReLU(256+256+256, 128, kernel_size=3, padding=1)
        self.dec2_2= Conv2dReLU(128, 128, kernel_size=3, padding=1)

        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dyn1  = DynDepthwiseInject(c=64,  k=k, reduce=reduce, use_gate=True)
        self.dec1  = Conv2dReLU(128+64+64, 64, kernel_size=3, padding=1)
        self.dec1_2= Conv2dReLU(64, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None):
        x = self.init_fuse(x_tar)

        x = self.up3(x)
        t3_dyn = self.dyn3(t3, r3)
        x = torch.cat([x, t3_dyn, r3], dim=1)
        x = self.dec3(x); x = self.dec3_2(x)

        x = self.up2(x)
        t2_dyn = self.dyn2(t2, r2)
        x = torch.cat([x, t2_dyn, r2], dim=1)
        x = self.dec2(x); x = self.dec2_2(x)

        x = self.up1(x)
        t1_dyn = self.dyn1(t1, r1)
        x = torch.cat([x, t1_dyn, r1], dim=1)
        x = self.dec1(x); x = self.dec1_2(x)

        x = self.final_up(x)
        return self.out_conv(x)




class refinement_v32(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v32, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DynDecoder_All_Gated()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6525(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6525, self).__init__()
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
        self.refinement_module = refinement_v32(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v33(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v33, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DynDecoder_All_NoGate()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6526(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6526, self).__init__()
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
        self.refinement_module = refinement_v33(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v34(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v34, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = DynDecoder_All_Gated()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6527(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6527, self).__init__()
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
        self.refinement_module = refinement_v34(config_small)
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
        target_image = torch.cat([ori_image, ref_image], dim=1)  #bs,2,224,224
        ref_image = torch.cat([ori_mask, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

@torch.no_grad()
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





class Decoder_AlignOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 轻量对齐（t与r都对齐到各自通道空间）
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)   # 512+512+512 -> 256

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)    # 256+256+256 -> 128

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)     # 128+64+64 -> 64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        x = self.init_fuse(x_tar)                           # (B,512,14,14)

        x = self.up3(x)                                     # (B,512,28,28)
        t3a = self.t_align3(t3); r3a = self.r_align3(r3)    # (B,512,28,28)
        x = self.dec3(torch.cat([x, t3a, r3a], dim=1))      # -> (B,256,28,28)

        x = self.up2(x)                                     # (B,256,56,56)
        t2a = self.t_align2(t2); r2a = self.r_align2(r2)
        x = self.dec2(torch.cat([x, t2a, r2a], dim=1))      # -> (B,128,56,56)

        x = self.up1(x)                                     # (B,128,112,112)
        t1a = self.t_align1(t1); r1a = self.r_align1(r1)
        x = self.dec1(torch.cat([x, t1a, r1a], dim=1))      # -> (B,64,112,112)

        x = self.final_up(x)                                # (B,64,224,224)
        return self.out_conv(x)



class Decoder_ConfOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        """
        confs: (conf_112, conf_56, conf_28)，每个形状 (B,1,H,W)
        """
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)                 # (B,512,14,14)

        x = self.up3(x)                           # (B,512,28,28)
        r3_use = r3 * conf_28                    # (B,512,28,28)  广播到通道
        x = self.dec3(torch.cat([x, t3, r3_use], dim=1))   # -> (B,256,28,28)

        x = self.up2(x)                           # (B,256,56,56)
        r2_use = r2 * conf_56
        x = self.dec2(torch.cat([x, t2, r2_use], dim=1))   # -> (B,128,56,56)

        x = self.up1(x)                           # (B,128,112,112)
        r1_use = r1 * conf_112
        x = self.dec1(torch.cat([x, t1, r1_use], dim=1))   # -> (B,64,112,112)

        x = self.final_up(x)                      # (B,64,224,224)
        return self.out_conv(x)

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




# ---- 轻量跨尺度注入：把若干邻接尺度对齐到当前尺度，1x1降维→拼接→1x1融合→残差加到当前特征 ----
class CrossScaleFuse(nn.Module):
    def __init__(self, out_c, neighbor_cs, mid=128, use_se=False):
        """
        out_c:     输出通道 = 当前尺度 t/r 的通道数（保持不变，用于残差相加）
        neighbor_cs: list[int]，参与注入的邻接特征的通道列表（每个都会被投影到 mid）
        mid:       每个邻接特征先投影到 mid，再拼接融合到 out_c
        use_se:    是否对融合后的上下文做 SE 门控（由当前尺度特征产生通道门）
        """
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, mid, 1, bias=False) for c in neighbor_cs])
        self.fuse = nn.Conv2d(mid * len(neighbor_cs), out_c, 1, bias=False)
        self.use_se = use_se
        if use_se:
            self.se_fc1 = nn.Conv2d(out_c, out_c // 4, 1)
            self.se_fc2 = nn.Conv2d(out_c // 4, out_c, 1)

    def forward(self, cur_feat, neighbors):
        """
        cur_feat:  当前尺度特征 (B, out_c, H, W) —— 用于生成门控
        neighbors: list[Tensor]，每个已 resize 到 (B, c_i, H, W)
        """
        zs = []
        for x, p in zip(neighbors, self.proj):
            zs.append(p(x))
        if len(zs) == 0:
            return cur_feat
        z = torch.cat(zs, dim=1)          # (B, mid*k, H, W)
        z = self.fuse(z)                  # (B, out_c, H, W)
        if self.use_se:
            g = F.adaptive_avg_pool2d(cur_feat, 1)
            g = F.gelu(self.se_fc1(g))
            g = torch.sigmoid(self.se_fc2(g))
            z = z * g                     # SE 门控
        return cur_feat + z               # 残差注入

# 你已有的对齐模块：LinearAlign, Conv2dReLU 等保持不变
# class LinearAlign(nn.Module): ...
# class Conv2dReLU(nn.Module): ...

class Decoder_AlignPlusConf_crossscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 轻量对齐（同你原来）
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)  # 28
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)  # 56
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)   # 112

        # ---- 新增：跨尺度注入模块（不改 dec* 输入） ----
        # 到 28×28：注入来自 56(256c) 与 112(64c)
        self.cs_t_28 = CrossScaleFuse(out_c=512, neighbor_cs=[256, 64], mid=128)
        self.cs_r_28 = CrossScaleFuse(out_c=512, neighbor_cs=[256, 64], mid=128)

        # 到 56×56：注入来自 28(512c) 与 112(64c)
        self.cs_t_56 = CrossScaleFuse(out_c=256, neighbor_cs=[512, 64], mid=64)
        self.cs_r_56 = CrossScaleFuse(out_c=256, neighbor_cs=[512, 64], mid=64)

        # 到 112×112：注入来自 56(256c)（可选再加 28→112 的 512c）
        self.cs_t_112 = CrossScaleFuse(out_c=64, neighbor_cs=[256], mid=32)
        self.cs_r_112 = CrossScaleFuse(out_c=64, neighbor_cs=[256], mid=32)

        # 逐级解码（保持不变的 in_channels）
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512(x) + 512(t3a) + 512(r3_use)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)   # 256 + 256 + 256

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)    # 128 + 64 + 64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def _rz(self, x, size):  # 统一的 resize
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs

        # 14→28
        x = self.init_fuse(x_tar)             # (B,512,14,14)
        x = self.up3(x)                       # (B,512,28,28)

        # --- 28×28：同尺度 + 跨尺度注入 ---
        t3a = self.t_align3(t3)               # (B,512,28,28)
        r3a = self.r_align3(r3)

        # 跨尺度注入到 t3a / r3a
        # 来自 56, 112 的邻接尺度（参考分支叠加 conf 门控）
        B, _, H28, W28 = t3a.shape
        t2_28 = self._rz(self.t_align2.identity if hasattr(self.t_align2, 'identity') else t2, (H28,W28)) if t2.shape[-1]!=H28 else t2
        t1_28 = self._rz(t1, (H28,W28))
        r2_28 = self._rz(r2 * self._rz(conf_56, (r2.shape[-2], r2.shape[-1])), (H28,W28))
        r1_28 = self._rz(r1 * self._rz(conf_112, (r1.shape[-2], r1.shape[-1])), (H28,W28))

        # 对齐注入（内部有1x1降维与SE门控），残差加回
        t3a = self.cs_t_28(t3a, [t2_28, t1_28])
        r3a = self.cs_r_28(r3a, [r2_28, r1_28])

        # 原有同尺度融合（r3 仍然乘 conf_28 作为主路的同尺度门控）
        r3_use = r3a * self._rz(conf_28, (H28, W28))
        x = self.dec3(torch.cat([x, t3a, r3_use], dim=1))  # -> (B,256,28,28)

        # --- 56×56 ---
        x = self.up2(x)                        # (B,256,56,56)
        t2a = self.t_align2(t2)                # (B,256,56,56)
        r2a = self.r_align2(r2)

        # 跨尺度：来自 28↑、112↓
        H56, W56 = t2a.shape[-2:]
        t3_56 = self._rz(t3, (H56,W56))
        t1_56 = self._rz(t1, (H56,W56))
        r3_56 = self._rz(r3, (H56,W56)) * self._rz(conf_28, (H56,W56))
        r1_56 = self._rz(r1, (H56,W56)) * self._rz(conf_112, (H56,W56))

        t2a = self.cs_t_56(t2a, [t3_56, t1_56])
        r2a = self.cs_r_56(r2a, [r3_56, r1_56])

        r2_use = r2a * self._rz(conf_56, (H56,W56))
        x = self.dec2(torch.cat([x, t2a, r2_use], dim=1))  # -> (B,128,56,56)

        # --- 112×112 ---
        x = self.up1(x)                        # (B,128,112,112)
        t1a = self.t_align1(t1)                # (B,64,112,112)
        r1a = self.r_align1(r1)

        # 跨尺度：来自 56↑（可选再加 28↑）
        H112, W112 = t1a.shape[-2:]
        t2_112 = self._rz(t2, (H112,W112))
        r2_112 = self._rz(r2, (H112,W112)) * self._rz(conf_56, (H112,W112))

        t1a = self.cs_t_112(t1a, [t2_112])
        r1a = self.cs_r_112(r1a, [r2_112])

        r1_use = r1a * self._rz(conf_112, (H112,W112))
        x = self.dec1(torch.cat([x, t1a, r1_use], dim=1))  # -> (B,64,112,112)

        x = self.final_up(x)                    # (B,64,224,224)
        return self.out_conv(x)


class refinement_v35(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v35, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_AlignOnly()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        # print('x:',x.shape) (bs,1024,14,14)
        # print('len:',len(features))
        # print('features:',features[0].shape) (bs,512,28,28)
        # print('features:',features[1].shape) (bs,256,56,56)
        # print('features:',features[2].shape) (bs,64,112,112)
        logits = self.decoder(
    features_tar[2], features_tar[1], features_tar[0], x_tar,
    features_ref[2], features_ref[1], features_ref[0], x_ref
)

        # logits = self.decoder(features[2], features[1], features[0], x)
        # logits = self.segmentation_head(x)
        return logits

class My_VisionTransformer_v6528(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6528, self).__init__()
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
        self.refinement_module = refinement_v35(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v36(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v36, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_ConfOnly()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6529(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6529, self).__init__()
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
        self.refinement_module = refinement_v36(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v37(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v37, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6530(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6530, self).__init__()
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
        self.refinement_module = refinement_v37(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c//r,1), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(c//r,1), c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class Decoder_AlignPlusConf_SEOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 轻量通道对齐
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        # 解码层 + SE
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)
        self.se3  = SE(256)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)
        self.se2  = SE(128)

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)
        self.se1  = SE(64)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs)==3
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)                   # (B,512,14,14)

        x = self.up3(x)                             # (B,512,28,28)
        t3a = self.t_align3(t3)
        r3a = self.r_align3(r3) * conf_28
        x = self.dec3(torch.cat([x, t3a, r3a], 1))  # -> (B,256,28,28)
        x = self.se3(x)

        x = self.up2(x)                             # (B,256,56,56)
        t2a = self.t_align2(t2)
        r2a = self.r_align2(r2) * conf_56
        x = self.dec2(torch.cat([x, t2a, r2a], 1))  # -> (B,128,56,56)
        x = self.se2(x)

        x = self.up1(x)                             # (B,128,112,112)
        t1a = self.t_align1(t1)
        r1a = self.r_align1(r1) * conf_112
        x = self.dec1(torch.cat([x, t1a, r1a], 1))  # -> (B,64,112,112)
        x = self.se1(x)

        x = self.final_up(x)                        # (B,64,224,224)
        return self.out_conv(x)


class Decoder_AlignPlusConf_DeltaProductOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(2560, 256, kernel_size=3, padding=1)  # 512*5

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(1280, 128, kernel_size=3, padding=1)  # 256*5

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(384,  64,  kernel_size=3, padding=1)  # 128+64+64+64+64

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs)==3
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)

        # 28×28
        x = self.up3(x)
        t3a = self.t_align3(t3)
        r3a = self.r_align3(r3) * conf_28
        tr3 = t3a * r3a
        td3 = t3a - r3a
        x = self.dec3(torch.cat([x, t3a, r3a, tr3, td3], 1))

        # 56×56
        x = self.up2(x)
        t2a = self.t_align2(t2)
        r2a = self.r_align2(r2) * conf_56
        tr2 = t2a * r2a
        td2 = t2a - r2a
        x = self.dec2(torch.cat([x, t2a, r2a, tr2, td2], 1))

        # 112×112
        x = self.up1(x)
        t1a = self.t_align1(t1)
        r1a = self.r_align1(r1) * conf_112
        tr1 = t1a * r1a
        td1 = t1a - r1a
        x = self.dec1(torch.cat([x, t1a, r1a, tr1, td1], 1))

        x = self.final_up(x)
        return self.out_conv(x)

class Decoder_AlignPlusConf_DeltaProduct_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(2560, 256, kernel_size=3, padding=1)
        self.se3  = SE(256)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(1280, 128, kernel_size=3, padding=1)
        self.se2  = SE(128)

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(384,  64,  kernel_size=3, padding=1)
        self.se1  = SE(64)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs)==3
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)

        # 28×28
        x = self.up3(x)
        t3a = self.t_align3(t3)
        r3a = self.r_align3(r3) * conf_28
        tr3 = t3a * r3a
        td3 = t3a - r3a
        x = self.dec3(torch.cat([x, t3a, r3a, tr3, td3], 1))
        x = self.se3(x)

        # 56×56
        x = self.up2(x)
        t2a = self.t_align2(t2)
        r2a = self.r_align2(r2) * conf_56
        tr2 = t2a * r2a
        td2 = t2a - r2a
        x = self.dec2(torch.cat([x, t2a, r2a, tr2, td2], 1))
        x = self.se2(x)

        # 112×112
        x = self.up1(x)
        t1a = self.t_align1(t1)
        r1a = self.r_align1(r1) * conf_112
        tr1 = t1a * r1a
        td1 = t1a - r1a
        x = self.dec1(torch.cat([x, t1a, r1a, tr1, td1], 1))
        x = self.se1(x)

        x = self.final_up(x)
        return self.out_conv(x)

class refinement_v38(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v38, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_AlignPlusConf_SEOnly()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6531(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6531, self).__init__()
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
        self.refinement_module = refinement_v38(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v39(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v39, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_AlignPlusConf_DeltaProductOnly()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6532(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6532, self).__init__()
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
        self.refinement_module = refinement_v39(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v40(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v40, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_AlignPlusConf_DeltaProduct_SE()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6533(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6533, self).__init__()
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
        self.refinement_module = refinement_v40(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class Decoder_34(nn.Module):
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
        r3_use = r3a * (1-conf_28)
        x = self.dec3(torch.cat([x, t3a, r3_use], dim=1))# -> (B,256,28,28)

        x = self.up2(x)                                  # (B,256,56,56)
        t2a = self.t_align2(t2); r2a = self.r_align2(r2)
        r2_use = r2a * (1-conf_56)
        x = self.dec2(torch.cat([x, t2a, r2_use], dim=1))# -> (B,128,56,56)

        x = self.up1(x)                                  # (B,128,112,112)
        t1a = self.t_align1(t1); r1a = self.r_align1(r1)
        r1_use = r1a * (1-conf_112)
        x = self.dec1(torch.cat([x, t1a, r1_use], dim=1))# -> (B,64,112,112)

        x = self.final_up(x)                             # (B,64,224,224)
        return self.out_conv(x)


class refinement_v41(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v41, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_34()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6534(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6534, self).__init__()
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
        self.refinement_module = refinement_v41(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class Decoder_35(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # 轻量对齐
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(2048, 256, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(1024, 128, kernel_size=3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(320, 64, kernel_size=3, padding=1)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)                        # (B,512,14,14)

        x = self.up3(x)                                  # (B,512,28,28)
        t3a = self.t_align3(t3); r3a = self.r_align3(r3) # (B,512,28,28)
        r3_use = r3a * (1-conf_28)
        x = self.dec3(torch.cat([x, t3a, r3a, r3_use], dim=1))# -> (B,256,28,28)

        x = self.up2(x)                                  # (B,256,56,56)
        t2a = self.t_align2(t2); r2a = self.r_align2(r2)
        r2_use = r2a * (1-conf_56)
        x = self.dec2(torch.cat([x, t2a, r2a, r2_use], dim=1))# -> (B,128,56,56)

        x = self.up1(x)                                  # (B,128,112,112)
        t1a = self.t_align1(t1); r1a = self.r_align1(r1)
        r1_use = r1a * (1-conf_112)
        x = self.dec1(torch.cat([x, t1a, r1a, r1_use], dim=1))# -> (B,64,112,112)

        x = self.final_up(x)                             # (B,64,224,224)
        return self.out_conv(x)


class refinement_v42(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v42, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder(config, img_size)
        self.reference_encoder = reference_encoder(config, img_size)
        self.decoder = Decoder_35()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6535(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6535, self).__init__()
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
        self.refinement_module = refinement_v42(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class refinement_v43(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v43, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(config, mode="crossse", use_on=("root", "b1", "b2"))
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6536(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6536, self).__init__()
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
        self.refinement_module = refinement_v43(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v44(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v44, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(config, mode="gatedadd", use_on=("root", "b1", "b2"))
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6537(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6537, self).__init__()
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
        self.refinement_module = refinement_v44(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v45(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v45, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(config, mode="deltaproduct", use_on=("root", "b1", "b2"))
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6538(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6538, self).__init__()
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
        self.refinement_module = refinement_v45(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v46(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v46, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(config,
                                      mode="xattn_global",   # 第四个模式
                                      use_on=("b1","b2"),
                                      strides={"b1":4, "b2":2},  # 56->14, 28->14
                                      heads=4, red=2)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6539(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6539, self).__init__()
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
        self.refinement_module = refinement_v46(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v47(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v47, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config, 
    mode="xattn_tokens_conv",
    use_on=("root","b1","b2"),
    heads=4, red=2,
    kv_strides={"root": 8, "b1": 4, "b2": 2},  # root 112->14; b1 56->14; b2 28->14
    kv_kernel=3, kv_padding=1
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6540(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6540, self).__init__()
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
        self.refinement_module = refinement_v47(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v48(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v48, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config,
    mode="xattn_tokens_conv",
    use_on=("root","b1","b2"),
    heads=4, red=2,
    kv_strides={"root": 16, "b1": 8, "b2": 4},  # root 112->14; b1 56->14; b2 28->14
    kv_kernel=3, kv_padding=1
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6541(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6541, self).__init__()
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
        self.refinement_module = refinement_v48(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v49(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v49, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config,
    mode="xattn_tokens_conv",
    use_on=("root","b1","b2"),
    heads=4, red=2,
    kv_strides={"root": 4, "b1": 2, "b2": 1},  # root 112->14; b1 56->14; b2 28->14
    kv_kernel=3, kv_padding=1
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6542(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6542, self).__init__()
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
        self.refinement_module = refinement_v49(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v50(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v50, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config,
    mode="xattn_tokens_conv",
    use_on=("root","b1","b2"),
    heads=4, red=2,
    kv_strides={"root": 2, "b1": 1, "b2": 1},  # root 112->14; b1 56->14; b2 28->14
    kv_kernel=3, kv_padding=1
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6543(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6543, self).__init__()
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
        self.refinement_module = refinement_v50(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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



class refinement_v51(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v51, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config,
    mode="mlp_fuse",
    use_on=("root","b1","b2"),
    heads=4, red=2,
    kv_strides={"root": 2, "b1": 1, "b2": 1},  # root 112->14; b1 56->14; b2 28->14
    kv_kernel=3, kv_padding=1
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6544(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6544, self).__init__()
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
        self.refinement_module = refinement_v51(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v52(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(refinement_v52, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.dual_encoder = DualEncoderWithInteraction(
    config,
    mode="xattn_query",
    use_on=("root","b1","b2"),                  # 先在 b1/b2 开
    heads=4, red=2,
    kv_strides={"root": 2, "b1": 1, "b2": 1},         # 控制 K/V 的 token 数
    interactor_kwargs=dict(
        query_tokens=64,      # ← 原先报错的参数，现在从这里传
        query_share=False,
        query_res_scale=0.1
    )
)

        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

    def forward(self, target, reference):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,5,1,1)
        # x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # print('x:',x.shape)
        # reference 输入 = [ref_image, ref_mask, ref_gt]，shape (B,3,224,224)
        ref_mask = reference[:, 1:2]   # (B,1,224,224)
        ref_gt   = reference[:, 2:3]   # (B,1,224,224)
        # x_tar, features_tar = self.target_encoder(target)
        # x_ref, features_ref = self.reference_encoder(reference)
        x_tar, features_tar, x_ref, features_ref = self.dual_encoder(target, reference)
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

class My_VisionTransformer_v6545(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6545, self).__init__()
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
        self.refinement_module = refinement_v52(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


#start to use pvtv2 as backbone
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

class PVTv2Encoder(nn.Module):
    """与之前版本一致，只是把 input_proj 换成更强一点的 InputAdapter。"""
    def __init__(self, variant='pvt_v2_b2', in_chans=3, keep_rgb_weights=True, pretrained=True):
        super().__init__()
        self.keep_rgb_weights = keep_rgb_weights

        if keep_rgb_weights and in_chans != 3:
            self.input_proj = InputAdapter(in_chans, mid_ch=32, out_ch=3, use_bn=True, use_act=True)
            backbone_in = 3
        else:
            self.input_proj = None
            backbone_in = in_chans

        self.backbone = create_model(
            variant, pretrained=pretrained, features_only=True,
            out_indices=(0,1,2,3), in_chans=backbone_in
        )
        chs = [fi['num_chs'] for fi in self.backbone.feature_info]

        self.to_512_28 = nn.Conv2d(chs[1], 512, kernel_size=1, bias=False)
        self.to_256_56 = nn.Conv2d(chs[0], 256, kernel_size=1, bias=False)
        self.to_064_112 = nn.Conv2d(chs[0],  64, kernel_size=1, bias=False)
        self.to_1024_14 = nn.Conv2d(chs[2], 1024, kernel_size=1, bias=False)

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)  # n→3，学得“伪RGB”，更稳
        s1, s2, s3, s4 = self.backbone(x)  # 56,28,14,7

        f28  = self.to_512_28(s2)
        f56  = self.to_256_56(s1)
        f112 = F.interpolate(s1, scale_factor=2, mode='bilinear', align_corners=False)
        f112 = self.to_064_112(f112)
        x14  = self.to_1024_14(s3)

        return x14, [f28, f56, f112]

# 你的两个 encoder 保持不变，只把 in_chans 设成 2 / 3：
class target_encoder_pvt(nn.Module):
    def __init__(self, config, img_size=224, in_chans=2, variant='pvt_v2_b2', pretrained=True):
        super().__init__()
        self.enc = PVTv2Encoder(variant=variant, in_chans=in_chans, keep_rgb_weights=True, pretrained=pretrained)
    def forward(self, x):
        return self.enc(x)

class reference_encoder_pvt(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, variant='pvt_v2_b2', pretrained=True):
        super().__init__()
        self.enc = PVTv2Encoder(variant=variant, in_chans=in_chans, keep_rgb_weights=False, pretrained=pretrained)
    def forward(self, x):
        return self.enc(x)


class refinement_v53(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(refinement_v53, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_pvt(config, img_size, in_chans=2, pretrained=pretrained)
        self.reference_encoder = reference_encoder_pvt(config, img_size, in_chans=3, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6546(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6546, self).__init__()
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
        self.refinement_module = refinement_v53(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class refinement_v54(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(refinement_v54, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_pvt(config, img_size, in_chans=2, variant='pvt_v2_b1', pretrained=pretrained)
        self.reference_encoder = reference_encoder_pvt(config, img_size, in_chans=3, variant='pvt_v2_b1', pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6547(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6547, self).__init__()
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
        self.refinement_module = refinement_v54(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class refinement_v55(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=False):
        super(refinement_v55, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_pvt(config, img_size, in_chans=2, variant='pvt_v2_b2', pretrained=pretrained)
        self.reference_encoder = reference_encoder_pvt(config, img_size, in_chans=3, variant='pvt_v2_b2', pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6548(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6548, self).__init__()
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
        self.refinement_module = refinement_v55(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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


class refinement_v56(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=False):
        super(refinement_v56, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_pvt(config, img_size, in_chans=2, variant='pvt_v2_b1', pretrained=pretrained)
        self.reference_encoder = reference_encoder_pvt(config, img_size, in_chans=3, variant='pvt_v2_b1', pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6549(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6549, self).__init__()
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
        self.refinement_module = refinement_v56(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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







# 可选：输入适配器（当预训练且 in_chans!=3 时更稳）
# class InputAdapter(nn.Module):
#     def __init__(self, in_ch, mid=32, out_ch=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
#             nn.BatchNorm2d(mid), nn.GELU(),
#             nn.Conv2d(mid, out_ch, 1, bias=True)
#         )
#     def forward(self, x): return self.net(x)

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

# 两个分支封装：target=2通道，reference=3通道
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

# 你原来的 refinement_v37 里，只需把 variant 传 'resnet18' 即可：
# self.target_encoder    = target_encoder(config, img_size, in_chans=2, variant='resnet18', pretrained=True)
# self.reference_encoder = reference_encoder(config, img_size, in_chans=3, variant='resnet18', pretrained=True)

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

class My_VisionTransformer_v6550(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6550, self).__init__()
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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




class refinement_v58(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=False):
        super(refinement_v58, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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

class My_VisionTransformer_v6551(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(My_VisionTransformer_v6551, self).__init__()
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
        self.refinement_module = refinement_v58(config_small)
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
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return final_pred

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





class My_VisionTransformer_v6552(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 2,
            detach_between_iters = True):
        super(My_VisionTransformer_v6552, self).__init__()
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
        return final_pred

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


class My_VisionTransformer_v6553(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 2,
            detach_between_iters = False):
        super(My_VisionTransformer_v6553, self).__init__()
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
        return final_pred

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


class My_VisionTransformer_v6555(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = False):
        super(My_VisionTransformer_v6555, self).__init__()
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
        return final_pred

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

class My_VisionTransformer_v6556(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 4,
            detach_between_iters = True):
        super(My_VisionTransformer_v6556, self).__init__()
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
        return final_pred

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




class My_VisionTransformer_v6557(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 4,
            detach_between_iters = False):
        super(My_VisionTransformer_v6557, self).__init__()
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
        return final_pred

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




class refinement_v58_crossscale(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(refinement_v58_crossscale, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf_crossscale()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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


class My_VisionTransformer_v6558(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6558, self).__init__()
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
        self.refinement_module = refinement_v58_crossscale(config_small)
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
        return final_pred

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



#DCA
from math import pi

# --------- 工具：把 [0,1]^2 归一化坐标映射到 grid_sample 的 [-1,1]^2 ----------
def norm_to_grid(coords_hw):  # coords in [0,1], shape (..., 2) where last is (y, x)
    # grid_sample 需要 (x,y) 顺序，且范围 [-1,1]
    y = coords_hw[..., 0] * 2 - 1
    x = coords_hw[..., 1] * 2 - 1
    return torch.stack([x, y], dim=-1)

# --------- 参考点采样（支持 full 与极简模式） ----------
class RefPointSampler(nn.Module):
    def __init__(self, mode="full", N=128):
        super().__init__()
        self.mode = mode
        self.N = N

    @torch.no_grad()
    def forward(self, ref_gt):  # (B,1,H,W) float/binary in {0,1}
        B, _, H, W = ref_gt.shape
        pts_list = []
        types_list = []
        for b in range(B):
            mask = (ref_gt[b,0] > 0.5).float()
            if mask.sum() < 1:  # 空mask回退
                cy, cx = H/2, W/2
                pts = torch.tensor([[cy/H, cx/W]], device=ref_gt.device).repeat(self.N,1)
                types = torch.zeros(self.N, dtype=torch.long, device=ref_gt.device)
                pts_list.append(pts); types_list.append(types); continue

            # 质心
            ys, xs = torch.meshgrid(torch.arange(H, device=ref_gt.device),
                                    torch.arange(W, device=ref_gt.device), indexing='ij')
            area = mask.sum()
            cy = (ys*mask).sum()/area
            cx = (xs*mask).sum()/area
            base = [(cy/H, cx/W)]; base_types=[0]  # 0=center

            # 主轴极值（用二阶矩近似）
            y0 = (ys - cy) * mask; x0 = (xs - cx) * mask
            Iyy = (y0*y0).sum()/area + 1e-6
            Ixx = (x0*x0).sum()/area + 1e-6
            Ixy = (x0*y0).sum()/area
            # 主方向
            theta = 0.5*torch.atan2(2*Ixy, (Ixx - Iyy) + 1e-6)
            dirs = [theta, theta+pi/2]
            ext = []
            for th in dirs:
                vy, vx = torch.sin(th), torch.cos(th)
                # 正向
                t = torch.linspace(-max(H,W), max(H,W), steps=512, device=ref_gt.device)
                ys_line = (cy + t*vy).round().long().clamp(0,H-1)
                xs_line = (cx + t*vx).round().long().clamp(0,W-1)
                mline = mask[ys_line, xs_line]
                idx_pos = torch.where(mline==1)[0]
                if len(idx_pos)>0:
                    y_pos = ys_line[idx_pos[-1]]/H; x_pos = xs_line[idx_pos[-1]]/W
                    ext.append((y_pos.item(), x_pos.item()))
                # 反向
                idx_neg = torch.where(mline==1)[0]
                if len(idx_neg)>0:
                    y_neg = ys_line[idx_neg[0]]/H; x_neg = xs_line[idx_neg[0]]/W
                    ext.append((y_neg.item(), x_neg.item()))
            for (yy,xx) in ext[:4]:
                base.append((yy,xx)); base_types.append(1)  # 1=extrema

            # 边界均匀点（不跑轮廓库，简化为极坐标采样）
            n_left = max(self.N - len(base), 0)
            if self.mode == "center_ext_uniform":
                K = n_left
            else:
                K = n_left
            if K>0:
                angles = torch.linspace(0, 2*pi, steps=K+1, device=ref_gt.device)[:-1]
                # 从质心向外扫描到边界
                bdry = []
                for ang in angles:
                    vy, vx = torch.sin(ang), torch.cos(ang)
                    t = torch.linspace(0, max(H,W), steps=512, device=ref_gt.device)
                    ys_line = (cy + t*vy).round().long().clamp(0,H-1)
                    xs_line = (cx + t*vx).round().long().clamp(0,W-1)
                    inside = mask[ys_line, xs_line]
                    idx = torch.where(inside==0)[0]
                    if len(idx)==0:
                        yb, xb = ys_line[-1]/H, xs_line[-1]/W
                    else:
                        k = max(idx[0]-1, 0)
                        yb, xb = ys_line[k]/H, xs_line[k]/W
                    bdry.append((yb.item(), xb.item()))
                base += bdry
                base_types += [2]*len(bdry)  # 2=bdry

            pts = torch.tensor(base[:self.N], device=ref_gt.device, dtype=torch.float)
            types = torch.tensor(base_types[:self.N], device=ref_gt.device, dtype=torch.long)
            pts_list.append(pts); types_list.append(types)
        pts = torch.stack(pts_list, dim=0)      # (B,N,2)  [y,x] in [0,1]
        types = torch.stack(types_list, dim=0)  # (B,N)
        return pts, types

# --------- 粗对齐（T0/T1/T2/T3） ----------
class AffineInit(nn.Module):
    def __init__(self, mode="similar"):  # none|shift|similar|affine
        super().__init__()
        self.mode = mode

    @torch.no_grad()
    def forward(self, ref_gt, tgt_pred, pts):
        """
        ref_gt:    (B,1,H,W), 0/1
        tgt_pred:  (B,1,H,W), 0/1 或概率(>0.5视作前景)
        pts:       (B,N,2),   归一化[y,x]∈[0,1]
        返回：     (B,N,2),   归一化[y,x]∈[0,1]，投到 target 的参考点
        """
        if self.mode == "none":  # T0
            return pts

        B, _, H, W = ref_gt.shape
        device = ref_gt.device
        eps = 1e-6

        # --- 二值/概率 → 0/1 掩码 ---
        Mr = (ref_gt  > 0.5).float()   # (B,1,H,W)
        Mt = (tgt_pred> 0.5).float()   # (B,1,H,W)

        # --- 可广播坐标网格（按 batch 独立统计）---
        ys = torch.arange(H, device=device, dtype=ref_gt.dtype).view(1,1,H,1)  # (1,1,H,1)
        xs = torch.arange(W, device=device, dtype=ref_gt.dtype).view(1,1,1,W)  # (1,1,1,W)

        # --- 质心 & 面积（逐样本）---
        Ar = Mr.sum(dim=(2,3), keepdim=True).clamp_min(1.0)             # (B,1,1,1)
        At = Mt.sum(dim=(2,3), keepdim=True).clamp_min(1.0)             # (B,1,1,1)
        cyr = (ys*Mr).sum(dim=(2,3), keepdim=True) / Ar                  # (B,1,1,1)
        cxr = (xs*Mr).sum(dim=(2,3), keepdim=True) / Ar
        cyt = (ys*Mt).sum(dim=(2,3), keepdim=True) / At
        cxt = (xs*Mt).sum(dim=(2,3), keepdim=True) / At

        # 若某样本没有前景，回退到图像中心
        empty_r = (Mr.sum(dim=(2,3), keepdim=True) < 1.5)
        empty_t = (Mt.sum(dim=(2,3), keepdim=True) < 1.5)
        cyr = torch.where(empty_r, torch.full_like(cyr, H/2.0), cyr)
        cxr = torch.where(empty_r, torch.full_like(cxr, W/2.0), cxr)
        cyt = torch.where(empty_t, torch.full_like(cyt, H/2.0), cyt)
        cxt = torch.where(empty_t, torch.full_like(cxt, W/2.0), cxt)

        # 平移量（逐样本）
        dcy = (cyt - cyr)   # (B,1,1,1)
        dcx = (cxt - cxr)

        if self.mode == "shift":  # T1：仅平移
            y = (pts[...,0:1]*H + dcy.squeeze(-1).squeeze(-1))  # (B,N,1)
            x = (pts[...,1:2]*W + dcx.squeeze(-1).squeeze(-1))  # (B,N,1)
            y = y.clamp(0, H-1) / H
            x = x.clamp(0, W-1) / W
            return torch.cat([y, x], dim=-1)

        # --- 二阶矩 & 主轴角（逐样本）---
        # 以各自质心为原点
        y0r = ys - cyr;  x0r = xs - cxr
        y0t = ys - cyt;  x0t = xs - cxt

        Iyy_r = ((y0r*y0r)*Mr).sum(dim=(2,3), keepdim=True)/Ar + eps
        Ixx_r = ((x0r*x0r)*Mr).sum(dim=(2,3), keepdim=True)/Ar + eps
        Ixy_r = ((x0r*y0r)*Mr).sum(dim=(2,3), keepdim=True)/Ar

        Iyy_t = ((y0t*y0t)*Mt).sum(dim=(2,3), keepdim=True)/At + eps
        Ixx_t = ((x0t*x0t)*Mt).sum(dim=(2,3), keepdim=True)/At + eps
        Ixy_t = ((x0t*y0t)*Mt).sum(dim=(2,3), keepdim=True)/At

        theta_r = 0.5*torch.atan2(2*Ixy_r, (Ixx_r - Iyy_r) + eps)  # (B,1,1,1)
        theta_t = 0.5*torch.atan2(2*Ixy_t, (Ixx_t - Iyy_t) + eps)
        dtheta  = (theta_t - theta_r)                               # (B,1,1,1)

        # 缩放（相似：等比，仿射：各向异性）
        s_iso = torch.sqrt((At/Ar).clamp_min(1e-6)).clamp(0.5, 2.0)  # (B,1,1,1)
        if self.mode == "affine":  # T3：各向异性缩放（近似仿射）
            sx = torch.sqrt((Ixx_t/Ixx_r).clamp_min(1e-6)).clamp(0.5, 2.0)  # (B,1,1,1)
            sy = torch.sqrt((Iyy_t/Iyy_r).clamp_min(1e-6)).clamp(0.5, 2.0)
        else:
            sx = sy = s_iso

        # --- 把点从归一化映射到像素坐标，相对 ref 质心做旋转+缩放，再平移到 tgt 质心 ---
        y = pts[...,0]*H  # (B,N)
        x = pts[...,1]*W

        y = y - cyr.view(B,1)      # 相对 ref 质心
        x = x - cxr.view(B,1)

        cos = torch.cos(dtheta).view(B,1)
        sin = torch.sin(dtheta).view(B,1)
        # 先旋转
        y_rot =  cos*y - sin*x
        x_rot =  sin*y + cos*x
        # 再各向异性缩放
        y_scl = sy.view(B,1) * y_rot
        x_scl = sx.view(B,1) * x_rot
        # 最后平移到 target 质心
        y_pix = y_scl + cyt.view(B,1)
        x_pix = x_scl + cxt.view(B,1)

        # 裁剪并归一化回 [0,1]
        y_out = y_pix.clamp(0, H-1) / H
        x_out = x_pix.clamp(0, W-1) / W
        return torch.stack([y_out, x_out], dim=-1)  # (B,N,2)


# --------- Query 构造（E6/E7/B0 切换） ----------
class QueryBuilder(nn.Module):
    def __init__(self, C=256, comp="full"):  # geom_only | geom_plus_refF | full
        super().__init__()
        self.C = C
        self.comp = comp

        # 几何两块：pos(C/4) + type(C/4) = C/2
        self.embed_type = nn.Embedding(8, C//4)
        self.fc_pos = nn.Linear(2, C//4)

        # ref 多尺度外观压缩 + 汇聚到 C
        self.proj_r14  = nn.Conv2d(C, C//2, 1)
        self.proj_r28  = nn.Conv2d(C, C//2, 1)
        self.proj_r56  = nn.Conv2d(C, C//2, 1)
        self.proj_r112 = nn.Conv2d(C, C//2, 1)
        self.fc_refF   = nn.Linear(C*2, C)  # sample_refF 输出是 2C

        # ref_pred 统计
        self.fc_sem = nn.Sequential(nn.Linear(4, C//4), nn.GELU(), nn.Linear(C//4, C//4))

        # 计算 concat 后的总输入维度（关键修复点）
        base = C // 2                                  # pos + type
        add_refF = C if comp in ["geom_plus_refF", "full"] else 0
        add_sem  = C // 4 if comp == "full" else 0
        self.in_dim = base + add_refF + add_sem        # geom_only=128; geom+refF=384; full=448

        # 输出投影用自适应的 in_dim
        self.out = nn.Sequential(nn.Linear(self.in_dim, C), nn.GELU(), nn.Linear(C, C))

        # Sobel 核注册为 buffer（保留你已有的修复）
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

    # 其余 sample_refF / ref_pred_stats / forward 保持你上一版的写法即可


    def sample_refF(self, pyr_r, pts):
        B,N,_ = pts.shape
        outs=[]
        for feat,proj in zip(pyr_r, [self.proj_r112,self.proj_r56,self.proj_r28,self.proj_r14]):
            Bc,C,H,W = feat.shape
            grid = norm_to_grid(pts).view(B,1,N,2)
            samp = F.grid_sample(proj(feat), grid, mode='bilinear', align_corners=True)  # (B,C/2,1,N)
            outs.append(samp.squeeze(2).transpose(1,2)) # (B,N,C/2)
        return torch.cat(outs, dim=-1)  # (B,N,2C)

    def ref_pred_stats(self, ref_pred, pts):
        B,N,_ = pts.shape
        grid = norm_to_grid(pts).view(B,1,N,2)
        p = F.grid_sample(ref_pred, grid, mode='bilinear', align_corners=True).squeeze(2).transpose(1,2)  # (B,N,1)
        # 使用已注册的 Sobel buffer；仅在 dtype 上对齐
        px = F.pad(ref_pred, (1,1,1,1), mode='replicate')
        gx = F.conv2d(px, self.sobel_kx.to(ref_pred.dtype))
        gy = F.conv2d(px, self.sobel_ky.to(ref_pred.dtype))
        g = torch.sqrt(gx*gx + gy*gy + 1e-6)
        g_s = F.grid_sample(g, grid, mode='bilinear', align_corners=True).squeeze(2).transpose(1,2)  # (B,N,1)
        sem = torch.cat([p, 1-p, g_s, 1-(g_s/(g_s.max().clamp_min(1e-6)))], dim=-1)  # (B,N,4)
        return self.fc_sem(sem)  # (B,N,C/4)

    def forward(self, pts, types, pyr_r, ref_pred):
        pos = self.fc_pos(pts)               # (B,N,C/4)
        typ = self.embed_type(types)         # (B,N,C/4)
        parts = [pos, typ]
        if self.comp in ["geom_plus_refF", "full"]:
            refF = self.sample_refF(pyr_r, pts)      # (B,N,2C)
            parts.append(self.fc_refF(refF))         # ★ 使用注册的线性层
        if self.comp == "full":
            parts.append(self.ref_pred_stats(ref_pred, pts))  # (B,N,C/4)
        q = torch.cat(parts, dim=-1)
        q = self.out(q)  # (B,N,C)
        return q


# --------- 轻量 Deformable Cross-Attention：只预测偏移和权重，输出“门控图” ----------
class MSDeformXAttnLite(nn.Module):
    def __init__(self, C=256, L=4, K=4):
        super().__init__()
        self.L, self.K, self.C = L, K, C
        self.fc_off = nn.Linear(C, L*K*2)
        self.fc_w   = nn.Linear(C, L*K)
        # 将门控图投射到各层通道后再做 residual add
        self.to_delta_112 = nn.Conv2d(1,  64, 1)
        self.to_delta_56  = nn.Conv2d(1, 256, 1)
        self.to_delta_28  = nn.Conv2d(1, 512, 1)
        self.to_delta_14  = nn.Conv2d(1,1024, 1)

    def forward(self, Q, ref_pts_tgt, pyr_tgt):
        """
        Q:        (B,N,C)
        ref_pts_tgt: (B,N,2) 参考点已经投到 target 的 [0,1]^2
        pyr_tgt:  list of 4 tensors [P112,P56,P28,P14]  各为 (B,C,H,W)
        返回：4 个 residual：delta112, delta56, delta28, delta14，形状配你的 decoder
        """
        B,N,C = Q.shape
        L, K = self.L, self.K
        off = self.fc_off(Q).view(B,N,L,K,2)     # 归一化偏移（以层分辨率为基准，后面会换算到 grid）
        w   = torch.softmax(self.fc_w(Q).view(B,N,L,K), dim=-1)  # 注意力权重
        # 生成门控热力图：把 (ref_pt + offset) 的 K 个采样点在各自层上画成高斯并累积，权重为 w
        gates = []
        for li, P in enumerate(pyr_tgt):  # P: (B,C,H,W)
            _,_,H,W = P.shape
            # 计算实际采样点坐标（y,x） in [0,1]
            base = ref_pts_tgt.unsqueeze(2).expand(-1, -1, K, -1)       # (B,N,K,2)
            delta = off[:, :, li, :, :]                           # (B,N,K,2)
            # 把偏移理解为相对像素的位移（这里简化：直接在 [0,1] 空间里做小偏移）
            samp = (base + 0.02*delta.tanh()).clamp(0.0, 1.0)                       # 小范围偏移稳定训练
                   

            # 生成高斯热力图并累积
            yy = torch.linspace(0, 1, H, device=P.device).view(1,1,1,H,1)
            xx = torch.linspace(0, 1, W, device=P.device).view(1,1,1,1,W)
            py = samp[...,0].view(B,N,K,1,1)
            px = samp[...,1].view(B,N,K,1,1)
            # sigma 随层次缩放
            sigma = 0.02 * (112.0/float(H))
            g = torch.exp(-((yy - py)**2 + (xx - px)**2) / (2*sigma*sigma))  # (B,N,K,H,W)
            wlk = w[:, :, li, :].view(B,N,K,1,1)
            heat = (g * wlk).sum(dim=2).sum(dim=1, keepdim=True)   # (B,1,H,W)
            gates.append(heat)  # 单通道门控

        # 门控 → 残差（投到对应通道）
        delta112 = self.to_delta_112(gates[0])
        delta56  = self.to_delta_56 (gates[1])
        delta28  = self.to_delta_28 (gates[2])
        delta14  = self.to_delta_14 (gates[3])
        return delta112, delta56, delta28, delta14

# --------- 顶层封装：把采点、仿射、Query、DCA 串起来，并输出4个残差 ----------
class DCAGateInjector(nn.Module):
    def __init__(self, C_dca=256, N=128, K=4, refpoint_mode="full",
                 affine_mode="similar", query_comp="full"):
        super().__init__()
        self.N = N
        self.sampler = RefPointSampler(mode=("center_ext_uniform" if refpoint_mode!="full" else "full"), N=N)
        self.affine  = AffineInit(mode=affine_mode)
        self.qbuild  = QueryBuilder(C=C_dca, comp=query_comp)
        self.proj_t112 = nn.Conv2d( 64, C_dca, 1)
        self.proj_t56  = nn.Conv2d(256, C_dca, 1)
        self.proj_t28  = nn.Conv2d(512, C_dca, 1)
        self.proj_t14  = nn.Conv2d(1024,C_dca, 1)
        self.proj_r112 = nn.Conv2d( 64, C_dca, 1)
        self.proj_r56  = nn.Conv2d(256, C_dca, 1)
        self.proj_r28  = nn.Conv2d(512, C_dca, 1)
        self.proj_r14  = nn.Conv2d(1024,C_dca, 1)
        self.msxattn   = MSDeformXAttnLite(C=C_dca, L=4, K=K)

    def forward(self, target, reference, feats_t, feats_r):
        """
        target:    (B, Ct, H, W) 假定第2通道是 target prediction
        reference: (B, 3, H, W)  [ref_image, ref_pred, ref_gt]
        feats_t:   (x_t, [f28_t, f56_t, f112_t])
        feats_r:   (x_r, [f28_r, f56_r, f112_r])
        """
        B, _, H, W = target.shape
        tgt_pred = target[:,1:2]  # (B,1,H,W)
        ref_pred = reference[:,1:2]
        ref_gt   = reference[:,2:3]

        x_t, (f28_t,f56_t,f112_t) = feats_t
        x_r, (f28_r,f56_r,f112_r) = feats_r

        # 1) 采点（在 ref_gt），并投到 target（仿射）
        pts_ref, types = self.sampler(ref_gt)                 # (B,N,2)
        pts_tgt = self.affine(ref_gt, tgt_pred, pts_ref)      # (B,N,2)

        # 2) 构建 reference 金字塔给 QueryBuilder 取外观（与 decoder 无关，不改变原通道）
        pyr_r = [ self.proj_r112(f112_r), self.proj_r56(f56_r),
                  self.proj_r28(f28_r),  self.proj_r14(x_r) ]  # →C_dca

        Q = self.qbuild(pts_ref, types, pyr_r, ref_pred)       # (B,N,C_dca)

        # 3) target 侧金字塔（给 DCA 使用）
        pyr_t = [ self.proj_t112(f112_t), self.proj_t56(f56_t),
                  self.proj_t28(f28_t),  self.proj_t14(x_t)  ] # →C_dca

        # 4) 轻量 deformable x-attn → 4 个尺度的残差（匹配 decoder 的通道）
        delta112, delta56, delta28, delta14 = self.msxattn(Q, pts_tgt, pyr_t)
        return delta112, delta56, delta28, delta14


class refinement_v59(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True,
                 # ★ 新增：DCA 配置
                 dca_mode="B0",      # "B0" | "E1" | "E4" | "E6" | "E7"
                 e4_affine="similar" # 仅 E4 用："none"|"shift"|"similar"|"affine"
                 ):
        super(refinement_v59, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.target_encoder = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()
        self.config = config

        # ★ 新增：一条 DCA 旁路（默认 B0）
        if dca_mode == "B0":
            self.dca = DCAGateInjector(C_dca=256, N=128, K=4,
                                       refpoint_mode="full",
                                       affine_mode="similar",
                                       query_comp="full")
        elif dca_mode == "E1":
            self.dca = DCAGateInjector(C_dca=256, N=32, K=4,
                                       refpoint_mode="center_ext_uniform",
                                       affine_mode="similar",
                                       query_comp="full")
        elif dca_mode == "E4":
            self.dca = DCAGateInjector(C_dca=256, N=128, K=4,
                                       refpoint_mode="full",
                                       affine_mode=e4_affine,  # ← T0/T1/T2/T3
                                       query_comp="full")
        elif dca_mode == "E6":
            self.dca = DCAGateInjector(C_dca=256, N=128, K=4,
                                       refpoint_mode="full",
                                       affine_mode="similar",
                                       query_comp="geom_only")
        elif dca_mode == "E7":
            self.dca = DCAGateInjector(C_dca=256, N=128, K=4,
                                       refpoint_mode="full",
                                       affine_mode="similar",
                                       query_comp="geom_plus_refF")
        else:
            raise ValueError(f"Unknown dca_mode: {dca_mode}")

    def forward(self, target, reference):
        # reference = [ref_image, ref_mask(pred), ref_gt]，shape (B,3,H,W)
        ref_mask = reference[:, 1:2]
        ref_gt   = reference[:, 2:3]

        x_tar, features_tar = self.target_encoder(target)       # x_tar: (B,1024,14,14)
        x_ref, features_ref = self.reference_encoder(reference) # same shapes

        # ★ NEW：拿到 DCA 产生的四个残差（直接配齐 decoder 的4个输入）
        d112, d56, d28, d14 = self.dca(
            target, reference,
            feats_t=(x_tar, features_tar),   # (x, [f28,f56,f112])
            feats_r=(x_ref, features_ref)
        )

        # ★ NEW：对 decoder 的四个 target 端输入做最小残差注入（add）
        f112_t = features_tar[2] + d112   # (B, 64,112,112)
        f56_t  = features_tar[1] + d56    # (B,256, 56, 56)
        f28_t  = features_tar[0] + d28    # (B,512, 28, 28)
        x_tar_ = x_tar + d14              # (B,1024,14,14)

        # 参考端保持不变（也可以日后在参考端做轻注入，这里先不动，确保可比性）
        f112_r, f56_r, f28_r = features_ref[2], features_ref[1], features_ref[0]

        # 你原来的置信度金字塔
        conf_112, conf_56, conf_28 = compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True)

        logits = self.decoder(
            f112_t, f56_t, f28_t, x_tar_,
            f112_r, f56_r, f28_r, x_ref,
            confs=(conf_112, conf_56, conf_28)
        )
        return logits



class My_VisionTransformer_v6559(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6559, self).__init__()
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
        self.refinement_module = refinement_v59(config_small, pretrained=True, dca_mode="B0")
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
        return final_pred

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


class My_VisionTransformer_v6560(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6560, self).__init__()
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
        self.refinement_module = refinement_v59(config_small, pretrained=True, dca_mode="E1")
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
        return final_pred

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




class My_VisionTransformer_v6561(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6561, self).__init__()
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
        self.refinement_module = refinement_v59(config_small, pretrained=True, dca_mode="E4", e4_affine="similar")
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
        return final_pred

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


class My_VisionTransformer_v6562(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6562, self).__init__()
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
        self.refinement_module = refinement_v59(config_small, pretrained=True, dca_mode="E6")
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
        return final_pred

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

class My_VisionTransformer_v6563(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6563, self).__init__()
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
        self.refinement_module = refinement_v59(config_small, pretrained=True, dca_mode="E7")
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
        return final_pred

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






# =========================
# Helpers
# =========================

def norm_to_grid(coords_hw):  # (..,2) with (y,x) in [0,1] -> (..,2) (x,y) in [-1,1] for grid_sample
    y = coords_hw[..., 0] * 2 - 1
    x = coords_hw[..., 1] * 2 - 1
    return torch.stack([x, y], dim=-1)

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=32, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

# =========================
# 参考点采样 & 粗对齐（per-sample）
# =========================

class RefPointSampler(nn.Module):
    """
    从 reference GT (B,1,H,W) 采 N 个点（y,x ∈ [0,1]），并返回类型标签。
    mode="full"：中心+主轴极值+等角度边界；"center_ext_uniform"：轻量版。
    """
    def __init__(self, mode="full", N=48):
        super().__init__()
        self.mode = mode
        self.N = N

    @torch.no_grad()
    def forward(self, ref_gt):  # (B,1,H,W) in {0,1}
        B, _, H, W = ref_gt.shape
        device = ref_gt.device
        pts_list, types_list = [], []
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        for b in range(B):
            mask = (ref_gt[b,0] > 0.5).float()
            if mask.sum() < 1:
                cy, cx = H/2, W/2
                pts = torch.tensor([[cy/H, cx/W]], device=device).repeat(self.N,1)
                types = torch.zeros(self.N, dtype=torch.long, device=device)
                pts_list.append(pts); types_list.append(types); continue

            area = mask.sum()
            cy = (ys*mask).sum()/area
            cx = (xs*mask).sum()/area
            base = [(cy/H, cx/W)]; base_types=[0]  # 0=center

            # 主轴极值
            y0 = (ys - cy) * mask
            x0 = (xs - cx) * mask
            Iyy = (y0*y0).sum()/area + 1e-6
            Ixx = (x0*x0).sum()/area + 1e-6
            Ixy = (x0*y0).sum()/area
            theta = 0.5*torch.atan2(2*Ixy, (Ixx - Iyy) + 1e-6)
            dirs = [theta, theta+pi/2]
            for th in dirs:
                vy, vx = torch.sin(th), torch.cos(th)
                t = torch.linspace(-max(H,W), max(H,W), steps=512, device=device)
                ys_line = (cy + t*vy).round().long().clamp(0,H-1)
                xs_line = (cx + t*vx).round().long().clamp(0,W-1)
                mline = mask[ys_line, xs_line]
                idx = torch.where(mline==1)[0]
                if len(idx)>0:
                    y1 = ys_line[idx[0]]/H; x1 = xs_line[idx[0]]/W
                    y2 = ys_line[idx[-1]]/H; x2 = xs_line[idx[-1]]/W
                    base += [(y1,x1),(y2,x2)]; base_types += [1,1]  # 1=extrema

            # 边界均匀射线
            K = max(self.N - len(base), 0)
            if K>0:
                angles = torch.linspace(0, 2*pi, steps=K+1, device=device)[:-1]
                for ang in angles:
                    vy, vx = torch.sin(ang), torch.cos(ang)
                    t = torch.linspace(0, max(H,W), steps=512, device=device)
                    ys_line = (cy + t*vy).round().long().clamp(0,H-1)
                    xs_line = (cx + t*vx).round().long().clamp(0,W-1)
                    inside = mask[ys_line, xs_line]
                    idx0 = torch.where(inside==0)[0]
                    if len(idx0)==0:
                        yb, xb = ys_line[-1]/H, xs_line[-1]/W
                    else:
                        k = max(idx0[0]-1, 0)
                        yb, xb = ys_line[k]/H, xs_line[k]/W
                    base.append((yb,xb)); base_types.append(2)  # 2=boundary
            pts = torch.tensor(base[:self.N], device=device, dtype=torch.float32)
            types = torch.tensor(base_types[:self.N], device=device, dtype=torch.long)
            pts_list.append(pts); types_list.append(types)
        return torch.stack(pts_list,0), torch.stack(types_list,0)  # (B,N,2), (B,N)

class AffineInit(nn.Module):
    """ per-sample 相似/仿射先验：none|shift|similar|affine """
    def __init__(self, mode="similar"):
        super().__init__()
        self.mode = mode

    @torch.no_grad()
    def forward(self, ref_gt, tgt_pred, pts):  # (B,1,H,W),(B,1,H,W),(B,N,2)
        if self.mode == "none":
            return pts
        B,_,H,W = ref_gt.shape
        device = ref_gt.device
        eps = 1e-6
        Mr = (ref_gt>0.5).float()
        Mt = (tgt_pred>0.5).float()
        ys = torch.arange(H, device=device, dtype=ref_gt.dtype).view(1,1,H,1)
        xs = torch.arange(W, device=device, dtype=ref_gt.dtype).view(1,1,1,W)
        Ar = Mr.sum(dim=(2,3), keepdim=True).clamp_min(1.0)
        At = Mt.sum(dim=(2,3), keepdim=True).clamp_min(1.0)
        cyr = (ys*Mr).sum(dim=(2,3), keepdim=True)/Ar
        cxr = (xs*Mr).sum(dim=(2,3), keepdim=True)/Ar
        cyt = (ys*Mt).sum(dim=(2,3), keepdim=True)/At
        cxt = (xs*Mt).sum(dim=(2,3), keepdim=True)/At
        empty_r = (Mr.sum(dim=(2,3), keepdim=True) < 1.5)
        empty_t = (Mt.sum(dim=(2,3), keepdim=True) < 1.5)
        cyr = torch.where(empty_r, torch.full_like(cyr, H/2.0), cyr)
        cxr = torch.where(empty_r, torch.full_like(cxr, W/2.0), cxr)
        cyt = torch.where(empty_t, torch.full_like(cyt, H/2.0), cyt)
        cxt = torch.where(empty_t, torch.full_like(cxt, W/2.0), cxt)
        dcy, dcx = (cyt - cyr), (cxt - cxr)
        if self.mode == "shift":
            y = (pts[...,0:1]*H + dcy.view(B,1,1))
            x = (pts[...,1:2]*W + dcx.view(B,1,1))
            return torch.cat([(y.clamp(0,H-1)/H), (x.clamp(0,W-1)/W)], dim=-1)

        # 二阶矩
        def _cov(M, cy, cx):
            y0 = torch.arange(H, device=device, dtype=M.dtype).view(1,1,H,1) - cy
            x0 = torch.arange(W, device=device, dtype=M.dtype).view(1,1,1,W) - cx
            Iyy = ((y0*y0)*M).sum(dim=(2,3), keepdim=True)/M.sum(dim=(2,3), keepdim=True).clamp_min(1.0) + eps
            Ixx = ((x0*x0)*M).sum(dim=(2,3), keepdim=True)/M.sum(dim=(2,3), keepdim=True).clamp_min(1.0) + eps
            Ixy = ((x0*y0)*M).sum(dim=(2,3), keepdim=True)/M.sum(dim=(2,3), keepdim=True).clamp_min(1.0)
            theta = 0.5*torch.atan2(2*Ixy, (Ixx - Iyy) + eps)
            return theta, Ixx, Iyy
        theta_r, Ixx_r, Iyy_r = _cov(Mr, cyr, cxr)
        theta_t, Ixx_t, Iyy_t = _cov(Mt, cyt, cxt)
        dtheta = (theta_t - theta_r)
        s_iso = torch.sqrt((At/Ar).clamp_min(1e-6)).clamp(0.5, 2.0)
        if self.mode == "affine":
            sx = torch.sqrt((Ixx_t/Ixx_r).clamp_min(1e-6)).clamp(0.5, 2.0)
            sy = torch.sqrt((Iyy_t/Iyy_r).clamp_min(1e-6)).clamp(0.5, 2.0)
        else:
            sx = sy = s_iso

        y = pts[...,0]*H - cyr.view(B,1)
        x = pts[...,1]*W - cxr.view(B,1)
        cos = torch.cos(dtheta).view(B,1); sin = torch.sin(dtheta).view(B,1)
        y_rot =  cos*y - sin*x
        x_rot =  sin*y + cos*x
        y_scl = sy.view(B,1)*y_rot; x_scl = sx.view(B,1)*x_rot
        y_pix = y_scl + cyt.view(B,1)
        x_pix = x_scl + cxt.view(B,1)
        y_out = y_pix.clamp(0,H-1)/H
        x_out = x_pix.clamp(0,W-1)/W
        return torch.stack([y_out, x_out], dim=-1)

# =========================
# Query 构造（E6/E7/B0 兼容）
# =========================

class QueryBuilder(nn.Module):
    """
    comp: "geom_only" | "geom_plus_refF" | "full"
    C=256 默认；提供 ref 外观采样与(可选) ref 语义统计
    """
    def __init__(self, C=256, comp="geom_plus_refF"):
        super().__init__()
        self.C = C
        self.comp = comp
        self.embed_type = nn.Embedding(8, C//4)
        self.fc_pos     = nn.Linear(2, C//4)

        # ref 多尺度外观压缩（各层先 1x1 到 C 再由外层传入）
        self.fc_refF    = nn.Linear(C*2, C)  # 拼4层后先 →2C（每层C/2），再线性回到C

        # ref_pred 统计（仅 "full" 时用）
        self.fc_sem = nn.Sequential(nn.Linear(4, C//4), nn.GELU(), nn.Linear(C//4, C//4))

        # 输出投影（根据 comp 自适应 in_dim）
        base = C//4 + C//4                 # pos + type = C/2
        add_refF = C if comp in ["geom_plus_refF","full"] else 0
        add_sem  = C//4 if comp=="full" else 0
        self.in_dim = base + add_refF + add_sem
        self.out = nn.Sequential(nn.Linear(self.in_dim, C), nn.GELU(), nn.Linear(C, C))

        # Sobel 核注册为 buffer（仅 "full" 用）
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

        # 各层外观预投影（把 C → C/2，便于拼接成 2C）
        self.proj_r112 = nn.Conv2d(C, C//2, 1)
        self.proj_r56  = nn.Conv2d(C, C//2, 1)
        self.proj_r28  = nn.Conv2d(C, C//2, 1)
        self.proj_r14  = nn.Conv2d(C, C//2, 1)

    def sample_refF(self, pyr_r, pts):
        B,N,_ = pts.shape
        outs=[]
        for feat,proj in zip(pyr_r, [self.proj_r112,self.proj_r56,self.proj_r28,self.proj_r14]):
            Bc,C,H,W = feat.shape
            grid = norm_to_grid(pts).view(B,1,N,2)
            samp = F.grid_sample(proj(feat), grid, mode='bilinear', align_corners=True)  # (B,C/2,1,N)
            outs.append(samp.squeeze(2).transpose(1,2)) # (B,N,C/2)
        Fcat = torch.cat(outs, dim=-1)  # (B,N,2C)
        return self.fc_refF(Fcat)       # (B,N,C)

    def ref_pred_stats(self, ref_pred, pts):
        B,N,_ = pts.shape
        grid = norm_to_grid(pts).view(B,1,N,2)
        p = F.grid_sample(ref_pred, grid, mode='bilinear', align_corners=True).squeeze(2).transpose(1,2)  # (B,N,1)
        px = F.pad(ref_pred, (1,1,1,1), mode='replicate').to(ref_pred.dtype)
        gx = F.conv2d(px, self.sobel_kx.to(ref_pred.dtype))
        gy = F.conv2d(px, self.sobel_ky.to(ref_pred.dtype))
        g  = torch.sqrt(gx*gx + gy*gy + 1e-6)
        g_s= F.grid_sample(g, grid, mode='bilinear', align_corners=True).squeeze(2).transpose(1,2)       # (B,N,1)
        sem= torch.cat([p, 1-p, g_s, 1-(g_s/(g_s.max().clamp_min(1e-6)))], dim=-1)  # (B,N,4)
        return self.fc_sem(sem)  # (B,N,C/4)

    def forward(self, pts_ref, types, pyr_r, ref_pred=None):
        pos = self.fc_pos(pts_ref)           # (B,N,C/4)
        typ = self.embed_type(types)         # (B,N,C/4)
        parts = [pos, typ]
        if self.comp in ["geom_plus_refF","full"]:
            parts.append(self.sample_refF(pyr_r, pts_ref))  # (B,N,C)
        if self.comp == "full" and ref_pred is not None:
            parts.append(self.ref_pred_stats(ref_pred, pts_ref))    # (B,N,C/4)
        q = torch.cat(parts, dim=-1)         # (B,N,in_dim)
        return self.out(q)                   # (B,N,C)

# =========================
# DCA 核心（输出 Z & E_l）
# =========================

class MSDeformXAttnCore(nn.Module):
    """
    简化版多尺度可形变 cross-attn：
    输入 Q(B,N,C), 参考点 pts_tgt(B,N,2), target pyramid [B,C,H,W]*4
    输出：Z(B,N,C) + E_l(B,C,H_l,W_l) 四层空间引导。
    """
    def __init__(self, C=256, L=4, K=4, r0=0.02):
        super().__init__()
        self.C, self.L, self.K, self.r0 = C, L, K, r0
        self.fc_off = nn.Linear(C, L*K*2)
        self.fc_w   = nn.Linear(C, L*K)
        # 层权投影（融合前给每层一个线性变换）
        self.proj_layer = nn.ModuleList([nn.Linear(C, C) for _ in range(L)])
        # 门控映射：把单通道热力投到 C
        self.toE = nn.ModuleList([nn.Conv2d(1, C, 1) for _ in range(L)])

    def forward(self, Q, pts_tgt, pyr_t):
        """
        Q: (B,N,C), pts_tgt: (B,N,2 in [0,1]), pyr_t: [P112,P56,P28,P14]
        return: Z(B,N,C), [E112,E56,E28,E14]
        """
        B,N,C = Q.shape
        L,K = self.L, self.K
        offs = self.fc_off(Q).view(B,N,L,K,2)             # (B,N,L,K,2)
        alps = torch.softmax(self.fc_w(Q).view(B,N,L,K), dim=-1)  # (B,N,L,K)

        Z_parts = []
        E_list = []
        for li, P in enumerate(pyr_t):  # li: 0..L-1
            _, Cc, H, W = P.shape
            # 采样坐标： (B,N,K,2) = pts + small offset
            base = pts_tgt.unsqueeze(2).expand(-1, -1, K, -1)         # (B,N,K,2)
            delta= torch.tanh(offs[:,:,li,:,:]) * self.r0             # (B,N,K,2)
            samp = (base + delta).clamp(0.0, 1.0)                     # (B,N,K,2) (y,x)

            # grid_sample: (x,y) in [-1,1]
            grid = norm_to_grid(samp).view(B, 1, N*K, 2)              # (B,1,NK,2)
            vals = F.grid_sample(P, grid, mode='bilinear', align_corners=True) # (B,C,1,NK)
            vals = vals.view(B, Cc, 1, N, K).permute(0,3,4,1,2).squeeze(-1)    # (B,N,K,C)

            # K 聚合
            a = alps[:,:,li,:].unsqueeze(-1)                          # (B,N,K,1)
            v = (vals * a).sum(dim=2)                                 # (B,N,C)

            # 层投影并收集
            v_proj = self.proj_layer[li](v)                           # (B,N,C)
            Z_parts.append(v_proj)

            # 栅格化热力（单通道）：对 K 与 N 的采样点加权 splat 成 H×W
            yy = torch.linspace(0, 1, H, device=P.device).view(1,1,1,H,1)
            xx = torch.linspace(0, 1, W, device=P.device).view(1,1,1,1,W)
            py = samp[...,0].view(B,N,K,1,1)
            px = samp[...,1].view(B,N,K,1,1)
            # 半径随层缩放（与 112 层对齐）
            sigma = 0.02 * (112.0/float(H))
            g = torch.exp(-((yy - py)**2 + (xx - px)**2) / (2*sigma*sigma))  # (B,N,K,H,W)
            wlk = a.view(B,N,K,1,1)  # (B,N,K,1,1)
            heat = (g * wlk).sum(dim=2).sum(dim=1, keepdim=True)             # (B,1,H,W)
            E_list.append(self.toE[li](heat))                                 # (B,C,H,W)

        Z = torch.stack(Z_parts, dim=0).sum(dim=0)  # (B,N,C) 跨层求和
        return Z, E_list  # [E112,E56,E28,E14]

# =========================
# DCA 封装：采点 -> 粗对齐 -> Query -> DCA
# =========================

class DCAExtractor(nn.Module):
    """
    上游封装：输入 target/reference 原图与其金字塔，输出
    Z(B,N,C) 与 E_l（四层）。
    """
    def __init__(self, C_dca=256, N=48, K=4, refpoint_mode="full",
                 affine_mode="similar", query_comp="geom_plus_refF", r0=0.02):
        super().__init__()
        self.N = N
        self.sampler = RefPointSampler(mode=("center_ext_uniform" if refpoint_mode!="full" else "full"), N=N)
        self.affine  = AffineInit(mode=affine_mode)
        self.qbuild  = QueryBuilder(C=C_dca, comp=query_comp)
        self.core    = MSDeformXAttnCore(C=C_dca, L=4, K=K, r0=r0)

        # 把编码器输出对齐到 C_dca
        self.proj_t112 = nn.Conv2d( 64, C_dca, 1)
        self.proj_t56  = nn.Conv2d(256, C_dca, 1)
        self.proj_t28  = nn.Conv2d(512, C_dca, 1)
        self.proj_t14  = nn.Conv2d(1024,C_dca, 1)
        self.proj_r112 = nn.Conv2d( 64, C_dca, 1)
        self.proj_r56  = nn.Conv2d(256, C_dca, 1)
        self.proj_r28  = nn.Conv2d(512, C_dca, 1)
        self.proj_r14  = nn.Conv2d(1024,C_dca, 1)

    def forward(self, target, reference, feats_t, feats_r):
        """
        target:    (B,2,H,W) = [I_t, P_t]
        reference: (B,3,H,W) = [I_r, P_r, G_r]
        feats_t:   (x_t, [f28_t,f56_t,f112_t])
        feats_r:   (x_r, [f28_r,f56_r,f112_r])
        """
        B, _, H, W = target.shape
        tgt_pred = target[:,1:2]
        ref_pred = reference[:,1:2]
        ref_gt   = reference[:,2:3]

        x_t, (f28_t,f56_t,f112_t) = feats_t
        x_r, (f28_r,f56_r,f112_r) = feats_r

        # 1) 参考点（在 ref GT）并粗对齐到 target
        pts_ref, types = self.sampler(ref_gt)                # (B,N,2), (B,N)
        pts_tgt = self.affine(ref_gt, tgt_pred, pts_ref)     # (B,N,2)

        # 2) 参考外观金字塔（只用于构造 Query）
        pyr_r = [ self.proj_r112(f112_r), self.proj_r56(f56_r),
                  self.proj_r28(f28_r),  self.proj_r14(x_r) ]  # (B,C,*,*)

        Q = self.qbuild(pts_ref, types, pyr_r, ref_pred if self.qbuild.comp=="full" else None)  # (B,N,C)

        # 3) target 侧金字塔（DCA K/V）
        pyr_t = [ self.proj_t112(f112_t), self.proj_t56(f56_t),
                  self.proj_t28(f28_t),  self.proj_t14(x_t)  ] # (B,C,*,*)

        # 4) DCA → Z 与 E_l
        Z, E_list = self.core(Q, pts_tgt, pyr_t)             # Z:(B,N,C); E_list len=4
        # 顺序与分辨率：E_list = [E112,E56,E28,E14]
        return Z, E_list, pyr_t

# =========================
# 轻量 Mask Head（FPN-lite + 门控 + 语义偏置）
# =========================

class MaskHeadFPNLite(nn.Module):
    """
    输入：pyr_t=[P112,P56,P28,P14] (均为 C 通道)；E_list 对应四层的空间引导；Z 语义 token
    输出：logits (B,num_classes,224,224)
    """
    def __init__(self, C=256, num_classes=2, img_size=224, gn_groups=32):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.img_size = img_size

        # 门控：E_l -> 1 通道 gate；语义偏置：z_ctx -> C 通道偏置
        self.g112 = nn.Conv2d(C, 1, 1)
        self.g56  = nn.Conv2d(C, 1, 1)
        self.g28  = nn.Conv2d(C, 1, 1)
        self.g14  = nn.Conv2d(C, 1, 1)

        self.b112 = nn.Linear(C, C)
        self.b56  = nn.Linear(C, C)
        self.b28  = nn.Linear(C, C)
        self.b14  = nn.Linear(C, C)

        # 每层一个细化 conv
        self.refine14  = ConvGNAct(C, C, 3, groups=gn_groups)
        self.refine28  = ConvGNAct(C, C, 3, groups=gn_groups)
        self.refine56  = ConvGNAct(C, C, 3, groups=gn_groups)
        self.refine112 = ConvGNAct(C, C, 3, groups=gn_groups)

        # 融合 conv（FPN 自顶向下）
        self.fuse28 = ConvGNAct(C, C, 3, groups=gn_groups)
        self.fuse56 = ConvGNAct(C, C, 3, groups=gn_groups)
        self.fuse112= ConvGNAct(C, C, 3, groups=gn_groups)

        # 头部
        self.head = nn.Sequential(
            ConvGNAct(C, C, 3, groups=gn_groups),
            nn.Conv2d(C, num_classes, 1)
        )

        # 可学习强度（防止一开始门控过猛），初值 0
        self.alpha_g112 = nn.Parameter(torch.zeros(1))
        self.alpha_g56  = nn.Parameter(torch.zeros(1))
        self.alpha_g28  = nn.Parameter(torch.zeros(1))
        self.alpha_g14  = nn.Parameter(torch.zeros(1))
        self.alpha_b    = nn.Parameter(torch.zeros(1))

    def _modulate(self, P, E, z_ctx, g_layer, b_layer, alpha_g):
        """
        P: (B,C,H,W), E:(B,C,H,W), z_ctx:(B,C)
        """
        gate = torch.sigmoid(g_layer(E))                   # (B,1,H,W)
        bias = b_layer(z_ctx).view(z_ctx.size(0), -1, 1, 1)  # (B,C,1,1)
        return P * (1.0 + alpha_g * gate) + self.alpha_b * bias

    def forward(self, pyr_t, E_list, Z):
        """
        pyr_t = [P112,P56,P28,P14] (B,C,H,W) ； E_list 同顺序；Z(B,N,C)
        """
        P112, P56, P28, P14 = pyr_t
        E112, E56, E28, E14 = E_list
        # 语义池化（也可用注意力池化，这里用均值）
        z_ctx = Z.mean(dim=1)  # (B,C)

        # 层内调制
        M14  = self._modulate(P14,  E14,  z_ctx, self.g14,  self.b14,  self.alpha_g14)
        M28  = self._modulate(P28,  E28,  z_ctx, self.g28,  self.b28,  self.alpha_g28)
        M56  = self._modulate(P56,  E56,  z_ctx, self.g56,  self.b56,  self.alpha_g56)
        M112 = self._modulate(P112, E112, z_ctx, self.g112, self.b112, self.alpha_g112)

        # 细化
        Y14  = self.refine14(M14)
        # FPN 自顶向下融合
        Y28  = self.fuse28( M28  + F.interpolate(Y14,  size=M28.shape[-2:], mode='bilinear', align_corners=True) )
        Y56  = self.fuse56( M56  + F.interpolate(Y28,  size=M56.shape[-2:], mode='bilinear', align_corners=True) )
        Y112 = self.fuse112(M112 + F.interpolate(Y56,  size=M112.shape[-2:], mode='bilinear', align_corners=True) )
        Y112 = self.refine112(Y112)

        logits = self.head(Y112)  # (B,num_classes,112,112)
        if logits.shape[-1] != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        return logits  # (B,num_classes,224,224)
        

# =========================
# 主网络：refinement_v60（无旧 decoder）
# =========================

class refinement_v60(nn.Module):
    """
    新版：抛弃旧 decoder。管线：
    编码器 -> DCAExtractor(Z & E_l) -> MaskHeadFPNLite -> logits
    """
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True,
                 dca_mode="E6",      # "B0"|"E1"|"E4"|"E6"|"E7"
                 e4_affine="similar",# 仅 E4
                 C_dca=256,
                 N_points=48,        # 小病灶默认更小的 N
                 K_samples=4,
                 r0=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = getattr(config, "classifier", None)

        # 编码器（沿用你的封装）
        self.target_encoder    = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)

        # DCA 配置映射
        if dca_mode == "B0":
            refpoint_mode="full";     query_comp="full";             affine_mode="similar"
        elif dca_mode == "E1":
            refpoint_mode="center_ext_uniform"; query_comp="full";   affine_mode="similar"
        elif dca_mode == "E4":
            refpoint_mode="full";     query_comp="full";             affine_mode=e4_affine
        elif dca_mode == "E6":
            refpoint_mode="full";     query_comp="geom_only";        affine_mode="similar"
        elif dca_mode == "E7":
            refpoint_mode="full";     query_comp="geom_plus_refF";   affine_mode="similar"
        else:
            raise ValueError(f"Unknown dca_mode: {dca_mode}")

        # 采点数量（E1 可适当更小）
        N_cfg = N_points if dca_mode!="E1" else min(32, N_points)

        # DCA 封装（输出 Z 与 E_l）
        self.dca = DCAExtractor(C_dca=C_dca, N=N_cfg, K=K_samples,
                                refpoint_mode=refpoint_mode,
                                affine_mode=affine_mode,
                                query_comp=query_comp,
                                r0=r0)

        # 新的 Mask Head
        self.mask_head = MaskHeadFPNLite(C=C_dca, num_classes=num_classes, img_size=img_size)

    def forward(self, target, reference):
        """
        target:    [I_t, P_t] (B,2,H,W)
        reference: [I_r, P_r, G_r] (B,3,H,W)
        """
        # 编码
        x_t, feats_t = self.target_encoder(target)         # x_t:(B,1024,14,14); feats_t=[f28,f56,f112]
        x_r, feats_r = self.reference_encoder(reference)

        # DCA：得到 Z 与各层空间引导 E_l，同时返回对齐后的 target 金字塔（C_dca 通道）
        Z, E_list, pyr_t = self.dca(target, reference, feats_t=(x_t, feats_t), feats_r=(x_r, feats_r))
        # E_list 顺序：[E112, E56, E28, E14]；pyr_t 顺序：[P112,P56,P28,P14]

        # FPN-lite Head
        logits = self.mask_head(pyr_t, E_list, Z)          # (B,num_classes,224,224)
        return logits



class My_VisionTransformer_v6564(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6564, self).__init__()
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
        self.refinement_module = refinement_v60(config_small, pretrained=True, dca_mode="E6")
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
        return final_pred

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



class My_VisionTransformer_v6565(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6565, self).__init__()
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
        self.refinement_module = refinement_v60(config_small, pretrained=True, dca_mode="B0")
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
        return final_pred

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



class My_VisionTransformer_v6566(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6566, self).__init__()
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
        self.refinement_module = refinement_v60(config_small, pretrained=True, dca_mode="E1")
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
        return final_pred

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


class My_VisionTransformer_v6567(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6567, self).__init__()
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
        self.refinement_module = refinement_v60(config_small, pretrained=True, dca_mode="E4")
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
        return final_pred

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

REPO_DIR = './dinov3'
WEIGHTS_PATH = './dinov3/dinov3/pretrained_ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
# dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='./dinov3/dinov3/pretrained_ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)







# 假定你已有：
# from your_module import InputAdapter, IMAGENET_MEAN, IMAGENET_STD

class DINOv3MidPyramidEncoder(nn.Module):
    def __init__(self, repo_dir, weights_path,
                 in_chans=3,
                 freeze_backbone=True,
                 layers_idx=(2, 5, 8, 11),
                 adapter_kwargs=None):
        super().__init__()
        # 1) 输入适配（可调 mid_ch/BN/激活）
        if adapter_kwargs is None:
            adapter_kwargs = {}
        self.input_adapter = InputAdapter(in_ch=in_chans, **adapter_kwargs)

        # 2) 加载本地 DINOv3 ViT-S/16
        self.backbone = torch.hub.load(
            repo_dir, 'dinov3_vits16', source='local', weights=weights_path
        )
        self.embed_dim = getattr(self.backbone, 'embed_dim', 384)  # ViT-S/16 缺省 384
        # 统一为升序且 0-based
        self.layers_idx = tuple(sorted(layers_idx))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # 3) 归一化参数
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1)
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std',  std,  persistent=False)

        # 4) 各层投影到目标通道
        C = self.embed_dim
        self.proj_deep_14 = nn.Conv2d(C, 1024, 1, bias=False)  # 最深层 → 1024@14
        self.proj_28      = nn.Conv2d(C,  512, 1, bias=False)  # 中间层 → 512@28
        self.proj_56      = nn.Conv2d(C,  256, 1, bias=False)  # 更浅层 → 256@56
        self.proj_112     = nn.Conv2d(C,   64, 1, bias=False)  # 最浅层 → 64@112

        self.bn_28  = nn.BatchNorm2d(512)
        self.bn_56  = nn.BatchNorm2d(256)
        self.bn_112 = nn.BatchNorm2d(64)

    def _preprocess(self, x):
        x = self.input_adapter(x)  # in_ch → 3
        # 尽量在相同 dtype/device 上做归一化
        x = (x - self.mean.to(dtype=x.dtype, device=x.device)) / self.std.to(dtype=x.dtype, device=x.device)
        return x

    def _tokens_to_map(self, tokens: torch.Tensor):
        """
        tokens: (B, N, C) → fmap (B, C, H, W)
        优先从 backbone.patch_embed.grid_size 读出网格，退化时用 sqrt(N)
        """
        B, N, C = tokens.shape
        H = W = None
        pe = getattr(self.backbone, 'patch_embed', None)
        if pe is not None and hasattr(pe, 'grid_size') and pe.grid_size is not None:
            # 有的实现 grid_size 是 int/tuple/tensor，统一拿前两个
            gs = pe.grid_size
            if isinstance(gs, (tuple, list)):
                H, W = int(gs[0]), int(gs[1])
            elif torch.is_tensor(gs):
                H = int(gs.flatten()[0].item())
                W = int(gs.flatten()[1].item() if gs.numel() > 1 else H)
            else:
                H = W = int(gs)
        if H is None or W is None:
            H = W = int(N ** 0.5)  # 回退
        assert H * W == N, f"Token 数 N={N} 无法重排为 HxW={H}x{W}"
        return tokens.transpose(1, 2).contiguous().view(B, C, H, W)

    def _normalize_out_tensor(self, o):
        """将可能的 (tokens, ...) / [tokens, ...] 规范为纯 tokens Tensor。"""
        if isinstance(o, (tuple, list)):
            o = o[0]
        return o

    def _infer_num_layers(self):
        """尽量稳健地估计 Transformer block 总层数 L。"""
        L = getattr(self.backbone, 'num_layers', None)
        if L is None:
            blocks = getattr(self.backbone, 'blocks', None)
            if blocks is not None:
                L = len(blocks)
        if L is None:
            # 保底：假定 12（ViT-S/16 常见），但建议尽量通过 blocks 获取
            L = 12
        return int(L)

    def _get_intermediate_tokens(self, x3):
        """
        优先使用 backbone 的 get_intermediate_layers；
        若仅支持 n= 语义，则用 [min_idx..L-1] 覆盖再映射回 self.layers_idx。
        返回：list[(B, N, C)]，顺序严格与 self.layers_idx 对齐（浅→深）。
        """
        if hasattr(self.backbone, 'get_intermediate_layers'):
            # 1) 尝试直接使用 layers=
            try:
                
                outs = self.backbone.get_intermediate_layers(
                    x3, layers=self.layers_idx, return_class_token=False, reshape=False
                )
                outs = [self._normalize_out_tensor(o) for o in outs]
                return outs
            except TypeError:
                # 2) 只支持 n= 的实现
                # print('not support layer index')
                L = self._infer_num_layers()
                # print('num layers:',L)
                layers_idx = list(self.layers_idx)  # 升序
                assert 0 <= layers_idx[0] < L and layers_idx[-1] < L, \
                    f"layers_idx {layers_idx} 超出层数范围 0..{L-1}"
                min_idx = layers_idx[0]
                n = L - min_idx  # 覆盖 [min_idx, ..., L-1]
                outs = self.backbone.get_intermediate_layers(
                    x3, n=n, return_class_token=False, reshape=False
                )
                outs = [self._normalize_out_tensor(o) for o in outs]
                # 绝对层 i 对应相对索引 (i - min_idx)
                sel = [outs[i - min_idx] for i in layers_idx]
                return sel

        # 3) 兜底：没有 get_intermediate_layers，则只取最终 tokens 并重复填充
        feats = self.backbone.forward_features(x3)
        if isinstance(feats, dict):
            if 'x_norm_patchtokens' in feats:
                pt = feats['x_norm_patchtokens']   # (B, N, C)
            elif 'x_norm' in feats:                 # (B,1+N,C) → 去 cls
                pt = feats['x_norm'][:, 1:, :]
            else:
                pt = self.backbone(x3)              # 某些实现 forward 直接返回 tokens
        else:
            pt = feats
        return [pt] * len(self.layers_idx)

    def forward(self, x):
        """
        输入: (B, in_chans, 224, 224)
        输出: x14 (B,1024,14,14), features=[f28,f56,f112]（与 decoder 顺序一致）
        """
        x3 = self._preprocess(x)

        # 若冻结则无梯度；若未冻结且在训练模式则保留梯度
        need_grad = self.training and any(p.requires_grad for p in self.backbone.parameters())
        with torch.set_grad_enabled(need_grad):
            toks = self._get_intermediate_tokens(x3)   # list of (B, N, C)

        # 严格保证返回数量与索引一致，并按浅→深解包
        assert len(toks) == len(self.layers_idx), f"期望 {len(self.layers_idx)} 层，实际 {len(toks)}"
        t_shallow, t_mid1, t_mid2, t_deep = toks  # 浅→深

        f_shallow = self._tokens_to_map(t_shallow)   # (B,C,14,14)
        f_mid1    = self._tokens_to_map(t_mid1)
        f_mid2    = self._tokens_to_map(t_mid2)
        f_deep    = self._tokens_to_map(t_deep)

        # 14×14 → 1024（瓶颈）
        x14 = self.proj_deep_14(f_deep)              # (B,1024,14,14)

        # 28：由较深的中间层上采样 ×2
        f28 = self.proj_28(f_mid2)                   # (B,512,14,14)
        f28 = F.interpolate(f28, scale_factor=2, mode='bilinear', align_corners=True)  # 14→28
        f28 = self.bn_28(f28).relu_()

        # 56：由较浅中间层上采样 ×4
        f56 = self.proj_56(f_mid1)                   # (B,256,14,14)
        f56 = F.interpolate(f56, scale_factor=4, mode='bilinear', align_corners=True)  # 14→56
        f56 = self.bn_56(f56).relu_()

        # 112：由最浅中间层上采样 ×8
        f112 = self.proj_112(f_shallow)              # (B,64,14,14)
        f112 = F.interpolate(f112, scale_factor=8, mode='bilinear', align_corners=True)  # 14→112
        f112 = self.bn_112(f112).relu_()

        features = [f28, f56, f112]                  # 顺序与 decoder 一致
        return x14, features


        # ……（下方投影/上采样等保持不变）

#Dinov3 as encoder
class refinement_v61(nn.Module):   #
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False, pretrained=True, freeze_backbone=True):
        super(refinement_v61, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = Transformer2(config, img_size, vis)
        # self.embeddings = Embeddings3(config, img_size)
        self.target_encoder = DINOv3MidPyramidEncoder(
            repo_dir=REPO_DIR, weights_path=WEIGHTS_PATH,
            in_chans=2, freeze_backbone=freeze_backbone, layers_idx=(2,5,8,11),
            adapter_kwargs=dict(mid_ch=16, use_bn=True, use_act=True),
        )
        self.reference_encoder = DINOv3MidPyramidEncoder(
            repo_dir=REPO_DIR, weights_path=WEIGHTS_PATH,
            in_chans=3, freeze_backbone=freeze_backbone, layers_idx=(2,5,8,11),
            adapter_kwargs=dict(mid_ch=16, use_bn=True, use_act=True),  
)

        # self.target_encoder = target_encoder_dinov3(config, img_size, pretrained=pretrained)
        # self.reference_encoder = reference_encoder_dinov3(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        # self.segmentation_head = SegmentationHead(
        #     in_channels=config['decoder_channels'][-1],
        #     out_channels=config['n_classes'],
        #     kernel_size=3,
        # )
        self.config = config

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


class My_VisionTransformer_v6568(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6568, self).__init__()
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
        self.refinement_module = refinement_v61(config_small, freeze_backbone=True)
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





class My_VisionTransformer_v6569(nn.Module):  
    def __init__(self, config, config_small, img_size=224, num_classes=21843, zero_head=False, vis=False, refine_iters = 3,
            detach_between_iters = True):
        super(My_VisionTransformer_v6569, self).__init__()
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
        self.refinement_module = refinement_v61(config_small, freeze_backbone=False)
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


