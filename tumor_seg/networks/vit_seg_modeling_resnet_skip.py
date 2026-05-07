import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]



class ResNetV3(nn.Module): #change input channel to 5
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(5, width, kernel_size=7, stride=2, bias=False, padding=3)),   #csp change input channel to 5
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class ResNetV4(nn.Module): #change input channel to 2
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(2, width, kernel_size=7, stride=2, bias=False, padding=3)),   #csp change input channel to 5
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]



class ResNetV5(nn.Module): #change input channel to 1
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),   #csp change input channel to 5
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]






class CrossSE(nn.Module):
    def __init__(self, c, r=16, gain=1.0, learnable_gain=True):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain), requires_grad=learnable_gain)
        red = max(c // r, 1)
        self.se_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(red, c, 1), nn.Sigmoid()
        )
        self.se_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(red, c, 1), nn.Sigmoid()
        )
    def forward(self, t, r):
        a_r = self.se_t(r)
        a_t = self.se_r(t)
        t_new = t + self.gain * (t * a_r - t)
        r_new = r + self.gain * (r * a_t - r)
        return t_new, r_new


class GatedAdd(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj_r = nn.Conv2d(c, c, 1, bias=False)
        self.proj_t = nn.Conv2d(c, c, 1, bias=False)
        self.gate_t = nn.Sequential(nn.Conv2d(2*c, c, 1, bias=False), nn.Sigmoid())
        self.gate_r = nn.Sequential(nn.Conv2d(2*c, c, 1, bias=False), nn.Sigmoid())
    def forward(self, t, r):
        gt = self.gate_t(torch.cat([t, r], dim=1))
        gr = self.gate_r(torch.cat([r, t], dim=1))
        t_new = t + gt * self.proj_r(r)
        r_new = r + gr * self.proj_t(t)
        return t_new, r_new


class DeltaProduct(nn.Module):
    def __init__(self, c, reduce=True):
        super().__init__()
        in_ch = 4 * c  # [t, r, t-r, t*r]
        mid = c if not reduce else max(c // 2, 1)
        self.fuse_t = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False)
        )
        self.fuse_r = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False)
        )
    def forward(self, t, r):
        tr = t * r
        td = t - r
        t_new = t + self.fuse_t(torch.cat([t, r, td, tr], dim=1))
        r_new = r + self.fuse_r(torch.cat([r, t, -td, tr], dim=1))
        return t_new, r_new


# =========================
# 新增：Global Tokens Cross-Attention（第四个 mode）
# =========================

class XAttnGlobalTokens(nn.Module):
    """
    轻量跨分支交互：Q 为自身全分辨率；K/V 为对方池化得到的 tokens。
    双向：t<-r 和 r<-t；输出形状保持 (B,C,H,W) 不变。
    """
    def __init__(self, c, heads=4, red=2, pool_stride=4):
        super().__init__()
        self.h = heads
        self.s = pool_stride
        c_red = max(c // red, 1)

        # t <- r
        self.q_t = nn.Conv2d(c, c_red, 1, bias=False)
        self.k_r = nn.Conv2d(c, c_red, 1, bias=False)
        self.v_r = nn.Conv2d(c, c_red, 1, bias=False)
        self.o_t = nn.Conv2d(c_red, c, 1, bias=False)

        # r <- t
        self.q_r = nn.Conv2d(c, c_red, 1, bias=False)
        self.k_t = nn.Conv2d(c, c_red, 1, bias=False)
        self.v_t = nn.Conv2d(c, c_red, 1, bias=False)
        self.o_r = nn.Conv2d(c_red, c, 1, bias=False)

    def _pool_tokens(self, y):
        return F.avg_pool2d(y, kernel_size=self.s, stride=self.s, ceil_mode=True)

    def _attn_once(self, q, k, v):
        B, Cr, H, W = q.shape
        Hp, Wp = k.shape[-2], k.shape[-1]
        h, d = self.h, Cr // self.h

        Q = q.reshape(B, h, d, H*W).permute(0,1,3,2)         # (B,h,Nq,d)
        K = k.reshape(B, h, d, Hp*Wp).permute(0,1,3,2)       # (B,h,Nk,d)
        V = v.reshape(B, h, d, Hp*Wp).permute(0,1,3,2)       # (B,h,Nk,d)

        A = (Q @ K.transpose(-2, -1)) / (d ** 0.5)           # (B,h,Nq,Nk)
        A = A.softmax(dim=-1)
        O = A @ V                                            # (B,h,Nq,d)
        O = O.permute(0,1,3,2).reshape(B, Cr, H, W)          # (B,Cr,H,W)
        return O

    def forward(self, t, r):
        # t <- r
        q_t = self.q_t(t)
        k_r = self.k_r(self._pool_tokens(r))
        v_r = self.v_r(self._pool_tokens(r))
        dt  = self._attn_once(q_t, k_r, v_r)
        t_new = t + self.o_t(dt)

        # r <- t
        q_r = self.q_r(r)
        k_t = self.k_t(self._pool_tokens(t))
        v_t = self.v_t(self._pool_tokens(t))
        dr  = self._attn_once(q_r, k_t, v_t)
        r_new = r + self.o_r(dr)

        return t_new, r_new



class XAttnTokensConv(nn.Module):
    """
    Cross-Attention with learnable tokenization for K/V:
      - Q: self, full-res (B, Cr, H, W)
      - K,V: other, via Conv2d(kernel=ks, stride=s, padding=p)  -> (B, Cr, H', W')
    设置 stride=1 即可不做任何空间降采样（K/V 与 Q 同分辨率）。
    """
    def __init__(self, c, heads=4, red=2, kv_stride=4, kv_kernel=3, kv_padding=1):
        super().__init__()
        assert (c // red) % heads == 0, "Cr must be divisible by heads"
        self.h = heads
        self.s = kv_stride

        Cr = max(c // red, 1)

        # t <- r
        self.q_t = nn.Conv2d(c, Cr, 1, bias=False)
        self.k_r = nn.Conv2d(c, Cr, kernel_size=kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.v_r = nn.Conv2d(c, Cr, kernel_size=kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.o_t = nn.Conv2d(Cr, c, 1, bias=False)

        # r <- t
        self.q_r = nn.Conv2d(c, Cr, 1, bias=False)
        self.k_t = nn.Conv2d(c, Cr, kernel_size=kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.v_t = nn.Conv2d(c, Cr, kernel_size=kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.o_r = nn.Conv2d(Cr, c, 1, bias=False)

    def _attn_once(self, q, k, v):
        """
        q: (B,Cr,H,W), k/v: (B,Cr,H',W')
        return: (B,Cr,H,W)
        """
        B, Cr, H, W   = q.shape
        Hp, Wp        = k.shape[-2], k.shape[-1]
        h             = self.h
        d             = Cr // h

        Q = q.reshape(B, h, d, H*W).permute(0,1,3,2)     # (B,h,Nq,d)
        K = k.reshape(B, h, d, Hp*Wp).permute(0,1,3,2)   # (B,h,Nk,d)
        V = v.reshape(B, h, d, Hp*Wp).permute(0,1,3,2)   # (B,h,Nk,d)

        A = (Q @ K.transpose(-2, -1)) / (d ** 0.5)       # (B,h,Nq,Nk)
        A = A.softmax(dim=-1)
        O = A @ V                                        # (B,h,Nq,d)
        O = O.permute(0,1,3,2).reshape(B, Cr, H, W)      # (B,Cr,H,W)
        return O

    def forward(self, t, r):
        # t <- r
        q_t = self.q_t(t)                 # (B,Cr,H,W)
        k_r = self.k_r(r)                 # (B,Cr,H',W')，当 stride=1 时 H'=H
        v_r = self.v_r(r)                 # 同上
        dt  = self._attn_once(q_t, k_r, v_r)
        t_new = t + self.o_t(dt)

        # r <- t
        q_r = self.q_r(r)
        k_t = self.k_t(t)
        v_t = self.v_t(t)
        dr  = self._attn_once(q_r, k_t, v_t)
        r_new = r + self.o_r(dr)

        return t_new, r_new




class MLPInteractor(nn.Module):
    """
    纯 MLP 跨分支交互（baseline）:
      - 输入: t, r ∈ R^{B×C×H×W}
      - 拼接特征: concat([t, r, t-r, t*r], dim=1)  (可配 use_pairwise=False 则只用 [t,r])
      - 两个 1×1 conv + GELU 的 MLP，分别回归 Δt, Δr
      - 残差更新: t' = t + γ·Δt, r' = r + γ·Δr
    形状保持不变。
    """
    def __init__(self, c, hidden_mul=2, use_pairwise=True, norm='gn', res_scale=0.1):
        super().__init__()
        self.use_pairwise = use_pairwise
        in_ch = 2 * c if not use_pairwise else 4 * c          # [t, r] 或 [t, r, t-r, t*r]
        hidden = max(int(hidden_mul * c), c)

        Norm = nn.Identity
        if norm == 'bn':
            Norm = lambda cc: nn.BatchNorm2d(cc)
        elif norm == 'gn':
            # 与你主干一致，避免 BN 对 batch size 的依赖
            Norm = lambda cc: nn.GroupNorm(num_groups=min(32, cc), num_channels=cc, eps=1e-6)

        # 用“共享的特征抽取” + “分支专属的头” 更干净
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            Norm(hidden),
            nn.GELU(),
        )
        self.head_t = nn.Sequential(
            nn.Conv2d(hidden, c, kernel_size=1, bias=True)
        )
        self.head_r = nn.Sequential(
            nn.Conv2d(hidden, c, kernel_size=1, bias=True)
        )

        # 可学习残差缩放，默认 0.1 更稳定
        self.res_scale_t = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))
        self.res_scale_r = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

    def forward(self, t, r):
        if self.use_pairwise:
            x = torch.cat([t, r, t - r, t * r], dim=1)  # (B, 4C, H, W)
        else:
            x = torch.cat([t, r], dim=1)                # (B, 2C, H, W)

        h = self.trunk(x)                               # (B, hidden, H, W)
        dt = self.head_t(h)                             # (B, C, H, W)
        dr = self.head_r(h)                             # (B, C, H, W)

        t_new = t + self.res_scale_t * dt
        r_new = r + self.res_scale_r * dr
        return t_new, r_new


import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryCrossAttn(nn.Module):
    """
    Object-Query Cross-Attention (双向):
      Stage1: learned queries (per-head) 从对方分支提取 M 个上下文 token
      Stage2: 用当前分支的像素查询与 token 再做注意力，把语义撒回像素
    形状保持不变: (t,r)->(t',r'), 均为 (B,C,H,W)
    """
    def __init__(self, c, n_queries=64, heads=4, red=2,
                 kv_stride=4, kv_kernel=3, kv_padding=1,
                 share_queries=False, res_scale=0.1):
        super().__init__()
        assert (c // red) % heads == 0, "C//red 必须能被 heads 整除"
        self.h  = heads
        self.M  = n_queries
        self.red = red
        self.share = share_queries

        Cr = max(c // red, 1)
        d  = Cr // heads
        self.Cr, self.d = Cr, d

        # --- K/V 生成（对方分支） ---
        # t <- r
        self.k_r = nn.Conv2d(c, Cr, kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.v_r = nn.Conv2d(c, Cr, kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        # r <- t
        self.k_t = nn.Conv2d(c, Cr, kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)
        self.v_t = nn.Conv2d(c, Cr, kv_kernel, stride=kv_stride, padding=kv_padding, bias=False)

        # --- Stage2 的像素查询（当前分支） ---
        self.qpix_t = nn.Conv2d(c, Cr, 1, bias=False)  # for t pixels
        self.qpix_r = nn.Conv2d(c, Cr, 1, bias=False)  # for r pixels

        # --- learned queries（Stage1 的 Q_learned）---
        # 形状: (1, h, M, d)，按 batch 广播
        if share_queries:
            self.q_tok = nn.Parameter(torch.randn(1, heads, n_queries, d) * 0.02)
            self.q_tok_t = self.q_tok
            self.q_tok_r = self.q_tok
        else:
            self.q_tok_t = nn.Parameter(torch.randn(1, heads, n_queries, d) * 0.02)  # 用于 t<-r
            self.q_tok_r = nn.Parameter(torch.randn(1, heads, n_queries, d) * 0.02)  # 用于 r<-t

        # --- 输出投影 + 残差缩放 ---
        self.proj_t = nn.Conv2d(Cr, c, 1, bias=False)
        self.proj_r = nn.Conv2d(Cr, c, 1, bias=False)
        self.res_scale_t = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))
        self.res_scale_r = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))

        # 可选：轻量 FFN（深度可分离卷积 + PW）
        self.ffn_t = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=True)
        )
        self.ffn_r = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=True)
        )

    def _stage1_tokens(self, q_tok, k_feat, v_feat):
        """
        Stage1: learned queries 从对方分支读出 M 个 token
        q_tok: (1,h,M,d)
        k_feat/v_feat: (B,Cr,H',W') -> (B,h,Nk,d)
        返回 tokens: (B,h,M,d)
        """
        B, Cr, Hp, Wp = k_feat.shape
        h, d = self.h, self.d
        Nk = Hp * Wp

        K = k_feat.reshape(B, h, d, Nk).permute(0,1,3,2)  # (B,h,Nk,d)
        V = v_feat.reshape(B, h, d, Nk).permute(0,1,3,2)  # (B,h,Nk,d)

        # 扩展 learned queries 到 batch
        Q = q_tok.expand(B, -1, -1, -1)                   # (B,h,M,d)

        attn = (Q @ K.transpose(-2, -1)) / (d ** 0.5)     # (B,h,M,Nk)
        attn = attn.softmax(dim=-1)
        tokens = attn @ V                                 # (B,h,M,d)
        return tokens

    def _stage2_broadcast(self, qpix, tokens):
        """
        Stage2: 像素查询与 tokens 再 attention，广播回像素
        qpix:   (B,Cr,H,W) -> (B,h,Nq,d)
        tokens: (B,h,M,d)
        返回 out: (B,Cr,H,W)
        """
        B, Cr, H, W = qpix.shape
        h, d = self.h, self.d
        Nq = H * W

        Qp = qpix.reshape(B, h, d, Nq).permute(0,1,3,2)   # (B,h,Nq,d)
        K2 = tokens                                       # (B,h,M,d)
        V2 = tokens
        attn2 = (Qp @ K2.transpose(-2, -1)) / (d ** 0.5)  # (B,h,Nq,M)
        attn2 = attn2.softmax(dim=-1)
        O  = attn2 @ V2                                   # (B,h,Nq,d)
        O  = O.permute(0,1,3,2).reshape(B, Cr, H, W)      # (B,Cr,H,W)
        return O

    def _one_dir(self, src, other, q_tok, k_other, v_other, qpix_src, proj_out, res_scale, ffn):
        """
        单向: src <- other
        """
        # Stage1: learned queries 读取对方 tokens
        tokens = self._stage1_tokens(q_tok, k_other, v_other)  # (B,h,M,d)
        # Stage2: 像素 attention 撒回像素
        qpix = qpix_src(src)                                   # (B,Cr,H,W)
        out  = self._stage2_broadcast(qpix, tokens)            # (B,Cr,H,W)
        # 残差 + 轻量 FFN
        src2 = src + res_scale * proj_out(out)                 # (B,C,H,W)
        src2 = src2 + ffn(src2)
        return src2

    def forward(self, t, r):
        # t <- r
        t_new = self._one_dir(
            src=t, other=r,
            q_tok=self.q_tok_t,
            k_other=self.k_r(r), v_other=self.v_r(r),
            qpix_src=self.qpix_t, proj_out=self.proj_t,
            res_scale=self.res_scale_t, ffn=self.ffn_t
        )
        # r <- t
        r_new = self._one_dir(
            src=r, other=t,
            q_tok=self.q_tok_r,
            k_other=self.k_t(t), v_other=self.v_t(t),
            qpix_src=self.qpix_r, proj_out=self.proj_r,
            res_scale=self.res_scale_r, ffn=self.ffn_r
        )
        return t_new, r_new



def make_interactor(mode: str, c: int, **kwargs):
    mode = (mode or "none").lower()

    # 通用默认
    heads   = kwargs.get("heads", 4)
    red     = kwargs.get("red", 2)

    if mode == "crossse":
        return CrossSE(c)
    elif mode == "gatedadd":
        return GatedAdd(c)
    elif mode == "deltaproduct":
        return DeltaProduct(c)

    elif mode == "xattn_global":
        # K/V token 用 avgpool 的版本
        pool_stride = kwargs.get("pool_stride", 4)
        return XAttnGlobalTokens(c, heads=heads, red=red, pool_stride=pool_stride)

    elif mode == "xattn_tokens_conv":
        # K/V token 用可学习卷积的版本（可 kv_stride=1）
        kv_stride = kwargs.get("kv_stride", 4)
        kv_kernel = kwargs.get("kv_kernel", 3)
        kv_padding= kwargs.get("kv_padding", 1)
        return XAttnTokensConv(c, heads=heads, red=red,
                               kv_stride=kv_stride, kv_kernel=kv_kernel, kv_padding=kv_padding)

    elif mode == "xattn_deform":
        deform_points       = kwargs.get("deform_points", 8)
        deform_offset_scale = kwargs.get("deform_offset_scale", 1.0)
        ffn_expand          = kwargs.get("ffn_expand", 2)
        return DeformableCrossAttention(c, heads=heads, red=red,
                                        n_points=deform_points,
                                        offset_scale=deform_offset_scale,
                                        ffn_expand=ffn_expand)

    elif mode == "xattn_query":
        # ★ 关键：此处读取 query_tokens 等参数；没有就走默认
        n_queries   = kwargs.get("query_tokens", 64)
        kv_stride   = kwargs.get("kv_stride", 4)
        kv_kernel   = kwargs.get("kv_kernel", 3)
        kv_padding  = kwargs.get("kv_padding", 1)
        share_q     = kwargs.get("query_share", False)
        res_scale   = kwargs.get("query_res_scale", 0.1)
        return QueryCrossAttn(c,
                              n_queries=n_queries, heads=heads, red=red,
                              kv_stride=kv_stride, kv_kernel=kv_kernel, kv_padding=kv_padding,
                              share_queries=share_q, res_scale=res_scale)

    elif mode == "mlp_fuse":
        mlp_hidden_mul   = kwargs.get("mlp_hidden_mul", 2)
        mlp_use_pairwise = kwargs.get("mlp_use_pairwise", True)
        mlp_norm         = kwargs.get("mlp_norm", "gn")
        mlp_res_scale    = kwargs.get("mlp_res_scale", 0.1)
        return MLPInteractor(c,
                             hidden_mul=mlp_hidden_mul,
                             use_pairwise=mlp_use_pairwise,
                             norm=mlp_norm,
                             res_scale=mlp_res_scale)

    elif mode == "none":
        return None
    else:
        raise ValueError(f"Unknown interaction mode: {mode}")







class DualEncoderWithInteraction(nn.Module):
    def __init__(self, cfg,
                 mode="crossse",
                 use_on=("b1","b2"),
                 strides=None,              # 仅 xattn_global 使用
                 heads=4, red=2,
                 kv_strides=None, kv_kernel=3, kv_padding=1,
                 interactor_kwargs: dict = None):   # ★ 新增
        super().__init__()
        self.tar_backbone = ResNetV4(block_units=cfg.resnet.num_layers, width_factor=cfg.resnet.width_factor)
        self.ref_backbone = ResNetV2(block_units=cfg.resnet.num_layers, width_factor=cfg.resnet.width_factor)

        width  = int(64 * cfg.resnet.width_factor)
        c_root = width
        c_b1   = width * 4
        c_b2   = width * 8

        self.use_root = "root" in use_on
        self.use_b1   = "b1"   in use_on
        self.use_b2   = "b2"   in use_on

        strides    = strides or {"root":8, "b1":4, "b2":2}
        kv_strides = kv_strides or {"root":8, "b1":4, "b2":2}
        interactor_kwargs = dict(interactor_kwargs or {})  # 复制一份，避免原地修改

        # root
        if self.use_root:
            kwargs_root = dict(interactor_kwargs)
            kwargs_root.update(dict(
                heads=heads, red=red,
                pool_stride=strides.get("root", 8),
                kv_stride=kv_strides.get("root", 8),
                kv_kernel=kv_kernel, kv_padding=kv_padding
            ))
            self.xfer_root = make_interactor(mode, c_root, **kwargs_root)
        else:
            self.xfer_root = None

        # b1
        if self.use_b1:
            kwargs_b1 = dict(interactor_kwargs)
            kwargs_b1.update(dict(
                heads=heads, red=red,
                pool_stride=strides.get("b1", 4),
                kv_stride=kv_strides.get("b1", 4),
                kv_kernel=kv_kernel, kv_padding=kv_padding
            ))
            self.xfer_b1 = make_interactor(mode, c_b1, **kwargs_b1)
        else:
            self.xfer_b1 = None

        # b2
        if self.use_b2:
            kwargs_b2 = dict(interactor_kwargs)
            kwargs_b2.update(dict(
                heads=heads, red=red,
                pool_stride=strides.get("b2", 2),
                kv_stride=kv_strides.get("b2", 2),
                kv_kernel=kv_kernel, kv_padding=kv_padding
            ))
            self.xfer_b2 = make_interactor(mode, c_b2, **kwargs_b2)
        else:
            self.xfer_b2 = None

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    


    @staticmethod
    def _align_to(x, right_size):
        b, c, h, w = x.shape
        if h == right_size and w == right_size:
            return x
        assert 0 < (right_size - h) < 3 and 0 < (right_size - w) < 3, f"x {x.shape} should {right_size}"
        feat = torch.zeros((b, c, right_size, right_size), device=x.device, dtype=x.dtype)
        feat[:, :, 0:h, 0:w] = x
        return feat

    def forward(self, target_2ch, reference_3ch):
        B, _, in_size, _ = target_2ch.size()

        # ----- root -----
        t = self.tar_backbone.root(target_2ch)      # (B,64,112,112)
        r = self.ref_backbone.root(reference_3ch)   # (B,64,112,112)
        if self.xfer_root is not None:
            t, r = self.xfer_root(t, r)            # 交互(112×112)
        feat_root_t = t
        feat_root_r = r

        # ----- pool -----
        t = self.pool(t)                            # (B,64,55,55)
        r = self.pool(r)

        # ----- block1 -----
        b1_t = self.tar_backbone.body._modules['block1'](t)   # ~ (B,256,55,55)
        b1_r = self.ref_backbone.body._modules['block1'](r)
        right_size_b1 = int(in_size / 4 / (0+1))              # 56
        feat_b1_t = self._align_to(b1_t, right_size_b1)       # (B,256,56,56)
        feat_b1_r = self._align_to(b1_r, right_size_b1)
        if self.xfer_b1 is not None:
            feat_b1_t, feat_b1_r = self.xfer_b1(feat_b1_t, feat_b1_r)   # 交互(56×56)

        # ----- block2 -----
        b2_t = self.tar_backbone.body._modules['block2'](feat_b1_t)     # ~ (B,512,27,27)
        b2_r = self.ref_backbone.body._modules['block2'](feat_b1_r)
        right_size_b2 = int(in_size / 4 / (1+1))                        # 28
        feat_b2_t = self._align_to(b2_t, right_size_b2)                 # (B,512,28,28)
        feat_b2_r = self._align_to(b2_r, right_size_b2)
        if self.xfer_b2 is not None:
            feat_b2_t, feat_b2_r = self.xfer_b2(feat_b2_t, feat_b2_r)   # 交互(28×28)

        # ----- block3 （主干输出）-----
        x_tar = self.tar_backbone.body._modules['block3'](feat_b2_t)    # ~ (B,1024,13/14,13/14)
        x_ref = self.ref_backbone.body._modules['block3'](feat_b2_r)

        features_tar = [feat_b2_t, feat_b1_t, feat_root_t]               # [(512,28,28),(256,56,56),(64,112,112)]
        features_ref = [feat_b2_r, feat_b1_r, feat_root_r]
        return x_tar, features_tar, x_ref, features_ref


