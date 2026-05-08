# -*- coding: utf-8 -*-
"""
Refinement module for MedSAM2 + TRACE variant
Bundles the add-on module (TRACE) and its dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU"""
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


class LinearAlign(nn.Module):
    """1x1 Conv + BN (no activation); lightweight channel alignment."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
    def forward(self, x):
        return self.bn(self.conv(x))


@torch.no_grad()
def compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True):
    """
    Compute the confidence pyramid.
    Args:
        ref_mask: (B,1,H,W) probability (foreground channel)
        ref_gt:   (B,1,H,W) {0,1}
        w_g: weight, default 0.7
        detach: whether to detach gradients
    Returns:
        (conf_64, conf_128, conf_256): three-scale confidence maps matching the decoder's three fusion points
        For a 512x512 input, the decoder fuses at 64x64, 128x128 and 256x256 scales
    """
    eps = 1e-6
    p = ref_mask.clamp(eps, 1 - eps)
    # Agreement with the GT
    conf_gt = 1.0 - (p - ref_gt).abs()                    # [0,1]
    # Entropy-based confidence (1 - normalized entropy)
    H = -(p*torch.log(p) + (1-p)*torch.log(1-p))          # [0, log2]
    conf_ent = 1.0 - (H / (torch.log(torch.tensor(2.0, device=p.device))))
    conf = torch.clamp(w_g * conf_gt + (1 - w_g) * conf_ent, 0.0, 1.0)  # (B,1,H,W)
    if detach:
        conf = conf.detach()

    # For a 512x512 input, build three-scale confidence maps
    # Matching the decoder's three fusion scales: 64x64, 128x128, 256x256
    conf_64 = F.interpolate(conf, size=(64,64), mode='bilinear', align_corners=True)
    conf_128 = F.interpolate(conf, size=(128,128), mode='bilinear', align_corners=True)
    conf_256 = F.interpolate(conf, size=(256,256), mode='bilinear', align_corners=True)
    return conf_64, conf_128, conf_256


class InputAdapter(nn.Module):
    """Project the n-channel [pred, image(, gt)] tensor down to 3 channels so the full ImageNet-pretrained backbone can be reused.
       Architecture: Conv3x3 -> BN -> GELU -> Conv1x1 (light and stable).
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
    Use timm's ResNet (resnet18 here) as the backbone,
    and project it into the interface the original decoder expects.
    
    For a 512x512 input, ResNet18 outputs:
    - stem (indices=0): 256x256, 64 channels
    - layer1 (indices=1): 128x128, 64 channels  
    - layer2 (indices=2): 64x64, 128 channels
    - layer3 (indices=3): 32x32, 256 channels
    
    Decoder expects:
    - x_tar: (B, 1024, 32, 32) projected from layer3
    - features: [(B,512,64,64), (B,256,128,128), (B,64,256,256)] 
                corresponds to [layer2 proj, layer1 proj, stem proj]
    """
    def __init__(self, variant='resnet18', in_chans=3, pretrained=True, keep_rgb_weights=None):
        super().__init__()
        if keep_rgb_weights is None:
            keep_rgb_weights = (pretrained and in_chans != 3)
        self.input_proj = InputAdapter(in_chans, out_ch=3) if keep_rgb_weights else None
        backbone_in = 3 if keep_rgb_weights else in_chans

        # Key: variant is set to resnet18
        self.backbone = create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # [256x256, 128x128, 64x64, 32x32] for 512x512 input
            in_chans=backbone_in
        )
        chs = [fi['num_chs'] for fi in self.backbone.feature_info]  # resnet18: [64, 64, 128, 256]

        # Align channels with what the decoder expects
        # f256: stem 256x256, 64ch -> 64ch (kept as-is, used for the final fusion)
        # f128: layer1 128x128, 64ch -> 256ch
        # f64: layer2 64x64, 128ch -> 512ch  
        # f32: layer3 32x32, 256ch -> 1024ch (used as the bottleneck x)
        self.to_064_256 = nn.Conv2d(chs[0],  64, 1, bias=False)    # stem(256x256, 64ch) → 64ch
        self.to_256_128 = nn.Conv2d(chs[1], 256, 1, bias=False)    # layer1(128x128, 64ch) → 256ch
        self.to_512_64  = nn.Conv2d(chs[2], 512, 1, bias=False)    # layer2(64x64, 128ch) → 512ch
        self.to_1024_32 = nn.Conv2d(chs[3], 1024,1, bias=False)    # layer3(32x32, 256ch) → 1024ch

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)           # n -> 3 (only when reusing the 3-channel pretrained weights)
        # timm ordering: [256x256, 128x128, 64x64, 32x32] for 512x512 input
        f256, f128, f64, f32 = self.backbone(x)

        # Align channels
        f256_proj = self.to_064_256(f256)   # (B,64,256,256)  -- corresponds to original t1
        f128_proj = self.to_256_128(f128)   # (B,256,128,128) -- corresponds to original t2
        f64_proj  = self.to_512_64(f64)     # (B,512,64,64)   -- corresponds to original t3
        f32_proj  = self.to_1024_32(f32)    # (B,1024,32,32)  -- corresponds to original x_tar

        # Decoder expects order [t3, t2, t1] = [512@64, 256@128, 64@256]
        features = [f64_proj, f128_proj, f256_proj]
        return f32_proj, features


class target_encoder_resnet(nn.Module):
    """Target encoder: takes 2 input channels (image + mask)."""
    def __init__(self, config, img_size=512, in_chans=2, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=True)
    def forward(self, x): 
        return self.enc(x)


class reference_encoder_resnet(nn.Module):
    """Reference encoder: takes 3 input channels (ref_image + ref_mask + ref_gt)."""
    def __init__(self, config, img_size=512, in_chans=3, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=False)
    def forward(self, x): 
        return self.enc(x)


class Decoder_AlignPlusConf(nn.Module):
    """
    Decoder with alignment and confidence weighting
    Adapted for 512x512 input:
    - Start from x_tar (1024@32x32)
    - up3: 32x32 -> 64x64, fused with t3 (512@64x64)
    - up2: 64x64 -> 128x128, fused with t2 (256@128x128)
    - up1: 128x128 -> 256x256, fused with t1 (64@256x256)
    - final_up: 256x256 -> 512x512
    """
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # Lightweight alignment
        self.t_align3 = LinearAlign(512); self.r_align3 = LinearAlign(512)
        self.t_align2 = LinearAlign(256); self.r_align2 = LinearAlign(256)
        self.t_align1 = LinearAlign(64);  self.r_align1 = LinearAlign(64)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = Conv2dReLU(1536, 256, kernel_size=3, padding=1)  # 512+512+512 = 1536

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = Conv2dReLU(768, 128, kernel_size=3, padding=1)  # 256+256+256 = 768

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = Conv2dReLU(256, 64, kernel_size=3, padding=1)  # 128+64+64 = 256

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)   # outputs 1-channel mask logits

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        """
        Args:
            t1, t2, t3: target features at 3 scales
                - t3: (B, 512, 64, 64) corresponds to layer2
                - t2: (B, 256, 128, 128) corresponds to layer1
                - t1: (B, 64, 256, 256) corresponds to stem
            x_tar: target bottleneck (B, 1024, 32, 32) corresponds to layer3
            r1, r2, r3: reference features at 3 scales (same as t1, t2, t3)
            x_ref: reference bottleneck (B, 1024, 32, 32)
            confs: (conf_64, conf_128, conf_256) confidence maps
        """
        assert confs is not None and len(confs) == 3, "need confs=(conf_64, conf_128, conf_256)"
        conf_64, conf_128, conf_256 = confs

        # Start at 32x32
        x = self.init_fuse(x_tar)                        # (B,1024,32,32) -> (B,512,32,32)

        # First upsample + fuse: 32x32 -> 64x64, with t3 (64x64)
        x = self.up3(x)                                  # (B,512,32,32) -> (B,512,64,64)
        t3a = self.t_align3(t3); r3a = self.r_align3(r3) # (B,512,64,64) -> (B,512,64,64)
        r3_use = r3a * conf_64                           # gated by conf_64
        x = self.dec3(torch.cat([x, t3a, r3_use], dim=1))# -> (B,256,64,64)

        # Second upsample + fuse: 64x64 -> 128x128, with t2 (128x128)
        x = self.up2(x)                                  # (B,256,64,64) -> (B,256,128,128)
        t2a = self.t_align2(t2); r2a = self.r_align2(r2) # (B,256,128,128) -> (B,256,128,128)
        r2_use = r2a * conf_128                          # gated by conf_128
        x = self.dec2(torch.cat([x, t2a, r2_use], dim=1))# -> (B,128,128,128)

        # Third upsample + fuse: 128x128 -> 256x256, with t1 (256x256)
        x = self.up1(x)                                  # (B,128,128,128) -> (B,128,256,256)
        t1a = self.t_align1(t1); r1a = self.r_align1(r1) # (B,64,256,256) -> (B,64,256,256)
        r1_use = r1a * conf_256                          # gated by conf_256
        x = self.dec1(torch.cat([x, t1a, r1_use], dim=1))# -> (B,64,256,256)

        # Final upsample to 512x512
        x = self.final_up(x)                             # (B,64,256,256) -> (B,64,512,512)
        return self.out_conv(x)                          # -> (B,1,512,512)


class TRACE(nn.Module):
    """
    Refinement module v57 for MedSAM2
    Uses ResNet18 encoder + Decoder_AlignPlusConf decoder.
    """
    def __init__(self, config, img_size=512, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(TRACE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = getattr(config, 'classifier', None)
        self.target_encoder = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()
        self.config = config

    def forward(self, target, reference):
        """
        Args:
            target: (B, 2, H, W) [image, mask]
            reference: (B, 3, H, W) [ref_image, ref_mask, ref_gt]
        Returns:
            logits: (B, 1, H, W) mask logits
        """
        ref_mask = reference[:, 1:2]   # (B,1,H,W)
        ref_gt   = reference[:, 2:3]   # (B,1,H,W)
        x_tar, features_tar = self.target_encoder(target)
        x_ref, features_ref = self.reference_encoder(reference)
        
        # Compute the confidence pyramid (matching the decoder's three fusion scales: 64x64, 128x128, 256x256)
        conf_64, conf_128, conf_256 = compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True)
        
        logits = self.decoder(
            features_tar[2], features_tar[1], features_tar[0], x_tar,
            features_ref[2], features_ref[1], features_ref[0], x_ref,
            confs=(conf_64, conf_128, conf_256)
        )
        return logits
