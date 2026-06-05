import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import json
from PIL import Image
import math
import copy
from vit_seg_modeling_resnet_skip import ResNetV2, ResNetV3
class Dataset_middle(Dataset):   #use middle slice GT to extract box prompt, middle slice as reference
    def __init__(self, base_dir, mode):
        
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.ref_map = {}
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        dict_path = os.path.join(self.data_dir, self.mode, "annotation_dict_middle.json")
        if os.path.exists(dict_path):
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
        else:
            print(f"[WARN] middle-slice json not found: {dict_path}. Will fall back to per-sample self-reference.")
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                mask_path = ct_path.replace('CT', 'Mask')
                
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0


        x, y = image.shape
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2]==4:
            img_3c = image[:,:,:3]
        else:
            img_3c = image
        # if x != self.output_size[0] or y != self.output_size[1]:
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        mask_1024 = transform.resize(
                mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        
        case_num, slice_num = mask_path.split('/')[-2:]

        # Default to the current slice as its own reference; the JSON lookup
        # below overrides these when it records a usable middle-slice reference.
        ref_image_path = image_path
        ref_mask_path = mask_path
        info = self.ref_map.get(case_num)
        if isinstance(info, dict):
            # Prefer the absolute path recorded in the JSON
            cand_ct  = info.get("ct_path")
            cand_msk = info.get("mask_path")  # may be None
            if cand_ct and os.path.exists(cand_ct):
                ref_image_path = cand_ct
            else:
                print(f"[WARN] middle-slice not found: {cand_ct}. Will fall back to current slice.")
            # ref_mask_path may be None (if the patient has no mask); fall back to the current sample's mask
            if cand_msk and os.path.exists(cand_msk):
                ref_mask_path = cand_msk
            else:
                print(f"[WARN] middle-slice not found: {cand_msk}. Will fall back to current slice.")

        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        
        if len(ref_image.shape) == 2:
            ref_img_3c = np.repeat(ref_image[:, :, None], 3, axis=-1)
        elif len(ref_image.shape) == 3 and ref_image.shape[2]==4:
            ref_img_3c = ref_image[:,:,:3]
        else:
            ref_img_3c = ref_image
        # if x != self.output_size[0] or y != self.output_size[1]:
        ref_img_1024 = transform.resize(
            ref_img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        ref_img_1024 = (ref_img_1024 - ref_img_1024.min()) / np.clip(
            ref_img_1024.max() - ref_img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        ref_img_1024 = np.transpose(ref_img_1024, (2, 0, 1))
       

        # add perturbation to bounding box coordinates
        H, W = ref_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            # Use the second-to-last directory name as case_name
       
        # Print every key in the sample dict
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_img_1024).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_neighbor(Dataset):   # use NEIGHBOR slice GT to extract box prompt, neighbor slice as reference
    def __init__(self, base_dir, mode):
        self.data_dir = base_dir
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20

        self.image_paths, self.mask_paths = self._get_image_mask_paths()

        self.ref_map = {}
        dict_path = os.path.join(self.data_dir, self.mode, "annotation_dict_neighbor.json")
        if os.path.exists(dict_path):
            print(f"[INFO] neighbor-slice json found: {dict_path}")
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
        else:
            raise FileNotFoundError(f"[FATAL] neighbor-slice json not found: {dict_path}")

    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")

        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            if not os.path.isdir(patient_ct_folder):
                continue
            for ct_filename in os.listdir(patient_ct_folder):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                if not os.path.isfile(ct_path):
                    continue
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
                else:
                    raise FileNotFoundError(
                        f"[FATAL] Missing mask for CT slice:\n  CT: {ct_path}\n  Mask: {mask_path}"
                    )
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path  = self.mask_paths[idx]
        img_name   = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask  = mask / 255.0

        # --- image -> 3c -> 1024 ---
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img_3c = image[:, :, :3]
        else:
            img_3c = image

        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024 = np.transpose(img_1024, (2, 0, 1))

        mask_1024 = transform.resize(
            mask, (1024, 1024), order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )

        # ========== [CHANGED] pull the reference from the neighbor JSON ==========
        case_num, slice_num = mask_path.split('/')[-2:]  # case_num=patient, slice_num=filename

        if case_num not in self.ref_map:
            raise KeyError(f"[FATAL] Patient '{case_num}' not found in neighbor json.")
        if slice_num not in self.ref_map[case_num]:
            raise KeyError(f"[FATAL] Slice '{slice_num}' not found in neighbor json for patient '{case_num}'.")

        entry = self.ref_map[case_num][slice_num]
        ref_mask_path  = entry.get("ref_mask_path", None)
        ref_image_path = entry.get("ref_ct_path", None)

        if not ref_mask_path or not os.path.exists(ref_mask_path):
            raise FileNotFoundError(
                f"[FATAL] ref_mask_path missing or not exists.\n"
                f"  patient={case_num}, slice={slice_num}\n"
                f"  ref_mask_path={ref_mask_path}"
            )
        if not ref_image_path or not os.path.exists(ref_image_path):
            raise FileNotFoundError(
                f"[FATAL] ref_ct_path missing or not exists.\n"
                f"  patient={case_num}, slice={slice_num}\n"
                f"  ref_ct_path={ref_image_path}"
            )

        # --- load ref mask -> 1024 ---
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32) / 255.0
        ref_mask_1024 = transform.resize(
            ref_mask, (1024, 1024), order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )

        # --- bbox from ref_mask ---
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError(
                f"[FATAL] ref_mask is empty (no positive pixels), cannot compute bbox.\n"
                f"  patient={case_num}, slice={slice_num}\n"
                f"  ref_mask_path={ref_mask_path}"
            )

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # --- load ref image -> 1024 ---
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        if len(ref_image.shape) == 2:
            ref_img_3c = np.repeat(ref_image[:, :, None], 3, axis=-1)
        elif len(ref_image.shape) == 3 and ref_image.shape[2] == 4:
            ref_img_3c = ref_image[:, :, :3]
        else:
            ref_img_3c = ref_image

        ref_img_1024 = transform.resize(
            ref_img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        ref_img_1024 = (ref_img_1024 - ref_img_1024.min()) / np.clip(
            ref_img_1024.max() - ref_img_1024.min(), a_min=1e-8, a_max=None
        )
        ref_img_1024 = np.transpose(ref_img_1024, (2, 0, 1))

        # --- bbox shift ---
        H, W = ref_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_img_1024).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class MedSAM_Wrapper(nn.Module):
    """Vanilla MedSAM (no TRACE refinement) — the baseline model. Mirrors the
    MedSAM_Wrapper used by the simulation drivers, and matches a finetuned MedSAM
    checkpoint that has no `refinement.*` weights."""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding, features = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 1:
                box_torch = box_torch.unsqueeze(0).unsqueeze(0)
            elif len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks


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
class LinearAlign(nn.Module):
    """1x1 Conv + BN (no activation); lightweight channel alignment."""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(ch)
    def forward(self, x):
        return self.bn(self.conv(x))
def compute_conf_and_pyramid(ref_mask, ref_gt, w_g=0.7, detach=True):
    """
    ref_mask: (B,1,224,224) probability (foreground channel)
    ref_gt:   (B,1,224,224)  {0,1}
    return: (conf_112, conf_56, conf_28); three scales
    """
    eps = 1e-6
    p = ref_mask.clamp(eps, 1 - eps)
    # Agreement with the GT
    conf_gt = 1.0 - (p - ref_gt).abs()                    # [0,1]
    # Entropy-based confidence (1 - normalized entropy)
    H = -(p*torch.log(p) + (1-p)*torch.log(1-p))          # [0, log2]
    conf_ent = 1.0 - (H / (torch.log(torch.tensor(2.0, device=p.device))))
    conf = torch.clamp(w_g * conf_gt + (1 - w_g) * conf_ent, 0.0, 1.0)  # (B,1,224,224)
    if detach:
        conf = conf.detach()

    conf_112 = F.interpolate(conf, size=(512,512), mode='bilinear', align_corners=True)
    conf_56  = F.interpolate(conf, size=(256,256),   mode='bilinear', align_corners=True)
    conf_28  = F.interpolate(conf, size=(128,128),   mode='bilinear', align_corners=True)
    return conf_112, conf_56, conf_28
class Decoder_AlignPlusConf(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fuse = Conv2dReLU(1024, 512, kernel_size=3, padding=1)

        # Lightweight alignment
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
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)   # 1 output channel (MedSAM uses a single-channel mask, not 2 classes)

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs

        x = self.init_fuse(x_tar)                        #(B,1024,64,64) -> (B,512,64,64)    # (B,512,14,14)

        x = self.up3(x)                                  # (B,512,64,64) -> (B,512,128,128)                # (B,512,28,28)
        t3a = self.t_align3(t3); r3a = self.r_align3(r3) # (B,512,128,128) -> (B,512,128,128)  # (B,512,28,28)
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
    and project it into the interface the original decoder expects:
      x        = (B, 1024, 14, 14)   <- projected from layer3 (14x14) to 1024 channels
      features = [ (B,512,28,28), (B,256,56,56), (B,64,112,112) ]
                 ^    layer2         ^ layer1         ^ stem (upsampled)
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
            out_indices=(0, 1, 2, 3),  # [112, 56, 28, 14]
            in_chans=backbone_in
        )
        chs = [fi['num_chs'] for fi in self.backbone.feature_info]  # resnet18: [64, 64, 128, 256]

        # Align channels with what the decoder expects
        self.to_064_112 = nn.Conv2d(chs[0],  64, 1, bias=False)    # stem(112) → 64
        self.to_256_56  = nn.Conv2d(chs[1], 256, 1, bias=False)    # layer1(56) → 256
        self.to_512_28  = nn.Conv2d(chs[2], 512, 1, bias=False)    # layer2(28) → 512
        self.to_1024_14 = nn.Conv2d(chs[3], 1024,1, bias=False)    # layer3(14) → 1024

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)           # n -> 3 (only when reusing the 3-channel pretrained weights)
        f112, f56, f28, f14 = self.backbone(x)  # timm ordering: [112,56,28,14]

        # 112：stem → 64
        f112 = self.to_064_112(f112)
        # 56：layer1 → 256
        f56  = self.to_256_56(f56)
        # 28：layer2 → 512
        f28  = self.to_512_28(f28)
        # 14: layer3 -> 1024 (used as the bottleneck x)
        x14  = self.to_1024_14(f14)

        # Decoder expects order [512@28, 256@56, 64@112]
        features = [f28, f56, f112]
        return x14, features
class target_encoder_resnet(nn.Module):
    def __init__(self, config, img_size=1024, in_chans=2, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=True)
    def forward(self, x): return self.enc(x)
class reference_encoder_resnet(nn.Module):
    def __init__(self, config, img_size=1024, in_chans=3, variant='resnet18', pretrained=True):
        super().__init__()
        self.enc = ResNetEncoderAligned(variant=variant, in_chans=in_chans,
                                        pretrained=pretrained, keep_rgb_weights=False)
    def forward(self, x): return self.enc(x)
class TRACE(nn.Module):   #
    def __init__(self, config, img_size=1024, num_classes=2, zero_head=False, vis=False, pretrained=True):
        super(TRACE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.target_encoder = target_encoder_resnet(config, img_size, pretrained=pretrained)
        self.reference_encoder = reference_encoder_resnet(config, img_size, pretrained=pretrained)
        self.decoder = Decoder_AlignPlusConf()

        self.config = config

    def forward(self, target, reference):
        # Reference input = [ref_image, ref_mask, ref_gt], shape (B,3,224,224)
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

        

        return logits
class MedSAM_with_TRACE(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        refinement,
        refine_iters = 3,
        detach_between_iters = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.refinement = refinement
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, ref_image, ref_gt):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        ref_image_embedding, ref_features = self.image_encoder(ref_image)  # (B, 256, 64, 64)
        # f0, f1, f2 = features  # bs,64,64,768
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ori_mask = torch.sigmoid(ori_res_masks)  # (bs, 1, 1024, 1024)
        

        low_res_ref_masks, _ = self.mask_decoder(
            image_embeddings=ref_image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_ref_masks = F.interpolate(
            low_res_ref_masks,
            size=(ref_image.shape[2], ref_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        ori_mask = torch.sigmoid(ori_res_masks)  # (bs, 1, 1024, 1024)
        ref_mask = torch.sigmoid(ori_res_ref_masks)
        target_image = torch.cat([image[:,0:1,:,:], ori_mask], dim=1)  #bs,2,224,224
        ref_3ch = torch.cat([ref_image[:,0:1,:,:], ref_mask, ref_gt], dim=1)  #bs,3,224,224
        preds_all = []  
        final_pred = self.refinement(target_image, ref_3ch)  #bs,2,224,224
        preds_all.append(final_pred)

        for it in range(1, max(1, self.refine_iters) ):
        # Feed the previous iteration's output as the next iteration's target prediction
            next_mask = torch.sigmoid(final_pred)

            if self.detach_between_iters:
                next_mask = next_mask.detach()

            target_2ch = torch.cat([image[:,0:1,:,:], next_mask], dim=1)      # (B,2,H,W)
            # Reference typically stays unchanged; if you want to update ref_pred too, swap ref_mask for the iterated logits_ref
            # Keep ref_3ch unchanged here:
            # ref_3ch = torch.cat([ref_image_raw, ref_mask, ref_gt], dim=1)

            final_pred = self.refinement(target_2ch, ref_3ch)
            preds_all.append(final_pred)
        return {"final": final_pred, "iters": preds_all}

