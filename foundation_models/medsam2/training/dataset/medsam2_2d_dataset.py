# -*- coding: utf-8 -*-
"""
MedSAM2 2D training dataset.
Supports both 'middle' and 'neighbor' reference-slice modes.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import transform
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class MedSAM2_2D_Dataset(Dataset):
    """
    MedSAM2 2D training dataset.
    Extract a box prompt from the middle- or neighbor-slice GT mask.
    """
    def __init__(self, base_dir, mode='train', ref_type='middle', image_size=512, bbox_shift=10, use_augmentation=True):
        """
        Args:
            base_dir: dataset root, e.g. './2D_data/colon'
            mode: 'train' or 'test'
            ref_type: 'middle' or 'neighbor'
            image_size: model input size, default 512 (MedSAM2 typically uses 512x512)
            bbox_shift: random perturbation range applied to the bounding box
            use_augmentation: enable data augmentation (used at train time, disabled at test time)
        """
        self.base_dir = base_dir
        self.dataset_name = os.path.basename(base_dir.rstrip('/'))
        self.mode = mode
        self.ref_type = ref_type
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # Collect image and mask paths
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        
        # Load the reference-slice annotation dict
        self.ref_map = {}
        if ref_type == 'middle':
            dict_path = os.path.join(self.base_dir, self.mode, "annotation_dict_middle.json")
        elif ref_type == 'neighbor':
            dict_path = os.path.join(self.base_dir, self.mode, "annotation_dict_neighbor.json")
        else:
            raise ValueError(f"ref_type must be 'middle' or 'neighbor', got {ref_type}")
        
        if os.path.exists(dict_path):
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
            print(f"[INFO] Loaded {ref_type} annotation dict from {dict_path}")
        else:
            print(f"[WARN] {ref_type} annotation dict not found: {dict_path}")
    
    def _get_image_mask_paths(self):
        """Collect all image and mask paths."""
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.base_dir, self.mode, "CT")
        
        if not os.path.exists(ct_dir):
            raise FileNotFoundError(f"CT directory not found: {ct_dir}")
        
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
        
        return image_paths, mask_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def _apply_affine_to_bbox(self, bbox, affine_params, img_size):
        """
        Apply an affine transform to the four bbox corners and recompute the bounding box.
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            affine_params: (angle, translate, scale, shear) from T.RandomAffine.get_params
            img_size: (width, height)
        
        Returns:
            transformed_bbox: [x_min, y_min, x_max, y_max]
        """
        angle, (tx, ty), scale, (shx, shy) = affine_params
        x_min, y_min, x_max, y_max = bbox
        W, H = img_size
        
        # Create a mask covering the bbox
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        bbox_mask[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 255
        
        # Convert to PIL Image
        bbox_mask_pil = Image.fromarray(bbox_mask)
        
        # Apply the affine transform
        bbox_mask_transformed = F.affine(
            bbox_mask_pil,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=(shx, shy),
            interpolation=InterpolationMode.NEAREST,
            fill=0
        )
        
        # Convert back to numpy
        bbox_mask_transformed = np.array(bbox_mask_transformed)
        
        # Extract a new bbox from the transformed mask
        y_indices, x_indices = np.where(bbox_mask_transformed > 128)
        if len(y_indices) > 0 and len(x_indices) > 0:
            new_x_min = float(np.min(x_indices))
            new_x_max = float(np.max(x_indices))
            new_y_min = float(np.min(y_indices))
            new_y_max = float(np.max(y_indices))
        else:
            # If the transformed bbox vanished, fall back to the original bbox
            new_x_min, new_y_min, new_x_max, new_y_max = x_min, y_min, x_max, y_max
        
        # Clamp to image bounds
        new_x_min = max(0, min(W - 1, new_x_min))
        new_x_max = max(0, min(W - 1, new_x_max))
        new_y_min = max(0, min(H - 1, new_y_min))
        new_y_max = max(0, min(H - 1, new_y_max))
        
        return np.array([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.float32)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(image_path)
        
        # Read the image and mask
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0  # normalize to [0, 1]
        
        # Convert to RGB
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img_3c = image[:, :, :3]
        else:
            img_3c = image
        
        # Resize the image to image_size
        img_resized = transform.resize(
            img_3c, (self.image_size, self.image_size), 
            order=3, preserve_range=True, anti_aliasing=True
        )
        
        # Make sure pixel values are in [0, 255]
        if img_resized.max() > 255 or img_resized.min() < 0:
            img_resized = (img_resized - img_resized.min()) / np.clip(
                img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
            ) * 255.0
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
        
        # Resize the mask
        mask_resized = transform.resize(
            mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )
        
        # Resolve the reference-mask path
        case_num = os.path.basename(os.path.dirname(mask_path))
        ref_mask_path = None
        ref_image_path = None
        
        if self.ref_type == 'middle':
            # 'middle' mode: one middle slice per patient
            info = self.ref_map.get(case_num)
            if isinstance(info, dict):
                cand_ct = info.get("ct_path")
                cand_msk = info.get("mask_path")
                if cand_msk and os.path.exists(cand_msk):
                    ref_mask_path = cand_msk
                if cand_ct and os.path.exists(cand_ct):
                    ref_image_path = cand_ct
        elif self.ref_type == 'neighbor':
            # 'neighbor' mode: each slice has its own neighbor slice
            slice_num = os.path.basename(mask_path)
            if case_num in self.ref_map and slice_num in self.ref_map[case_num]:
                entry = self.ref_map[case_num][slice_num]
                ref_mask_path = entry.get("ref_mask_path")
                ref_image_path = entry.get("ref_ct_path")
        
        # Raise if no reference mask was found
        if ref_mask_path is None or not os.path.exists(ref_mask_path):
            raise FileNotFoundError(f"Reference mask not found: {ref_mask_path}")
        
        # Read the reference mask and image (for the TRACE variant)
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32) / 255.0
        ref_mask_resized = transform.resize(
            ref_mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )
        
        # Read the reference image, if present
        ref_image_resized = None
        if ref_image_path and os.path.exists(ref_image_path):
            ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
            if len(ref_image.shape) == 2:
                ref_img_3c = np.repeat(ref_image[:, :, None], 3, axis=-1)
            elif len(ref_image.shape) == 3 and ref_image.shape[2] == 4:
                ref_img_3c = ref_image[:, :, :3]
            else:
                ref_img_3c = ref_image
            
            ref_image_resized = transform.resize(
                ref_img_3c, (self.image_size, self.image_size), 
                order=3, preserve_range=True, anti_aliasing=True
            )
            
            if ref_image_resized.max() > 255 or ref_image_resized.min() < 0:
                ref_image_resized = (ref_image_resized - ref_image_resized.min()) / np.clip(
                    ref_image_resized.max() - ref_image_resized.min(), a_min=1e-8, a_max=None
                ) * 255.0
            ref_image_resized = np.clip(ref_image_resized, 0, 255).astype(np.uint8)
        
        # Extract the bounding box from the reference mask
        y_indices, x_indices = np.where(ref_mask_resized > 0.5)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Fall back to the current mask if the reference mask is empty
            y_indices, x_indices = np.where(mask_resized > 0.5)
            if len(y_indices) == 0 or len(x_indices) == 0:
                # If the current mask is also empty, return a default box
                x_min, x_max = 0, self.image_size - 1
                y_min, y_max = 0, self.image_size - 1
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Apply random perturbation
        H, W = self.image_size, self.image_size
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W - 1, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H - 1, y_max + random.randint(0, self.bbox_shift))
        
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        # Convert to PIL Image (used for data augmentation and ToTensor)
        img_pil = Image.fromarray(img_resized, mode='RGB')
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8), mode='L')
        
        # Handle the reference image and mask (for the TRACE variant)
        ref_img_pil = None
        ref_mask_pil = None
        if ref_image_resized is not None:
            ref_img_pil = Image.fromarray(ref_image_resized, mode='RGB')
            ref_mask_pil = Image.fromarray((ref_mask_resized * 255).astype(np.uint8), mode='L')
        
        # Data augmentation (training only)
        if self.use_augmentation:
            # 1. RandomHorizontalFlip (p=0.5)
            if random.random() < 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)
                # Also flip the reference image and mask (for the TRACE variant)
                if ref_img_pil is not None:
                    ref_img_pil = F.hflip(ref_img_pil)
                if ref_mask_pil is not None:
                    ref_mask_pil = F.hflip(ref_mask_pil)
                # Flip the bbox x-coordinates
                W = self.image_size
                x_min_new = W - 1 - bbox[2]
                x_max_new = W - 1 - bbox[0]
                bbox[0] = x_min_new
                bbox[2] = x_max_new
            
            # 2. RandomAffine (degrees=25, shear=20)
            if random.random() < 1.0:  # always applied (p=1.0 in config)
                img_size = [self.image_size, self.image_size]  # [width, height]
                affine_params = T.RandomAffine.get_params(
                    degrees=[-25, 25],
                    translate=None,
                    scale_ranges=None,
                    shears=[-20, 20],
                    img_size=img_size,
                )
                
                # Apply the affine transform to the image and mask
                img_pil = F.affine(
                    img_pil,
                    *affine_params,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(0, 0, 0)
                )
                mask_pil = F.affine(
                    mask_pil,
                    *affine_params,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0
                )
                
                # Apply the same transform to the reference image and mask (for the TRACE variant)
                if ref_img_pil is not None:
                    ref_img_pil = F.affine(
                        ref_img_pil,
                        *affine_params,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(0, 0, 0)
                    )
                if ref_mask_pil is not None:
                    ref_mask_pil = F.affine(
                        ref_mask_pil,
                        *affine_params,
                        interpolation=InterpolationMode.NEAREST,
                        fill=0
                    )
                
                # Apply the affine transform to the bbox
                bbox = self._apply_affine_to_bbox(bbox, affine_params, img_size)
            
            # 3. ColorJitter (brightness=0.1, contrast=0.03, saturation=0.03, hue=0 -> no hue change)
            if random.random() < 1.0:  # always applied
                color_jitter = T.ColorJitter(
                    brightness=0.1,
                    contrast=0.03,
                    saturation=0.03,
                    hue=0  # no hue change (matches MedSAM2's config: hue: null)
                )
                img_pil = color_jitter(img_pil)
            
            # 4. RandomGrayscale (p=0.05)
            if random.random() < 0.05:
                grayscale = T.Grayscale(num_output_channels=3)
                img_pil = grayscale(img_pil)
        
        # Convert to tensor (0-255 -> 0-1)
        img_tensor = F.to_tensor(img_pil)  # (3, H, W), float [0, 1]
        
        # Normalize with ImageNet mean/std (matches SAM2Transforms)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_tensor = F.normalize(img_tensor, mean=mean, std=std)  # (3, H, W)
        
        # mask: PIL -> tensor -> (1, H, W)
        # F.to_tensor() already converts the 0-255 PIL Image to a 0-1 float tensor; no need to divide by 255.0
        mask_tensor = F.to_tensor(mask_pil).squeeze(0).float()  # (H, W), float [0, 1]
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
        
        # bbox: (4,)
        bbox_tensor = torch.from_numpy(bbox).float()
        
        result = {
            'image': img_tensor,  # (3, H, W), float, normalized
            'mask': mask_tensor,  # (1, H, W), float [0, 1]
            'bbox': bbox_tensor,  # (4,), float [x_min, y_min, x_max, y_max]
            'img_name': img_name,
        }
        
        # Add the reference image and mask (for the TRACE variant)
        if ref_img_pil is not None:
            ref_img_tensor = F.to_tensor(ref_img_pil)  # (3, H, W), float [0, 1]
            ref_img_tensor = F.normalize(ref_img_tensor, mean=mean, std=std)  # (3, H, W)
            result['ref_image'] = ref_img_tensor
        
        if ref_mask_pil is not None:
            ref_mask_tensor = F.to_tensor(ref_mask_pil).squeeze(0).float()  # (H, W), float [0, 1]
            ref_mask_tensor = ref_mask_tensor.unsqueeze(0)  # (1, H, W)
            result['ref_gt'] = ref_mask_tensor  # ref_mask_resized is the ref_gt
        
        return result