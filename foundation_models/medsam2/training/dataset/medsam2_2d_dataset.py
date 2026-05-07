# -*- coding: utf-8 -*-
"""
MedSAM2 2D训练数据集
支持middle和neighbor两种参考slice模式
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
    MedSAM2 2D训练数据集
    从middle或neighbor slice的GT Mask中提取box prompt
    """
    def __init__(self, base_dir, mode='train', ref_type='middle', image_size=512, bbox_shift=10, use_augmentation=True):
        """
        Args:
            base_dir: 数据集根目录，例如 './2D_data/colon'
            mode: 'train' 或 'test'
            ref_type: 'middle' 或 'neighbor'
            image_size: 图像尺寸，默认512（MedSAM2通常使用512x512）
            bbox_shift: bounding box的随机扰动范围
            use_augmentation: 是否使用数据增强（训练时使用，测试时不使用）
        """
        self.base_dir = base_dir
        self.dataset_name = os.path.basename(base_dir.rstrip('/'))
        self.mode = mode
        self.ref_type = ref_type
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # 获取图像和mask路径
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
        
        # 加载参考slice的annotation字典
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
        """获取所有图像和mask的路径"""
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
        对bbox的四个角点应用仿射变换，然后重新计算边界框
        
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
        
        # 创建一个包含bbox的mask
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        bbox_mask[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 255
        
        # 转换为PIL Image
        bbox_mask_pil = Image.fromarray(bbox_mask)
        
        # 应用仿射变换
        bbox_mask_transformed = F.affine(
            bbox_mask_pil,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=(shx, shy),
            interpolation=InterpolationMode.NEAREST,
            fill=0
        )
        
        # 转换为numpy
        bbox_mask_transformed = np.array(bbox_mask_transformed)
        
        # 从变换后的mask提取bbox
        y_indices, x_indices = np.where(bbox_mask_transformed > 128)
        if len(y_indices) > 0 and len(x_indices) > 0:
            new_x_min = float(np.min(x_indices))
            new_x_max = float(np.max(x_indices))
            new_y_min = float(np.min(y_indices))
            new_y_max = float(np.max(y_indices))
        else:
            # 如果变换后bbox消失，返回原bbox
            new_x_min, new_y_min, new_x_max, new_y_max = x_min, y_min, x_max, y_max
        
        # 限制在图像范围内
        new_x_min = max(0, min(W - 1, new_x_min))
        new_x_max = max(0, min(W - 1, new_x_max))
        new_y_min = max(0, min(H - 1, new_y_min))
        new_y_max = max(0, min(H - 1, new_y_max))
        
        return np.array([new_x_min, new_y_min, new_x_max, new_y_max], dtype=np.float32)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(image_path)
        
        # 读取图像和mask
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0  # 归一化到[0, 1]
        
        # 转换为RGB格式
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img_3c = image[:, :, :3]
        else:
            img_3c = image
        
        # 调整图像大小到image_size
        img_resized = transform.resize(
            img_3c, (self.image_size, self.image_size), 
            order=3, preserve_range=True, anti_aliasing=True
        )
        
        # 确保图像值在[0, 255]范围内
        if img_resized.max() > 255 or img_resized.min() < 0:
            img_resized = (img_resized - img_resized.min()) / np.clip(
                img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
            ) * 255.0
        img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
        
        # 调整mask大小
        mask_resized = transform.resize(
            mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )
        
        # 获取参考mask路径
        case_num = os.path.basename(os.path.dirname(mask_path))
        ref_mask_path = None
        ref_image_path = None
        
        if self.ref_type == 'middle':
            # middle模式：每个病人有一个middle slice
            info = self.ref_map.get(case_num)
            if isinstance(info, dict):
                cand_ct = info.get("ct_path")
                cand_msk = info.get("mask_path")
                if cand_msk and os.path.exists(cand_msk):
                    ref_mask_path = cand_msk
                if cand_ct and os.path.exists(cand_ct):
                    ref_image_path = cand_ct
        elif self.ref_type == 'neighbor':
            # neighbor模式：每个slice有一个对应的neighbor slice
            slice_num = os.path.basename(mask_path)
            if case_num in self.ref_map and slice_num in self.ref_map[case_num]:
                entry = self.ref_map[case_num][slice_num]
                ref_mask_path = entry.get("ref_mask_path")
                ref_image_path = entry.get("ref_ct_path")
        
        # 如果没有找到参考mask，报错
        if ref_mask_path is None or not os.path.exists(ref_mask_path):
            raise FileNotFoundError(f"Reference mask not found: {ref_mask_path}")
        
        # 读取参考mask和图像（for the TRACE variant）
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32) / 255.0
        ref_mask_resized = transform.resize(
            ref_mask, (self.image_size, self.image_size),
            order=0, preserve_range=True, mode="constant", anti_aliasing=False
        )
        
        # 读取参考图像（如果存在）
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
        
        # 从参考mask中提取bounding box
        y_indices, x_indices = np.where(ref_mask_resized > 0.5)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # 如果参考mask为空，使用当前mask
            y_indices, x_indices = np.where(mask_resized > 0.5)
            if len(y_indices) == 0 or len(x_indices) == 0:
                # 如果当前mask也为空，返回一个默认的box
                x_min, x_max = 0, self.image_size - 1
                y_min, y_max = 0, self.image_size - 1
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 添加随机扰动
        H, W = self.image_size, self.image_size
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W - 1, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H - 1, y_max + random.randint(0, self.bbox_shift))
        
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        # 转换为PIL Image（用于数据增强和ToTensor）
        img_pil = Image.fromarray(img_resized, mode='RGB')
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8), mode='L')
        
        # 处理参考图像和mask(for the TRACE variant)
        ref_img_pil = None
        ref_mask_pil = None
        if ref_image_resized is not None:
            ref_img_pil = Image.fromarray(ref_image_resized, mode='RGB')
            ref_mask_pil = Image.fromarray((ref_mask_resized * 255).astype(np.uint8), mode='L')
        
        # 数据增强（仅在训练时）
        if self.use_augmentation:
            # 1. RandomHorizontalFlip (p=0.5)
            if random.random() < 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)
                # 对参考图像和mask也应用翻转(for the TRACE variant)
                if ref_img_pil is not None:
                    ref_img_pil = F.hflip(ref_img_pil)
                if ref_mask_pil is not None:
                    ref_mask_pil = F.hflip(ref_mask_pil)
                # 翻转bbox的x坐标
                W = self.image_size
                x_min_new = W - 1 - bbox[2]
                x_max_new = W - 1 - bbox[0]
                bbox[0] = x_min_new
                bbox[2] = x_max_new
            
            # 2. RandomAffine (degrees=25, shear=20)
            if random.random() < 1.0:  # 始终应用（p=1.0 in config）
                img_size = [self.image_size, self.image_size]  # [width, height]
                affine_params = T.RandomAffine.get_params(
                    degrees=[-25, 25],
                    translate=None,
                    scale_ranges=None,
                    shears=[-20, 20],
                    img_size=img_size,
                )
                
                # 应用仿射变换到图像和mask
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
                
                # 对参考图像和mask也应用相同的变换(for the TRACE variant)
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
                
                # 应用仿射变换到bbox
                bbox = self._apply_affine_to_bbox(bbox, affine_params, img_size)
            
            # 3. ColorJitter (brightness=0.1, contrast=0.03, saturation=0.03, hue=0即不变换色调)
            if random.random() < 1.0:  # 始终应用
                color_jitter = T.ColorJitter(
                    brightness=0.1,
                    contrast=0.03,
                    saturation=0.03,
                    hue=0  # 不变换色调（对应MedSAM2配置中的hue: null）
                )
                img_pil = color_jitter(img_pil)
            
            # 4. RandomGrayscale (p=0.05)
            if random.random() < 0.05:
                grayscale = T.Grayscale(num_output_channels=3)
                img_pil = grayscale(img_pil)
        
        # 转换为tensor（0-255 -> 0-1）
        img_tensor = F.to_tensor(img_pil)  # (3, H, W), float [0, 1]
        
        # Normalize（ImageNet mean/std，与SAM2Transforms保持一致）
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_tensor = F.normalize(img_tensor, mean=mean, std=std)  # (3, H, W)
        
        # mask: PIL -> tensor -> (1, H, W)
        # F.to_tensor() 已经自动将0-255的PIL Image转换为0-1的float tensor，不需要再除以255.0
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
        
        # 添加参考图像和mask（for the TRACE variant）
        if ref_img_pil is not None:
            ref_img_tensor = F.to_tensor(ref_img_pil)  # (3, H, W), float [0, 1]
            ref_img_tensor = F.normalize(ref_img_tensor, mean=mean, std=std)  # (3, H, W)
            result['ref_image'] = ref_img_tensor
        
        if ref_mask_pil is not None:
            ref_mask_tensor = F.to_tensor(ref_mask_pil).squeeze(0).float()  # (H, W), float [0, 1]
            ref_mask_tensor = ref_mask_tensor.unsqueeze(0)  # (1, H, W)
            result['ref_gt'] = ref_mask_tensor  # ref_mask_resized 就是 ref_gt
        
        return result