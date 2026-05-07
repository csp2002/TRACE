# -*- coding: utf-8 -*-
"""
MedSAM2 Video训练数据集（用于v6554实验）
将2D数据组织为video格式：每个病人的所有slice作为一个video
使用middle slice的GT作为reference
"""

import os
import json
import numpy as np
import torch
from typing import List, Optional, Dict
from PIL import Image
from skimage import io

from training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo


class MedSAM2VideoSegmentLoader:
    """
    Segment loader for MedSAM2 video dataset
    每个病人的所有slice的mask
    """
    def __init__(self, mask_paths: List[str]):
        """
        Args:
            mask_paths: list of mask file paths for all slices of a patient
        """
        self.mask_paths = sorted(mask_paths)  # 确保顺序一致
    
    def load(self, frame_idx, obj_ids=None):
        """
        Load mask for a specific frame (slice)
        
        Args:
            frame_idx: index of the frame (slice) in the video
            obj_ids: object IDs to load (not used, always load single object)
        
        Returns:
            dict: {obj_id: mask_tensor} where obj_id=1 (single object)
        """
        if frame_idx >= len(self.mask_paths):
            raise IndexError(f"Frame index {frame_idx} out of range for {len(self.mask_paths)} frames")
        
        mask_path = self.mask_paths[frame_idx]
        
        # Load mask (assume binary mask, 0=background, 255=foreground)
        mask = io.imread(mask_path)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # 取第一个通道
        
        # Convert to binary (0 or 1)
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Convert to torch tensor
        mask_tensor = torch.from_numpy(mask_binary).to(torch.uint8)
        
        # Return as dict with object_id=1 (single object per slice)
        return {1: mask_tensor}


class MedSAM2VideoRawDataset(VOSRawDataset):
    """
    Video dataset for MedSAM2 v6554 experiment
    将2D数据组织为video格式：每个病人的所有slice作为一个video
    每个slice作为一个frame
    """
    def __init__(
        self,
        base_dir: str,
        mode: str = 'train',
        image_size: int = 512,
    ):
        """
        Args:
            base_dir: 数据集根目录，例如 './2D_data/colon'
            mode: 'train' 或 'test'
            image_size: 图像尺寸（用于resize）
        """
        self.base_dir = base_dir
        self.dataset_name = os.path.basename(base_dir.rstrip('/'))
        self.mode = mode
        self.image_size = image_size
        
        # 加载middle slice的annotation字典
        dict_path = os.path.join(self.base_dir, self.mode, "annotation_dict_middle.json")
        self.ref_map = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r") as f:
                self.ref_map = json.load(f)
            print(f"[INFO] Loaded middle annotation dict from {dict_path}")
        else:
            print(f"[WARN] Middle annotation dict not found: {dict_path}")
        
        # 获取所有病人的数据
        self.patient_videos = self._organize_patient_videos()
        print(f"[INFO] Found {len(self.patient_videos)} patient videos in {mode} set")
    
    def _organize_patient_videos(self) -> List[Dict]:
        """
        组织病人的数据：每个病人的所有slice作为一个video
        
        Returns:
            List of dicts, each dict contains:
            - patient_name: str
            - video_id: int
            - image_paths: List[str] (sorted by slice number)
            - mask_paths: List[str] (sorted by slice number)
            - ref_image_path: str (middle slice image path)
            - ref_mask_path: str (middle slice mask path)
        """
        ct_dir = os.path.join(self.base_dir, self.mode, "CT")
        if not os.path.exists(ct_dir):
            raise FileNotFoundError(f"CT directory not found: {ct_dir}")
        
        patient_videos = []
        video_id = 0
        
        # 遍历所有病人文件夹
        for patient_folder in sorted(os.listdir(ct_dir)):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            if not os.path.isdir(patient_ct_folder):
                continue
            
            # 获取该病人的所有slice
            image_paths = []
            mask_paths = []
            
            for ct_filename in sorted(os.listdir(patient_ct_folder)):
                ct_path = os.path.join(patient_ct_folder, ct_filename)
                if not os.path.isfile(ct_path):
                    continue
                
                mask_path = ct_path.replace('CT', 'Mask')
                if os.path.exists(mask_path):
                    image_paths.append(ct_path)
                    mask_paths.append(mask_path)
            
            # 至少需要2个slice才能组成video
            if len(image_paths) < 2:
                print(f"[WARN] Patient {patient_folder} has less than 2 slices, skipping")
                continue
            
            # 获取middle slice的路径（从annotation_dict_middle.json）
            patient_name = patient_folder
            ref_image_path = None
            ref_mask_path = None
            
            if patient_name in self.ref_map:
                ref_info = self.ref_map[patient_name]
                if isinstance(ref_info, dict):
                    ref_mask_path = ref_info.get("mask_path")
                    if ref_mask_path:
                        ref_image_path = ref_mask_path.replace('Mask', 'CT')
            
            # 如果没有找到middle slice，使用中间位置作为middle slice
            if ref_image_path is None or not os.path.exists(ref_image_path):
                middle_idx = len(image_paths) // 2
                ref_image_path = image_paths[middle_idx]
                ref_mask_path = mask_paths[middle_idx]
                print(f"[WARN] Middle slice not found in annotation dict for {patient_name}, using slice {middle_idx}")
            
            patient_videos.append({
                'patient_name': patient_name,
                'video_id': video_id,
                'image_paths': image_paths,
                'mask_paths': mask_paths,
                'ref_image_path': ref_image_path,
                'ref_mask_path': ref_mask_path,
            })
            video_id += 1
        
        return patient_videos
    
    def get_video(self, idx: int):
        """
        获取一个video（病人的所有slice）
        
        Args:
            idx: video index (patient index)
        
        Returns:
            video: VOSVideo object
            segment_loader: MedSAM2VideoSegmentLoader object
        """
        if idx >= len(self.patient_videos):
            raise IndexError(f"Video index {idx} out of range for {len(self.patient_videos)} videos")
        
        video_info = self.patient_videos[idx]
        
        # 创建VOSFrame列表
        frames = []
        for frame_idx, image_path in enumerate(video_info['image_paths']):
            frames.append(VOSFrame(
                frame_idx=frame_idx,
                image_path=image_path,
                data=None,  # 将在transforms中加载
            ))
        
        # 创建VOSVideo对象
        video = VOSVideo(
            video_name=video_info['patient_name'],
            video_id=video_info['video_id'],
            frames=frames,
        )
        
        # 创建SegmentLoader
        segment_loader = MedSAM2VideoSegmentLoader(video_info['mask_paths'])
        
        return video, segment_loader
    
    def get_ref_image_mask(self, idx: int):
        """
        获取reference image和mask（middle slice）
        
        Args:
            idx: video index (patient index)
        
        Returns:
            ref_image_path: str
            ref_mask_path: str
        """
        if idx >= len(self.patient_videos):
            raise IndexError(f"Video index {idx} out of range for {len(self.patient_videos)} videos")
        
        video_info = self.patient_videos[idx]
        return video_info['ref_image_path'], video_info['ref_mask_path']
    
    def __len__(self):
        return len(self.patient_videos)
