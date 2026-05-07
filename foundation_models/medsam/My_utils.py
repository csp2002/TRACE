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
class Dataset_sli2vol(Dataset):   #use pseudo label as reference mask, the input only includes original image and reference/pseudo mask
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
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
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
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
        ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        if not os.path.exists(ref_mask_path):
            ref_mask_path = dict[folder_name]
        
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
            ref_mask = ref_mask / 255.0
            ref_x, ref_y = ref_mask.shape
            # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
            ref_mask_1024 = transform.resize(
                    ref_mask,
                    (1024,1024),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
            y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
       
        # add perturbation to bounding box coordinates
        H, W = ref_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_vol2flow(Dataset):   #use pseudo label as reference mask, the input only includes original image and reference/pseudo mask
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
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
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
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
        ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        if not os.path.exists(ref_mask_path):
            ref_mask_path = dict[folder_name]
        
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        if ref_mask.max() == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
            ref_mask = ref_mask / 255.0
            ref_x, ref_y = ref_mask.shape
            # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
            ref_mask_1024 = transform.resize(
                    ref_mask,
                    (1024,1024),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
            y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
       
        # add perturbation to bounding box coordinates
        H, W = ref_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_v6(Dataset):   #use largest slice as reference, the input includes original and reference image
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
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
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
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
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        # if not os.path.exists(ref_mask_path):
        ref_mask_path = dict[folder_name]
        
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        if ref_mask.max() == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        # if len(y_indices) == 0 or len(x_indices) == 0:
        #     ref_mask_path = dict[folder_name]
        #     ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        #     ref_mask = ref_mask / 255.0
        #     ref_x, ref_y = ref_mask.shape
        #     # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        #     ref_mask_1024 = transform.resize(
        #             ref_mask,
        #             (1024,1024),
        #             order=0,
        #             preserve_range=True,
        #             mode="constant",
        #             anti_aliasing=False,
        #         )
        #     y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
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
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_img_1024).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_v6_2(Dataset):   #use middle slice GT to extract box prompt, middle slice as reference
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
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
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
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
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
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
        # print('case_num:', case_num)

        # info = self.ref_map.get(case_num, None)
        # print("[DEBUG] ref_map.get(case_num) =", repr(info), "type =", type(info))
        # ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        info = self.ref_map.get(case_num)
        if isinstance(info, dict):
            # print('info:', info)
            # 优先使用 JSON 里记录的绝对路径
            cand_ct  = info.get("ct_path")
            cand_msk = info.get("mask_path")  # 可能为 None
            if cand_ct and os.path.exists(cand_ct):
                ref_image_path = cand_ct
            else:
                print(f"[WARN] middle-slice not found: {cand_ct}. Will fall back to current slice.")
            # ref_mask_path 允许为 None（若该病人无任何 mask），这种情况回退到当前样本的 mask
            if cand_msk and os.path.exists(cand_msk):
                ref_mask_path = cand_msk
            else:
                print(f"[WARN] middle-slice not found: {cand_msk}. Will fall back to current slice.")

        # dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # folder_name = os.path.dirname(mask_path)
        # # if not os.path.exists(ref_mask_path):
        # ref_mask_path = dict[folder_name]
        # print('ref_mask_path:', ref_mask_path,"mask_path:", mask_path)
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        if ref_mask.max() == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        # if len(y_indices) == 0 or len(x_indices) == 0:
        #     ref_mask_path = dict[folder_name]
        #     ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        #     ref_mask = ref_mask / 255.0
        #     ref_x, ref_y = ref_mask.shape
        #     # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        #     ref_mask_1024 = transform.resize(
        #             ref_mask,
        #             (1024,1024),
        #             order=0,
        #             preserve_range=True,
        #             mode="constant",
        #             anti_aliasing=False,
        #         )
        #     y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
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
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_img_1024).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_v8(Dataset):   #randomly choose another slice in the same volume as reference, the input includes original and reference image, the box prompt are extracted by vol2flow prediction of the original image
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
        # mask_dir = os.path.join(self.root_dir, "Mask")
        
        for patient_folder in os.listdir(ct_dir):
            patient_ct_folder = os.path.join(ct_dir, patient_folder)
            # patient_mask_folder = os.path.join(mask_dir, patient_folder)
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
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
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
        ref_box_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        if not os.path.exists(ref_box_path):
            ref_box_path = dict[folder_name]   


        # 当前slice的信息
        patient_ct_folder = os.path.dirname(image_path)

        # 随机选择同病人下的另一个 slice 作为 reference
        available_slices = [
            f for f in os.listdir(patient_ct_folder)
            if f != slice_num 
        ]
        # print('image_path:', image_path)
        # print('available_slices:', available_slices)
        # raise Exception
        ref_filename = random.choice(available_slices)

        ref_image_path = os.path.join(patient_ct_folder, ref_filename)
        ref_mask_path = ref_image_path.replace("CT", "Mask")

        
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # if ref_mask.max() == 0:
        #     ref_mask_path = dict[folder_name]
        #     ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        
        box_mask = np.array(Image.open(ref_box_path).convert("L"), dtype=np.float32)
        box_mask = box_mask / 255.0
        box_mask_1024 = transform.resize(
                box_mask,
                (1024,1024),
                order = 0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
        )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(box_mask_1024 > 0)
        # if len(y_indices) == 0 or len(x_indices) == 0:
        #     ref_mask_path = dict[folder_name]
        #     ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        #     ref_mask = ref_mask / 255.0
        #     ref_x, ref_y = ref_mask.shape
        #     # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        #     ref_mask_1024 = transform.resize(
        #             ref_mask,
        #             (1024,1024),
        #             order=0,
        #             preserve_range=True,
        #             mode="constant",
        #             anti_aliasing=False,
        #         )
        #     y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
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
        H, W = box_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(ref_img_1024).float(),
            torch.tensor(ref_mask_1024[None, :, :]).float(),
            img_name,
        )
class Dataset_v9(Dataset):   # use NEIGHBOR slice GT to extract box prompt, neighbor slice as reference
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

        # ========== [CHANGED] 从 neighbor JSON 取 ref ==========
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
class refinement(nn.Module):
    def __init__(self):
        super(refinement, self).__init__()

        # Mask特征扩展到32通道
        self.ori_feat_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pseudo_feat_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # multi-scale特征对齐与融合
        self.f0_conv = nn.Conv2d(768, 128, kernel_size=3, padding=1)
        self.f1_conv = nn.Conv2d(768, 128, kernel_size=3, padding=1)
        self.f2_conv = nn.Conv2d(768, 128, kernel_size=3, padding=1)
        self.fuse_multi_scale = nn.Conv2d(384, 128, kernel_size=3, padding=1)

        # 动态gate生成网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终refinement预测层
        self.refine_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, original_mask, pseudo_mask, f0, f1, f2):
        bs, _, H, W = original_mask.size()
        f0 = f0.permute(0, 3, 1, 2)  # (bs,64,64,768) to (bs,768,64,64)
        f1 = f1.permute(0, 3, 1, 2)
        f2 = f2.permute(0, 3, 1, 2)
        # print('original_mask:', original_mask.shape, original_mask.max(), original_mask.min())
        # print('pseudo_mask:', pseudo_mask.shape, pseudo_mask.max(), pseudo_mask.min())

        # mask特征初始提取
        ori_feat = F.relu(self.ori_feat_conv(original_mask))          # (bs,32,H,W)
        pseudo_feat = F.relu(self.pseudo_feat_conv(pseudo_mask))      # (bs,32,H,W)

        # Step 2: multi-scale特征对齐与融合
        f0 = self.f0_conv(f0) # (bs,128,64,64)
        f1 = self.f1_conv(f1)                                                                  
        f2 = self.f2_conv(f2)
        multi_scale_feat = torch.cat([f0, f1, f2], dim=1)                                # (bs,384,64,64)
        multi_scale_feat = F.relu(self.fuse_multi_scale(multi_scale_feat))                          # (bs,128,64,64)
        multi_scale_feat = F.interpolate(multi_scale_feat, size=(H,W), mode='bilinear', align_corners=False) # (bs,128,H,W)

        # Step 3: 动态gate生成
        gate = self.gate_conv(multi_scale_feat)  # (bs,1,H,W), pseudo重要性权重

        # Step 4: 动态融合两个mask特征
        fused_feat = gate * pseudo_feat + (1 - gate) * ori_feat       # (bs,32,H,W)

        # Step 5: 残差预测最终mask
        delta_mask = self.refine_conv(fused_feat)                     # (bs,1,H,W)
        refined_mask = original_mask + delta_mask                     # (bs,1,H,W)

        return refined_mask
class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        refinement
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.refinement = refinement
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, ref_mask):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        f0, f1, f2 = features  # bs,64,64,768
        # print('f0:', f0.shape, 'f1:', f1.shape, 'f2:', f2.shape)
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
        final_mask = self.refinement(ori_res_masks, ref_mask, f0, f1, f2)
        return final_mask, ori_res_masks
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
        self.final = nn.Conv2d(64, 1, kernel_size=1)

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

        out = self.final(d1)              # (bs, 1, 224, 224)
        return out
class MedSAM_v3o1(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        refinement
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.refinement = refinement
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, ref_mask):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        f0, f1, f2 = features  # bs,64,64,768
        # print('f0:', f0.shape, 'f1:', f1.shape, 'f2:', f2.shape)
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
        new_image = torch.cat([image[:, 0:1, :, :], ref_mask, ori_mask], dim=1)  # (bs, 3, 1024, 1024)
        # print('new_image:', new_image.shape, new_image.max(), new_image.min())
        # raise Exception
        final_mask = self.refinement(new_image)
        # print('final_mask:', final_mask.shape, final_mask.max(), final_mask.min())
        # raise Exception
        return final_mask, ori_res_masks
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
        # print('original_mask:', original_mask.shape, original_mask.max(), original_mask.min())
        # print('pseudo_mask:', pseudo_mask.shape, pseudo_mask.max(), pseudo_mask.min())
        foreground_mask = torch.max(original_mask, pseudo_mask)  # (bs,1,224,224)
        # print('foreground_mask:', foreground_mask.shape, foreground_mask.max(), foreground_mask.min())
        mask_bias = self.mask_proj(foreground_mask)  # (bs,1,224,224)

        out = self.bottleneck(features)
        out = out + mask_bias
        # print('out:', out.shape, out.max(), out.min())
        modulation_map = torch.sigmoid(out)
        # print('modulation_map:', modulation_map.shape, modulation_map.max(), modulation_map.min())

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
        self.final = nn.Conv2d(64, 1, kernel_size=1)

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

        out = self.final(d1)              # (bs, 1, 224, 224)
        return out
class MedSAM_v3o2(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        refinement
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.refinement = refinement
        self.modulation_net = ModulationNet()
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, ref_mask):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        f0, f1, f2 = features  # bs,64,64,768
        # print('f0:', f0.shape, 'f1:', f1.shape, 'f2:', f2.shape)
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
        # print('ori_res_masks:', ori_res_masks.shape, ori_res_masks.max(), ori_res_masks.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # print('image:', image.shape, image.max(), image.min())
        ori_mask = torch.sigmoid(ori_res_masks)  # (bs, 1, 1024, 1024)
        # raise Exception
        modulation_map, foreground_mask = self.modulation_net(ori_mask, ref_mask)

        enhanced_image = image[:,0:1,:,:] * modulation_map  # (bs,1,224,224)

        
        # final_pred = self.refinement_module(enhanced_image)  #bs,2,224,224

        # new_image = torch.cat([image[:, 0:1, :, :], ref_mask, ori_res_masks], dim=1)  # (bs, 3, 1024, 1024)
        # print('new_image:', new_image.shape, new_image.max(), new_image.min())
        # raise Exception
        final_mask = self.refinement(enhanced_image)
        # print('final_mask:', final_mask.shape, final_mask.max(), final_mask.min())
        # raise Exception
        return final_mask, ori_res_masks, modulation_map, ref_mask
from os.path import join as pjoin
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

    conf_112 = F.interpolate(conf, size=(512,512), mode='bilinear', align_corners=True)
    conf_56  = F.interpolate(conf, size=(256,256),   mode='bilinear', align_corners=True)
    conf_28  = F.interpolate(conf, size=(128,128),   mode='bilinear', align_corners=True)
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
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)   #csp change 2 to 1 since medsam use 1 channel as final mask rather than 2 classes

    def forward(self, t1, t2, t3, x_tar, r1, r2, r3, x_ref=None, confs=None):
        assert confs is not None and len(confs) == 3, "need confs=(conf_112, conf_56, conf_28)"
        conf_112, conf_56, conf_28 = confs
        # print('conf_112:', conf_112.shape, conf_112.max(), conf_112.min())  #bs,1,112,112
        # print('conf_56:', conf_56.shape, conf_56.max(), conf_56.min())  #bs,1,56,56
        # print('conf_28:', conf_28.shape, conf_28.max(), conf_28.min())  #bs,1,28,28
        # print('x_tar:', x_tar.shape, x_tar.max(), x_tar.min())  #bs,1024,64,64, previously 14,14
        # print('x_ref:', x_ref.shape, x_ref.max(), x_ref.min())  #bs,1024,64,64
        # print('t1:', t1.shape, t1.max(), t1.min()) #bs,64,512,512
        # print('t2:', t2.shape, t2.max(), t2.min()) #bs,256,256,256
        # print('t3:', t3.shape, t3.max(), t3.min())  #bs,512,128,128
        # print('r1:', r1.shape, r1.max(), r1.min())
        # print('r2:', r2.shape, r2.max(), r2.min())
        # print('r3:', r3.shape, r3.max(), r3.min())

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
        # print('f0:', f0.shape, 'f1:', f1.shape, 'f2:', f2.shape)
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
        # print('ori_res_masks:', ori_res_masks.shape, ori_res_masks.max(), ori_res_masks.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # print('image:', image.shape, image.max(), image.min())
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
        # print('ori_res_masks:', ori_res_masks.shape, ori_res_masks.max(), ori_res_masks.min())
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # print('image:', image.shape, image.max(), image.min())
        ori_mask = torch.sigmoid(ori_res_masks)  # (bs, 1, 1024, 1024)
        ref_mask = torch.sigmoid(ori_res_ref_masks)
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
        # print('ori_mask:', ori_mask.shape, ori_mask.max(), ori_mask.min())
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        # print('image:', image.shape, image.max(), image.min())
        # print('ref_gt:', ref_gt.shape, ref_gt.max(), ref_gt.min())
        # new_image = torch.cat([image[:,0:1,:,:], ori_mask, ref_image[:,0:1,:,:], ref_mask, ref_gt], dim=1)
        target_image = torch.cat([image[:,0:1,:,:], ori_mask], dim=1)  #bs,2,224,224
        ref_3ch = torch.cat([ref_image[:,0:1,:,:], ref_mask, ref_gt], dim=1)  #bs,3,224,224
        preds_all = []  
        final_pred = self.refinement(target_image, ref_3ch)  #bs,2,224,224
        preds_all.append(final_pred)

        for it in range(1, max(1, self.refine_iters) ):
        # 把上一轮的输出变成下一轮的 target prediction
            # next_mask = torch.softmax(final_pred, dim=1)[:, 1:2] 
            next_mask = torch.sigmoid(final_pred)

            if self.detach_between_iters:
                next_mask = next_mask.detach()

            target_2ch = torch.cat([image[:,0:1,:,:], next_mask], dim=1)      # (B,2,H,W)
            # reference 一般保持不变；如果你希望同步更新 ref_pred，也可以把 ref_mask 改成 logits_ref 的迭代版
            # 这里保持 ref_3ch 不变：
            # ref_3ch = torch.cat([ref_image_raw, ref_mask, ref_gt], dim=1)

            final_pred = self.refinement(target_2ch, ref_3ch)
            preds_all.append(final_pred)
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return {"final": final_pred, "iters": preds_all}
