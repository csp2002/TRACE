# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
test the model trained with reference mask
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
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
from My_utils import *
import vit_seg_configs as configs

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6




# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
parser.add_argument(
    "--task_name", type=str, default=None,
    help="Override the auto-generated task_name. If omitted, derived from --use_trace + --ref "
         "as 'with_TRACE_<ref>' or 'finetune_<ref>'.",
)
parser.add_argument(
    "--use_trace", action="store_true", default=False,
    help="Load a MedSAM + TRACE checkpoint (model2). Without this flag, load baseline MedSAM (model1).",
)
parser.add_argument("--model_type", type=str, default="vit_b")

parser.add_argument(
    "--checkpoint", type=str, default="medsam_vit_b.pth"
)
parser.add_argument("--ckpt_path", type=str, default="")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument('--ref', type=str, default='neighbor', choices=['largest','neighbor','middle'], help='choose the reference slice')
args = parser.parse_args()

# Auto-construct task_name (mirror train_one_gpu.py).
if args.task_name is None:
    args.task_name = f"with_TRACE_{args.ref}" if args.use_trace else f"finetune_{args.ref}"



args.task_name = args.task_name + "-" + args.data
device = torch.device(args.device)
# %% set up model


    



def compute_metrics(pred, target, smooth=1e-6):
    # Cast the float prediction to a binary mask
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    # Flatten
    pred = pred.flatten()
    target = target.flatten()
    
    # Intersection, total, union
    intersection = np.sum(pred * target)
    total = np.sum(pred) + np.sum(target)
    union = total - intersection

    # Compute IoU and Dice
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (total + smooth)

    return iou, dice



def main():

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    if args.use_trace:
        config_small = configs.get_r18_s16_config()
        config_small.n_classes = 2
        config_small.n_skip = 3
        config_small.patches.grid = (int(1024 / 16), int(1024 / 16))
        medsam_model = MedSAM_with_TRACE(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            refinement=TRACE(config_small),
        )
        print('Using MedSAM + TRACE')
    else:
        medsam_model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            refinement=refinement(),
        )
        print('Using baseline MedSAM')

    
    ckpt = torch.load(args.ckpt_path, map_location=device)

    medsam_model.load_state_dict(
        ckpt["model"],
        strict = True,
    )
    print("load trained model from: ", args.ckpt_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    

    # Anchor 2D_data/ to the repo root (relative to this file) so the script
    # works regardless of cwd.
    _data_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '2D_data'
    )
    dataset_config = {
        'kits':     {'root_path': os.path.join(_data_root, 'kits'),     'num_classes': 2},
        'pancreas': {'root_path': os.path.join(_data_root, 'pancreas'), 'num_classes': 2},
        'lits':     {'root_path': os.path.join(_data_root, 'lits'),     'num_classes': 2},
        'colon':    {'root_path': os.path.join(_data_root, 'colon'),    'num_classes': 2},
    }
    root_path = dataset_config[args.data]['root_path']
    if args.ref == 'middle':
        test_dataset = Dataset_middle(base_dir=root_path, mode='test')
        print('Using middle-slice reference dataset')
    elif args.ref == 'neighbor':
        test_dataset = Dataset_neighbor(base_dir=root_path, mode='test')
        print('Using neighbor-slice reference dataset')
    else:
        raise ValueError(
            f"Unsupported --ref value: {args.ref!r}; expected 'middle' or 'neighbor'"
        )

    print("Number of testing samples: ", len(test_dataset))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
    )
    
    total_iou = 0
    total_dice = 0
    for step, (image, gt2D, boxes,  ref_img, ref_gt, _)  in enumerate(tqdm(test_dataloader)):
        
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D, ref_img, ref_gt = image.to(device), gt2D.to(device), ref_img.to(device), ref_gt.to(device)
        if args.use_trace:
            outputs = medsam_model(image, boxes_np, ref_img, ref_gt)
            medsam_pred = outputs['final']
        else:
            # Baseline MedSAM still consumes the reference mask and returns
            # (final_mask, ori_res_masks); pass ref_gt and unpack the tuple.
            medsam_pred, _ = medsam_model(image, boxes_np, ref_gt)
        medsam_pred = torch.sigmoid(medsam_pred)
        final_pred = medsam_pred > 0.5
        iou, dice = compute_metrics(
            final_pred.detach().cpu().numpy()[0, 0], gt2D.detach().cpu().numpy()[0, 0]
        )
        total_iou += iou
        total_dice += dice
    average_iou = total_iou / len(test_dataloader)
    average_dice = total_dice / len(test_dataloader)
    print("Average IoU: ", average_iou)
    print("Average Dice: ", average_dice)
        

    
    


if __name__ == "__main__":
    main()
