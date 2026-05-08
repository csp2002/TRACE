import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import glob, sys
import pydicom
import nibabel as nib
import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import imageio

def _resolve_split_path(args):
    """Resolve where to load split.pkl from. Priority order:

      1. --split_pkl, if explicitly provided.
      2. <data_prefix>/split.pkl, if it exists (legacy / 3DSAM-adapter style
         where the user keeps the split file alongside the raw NIfTI volumes).
      3. The bundled split file shipped at
         data_preparation/splits/<dataset>/split.pkl. These are the canonical
         splits used by the paper; they originate from the 3DSAM-adapter
         repository's `datafile/<dataset>/split.pkl` for pancreas, LiTS and
         colon, and from the official KiTS23 release for kits.
    """
    if args.split_pkl:
        return args.split_pkl
    candidate = os.path.join(args.data_prefix, 'split.pkl')
    if os.path.isfile(candidate):
        return candidate
    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'splits', args.data, 'split.pkl',
    )
    if os.path.isfile(bundled):
        return bundled
    raise FileNotFoundError(
        f"Could not find split.pkl. Tried --split_pkl ({args.split_pkl!r}), "
        f"{candidate!r}, and {bundled!r}."
    )


def data_extraction(args):
    data_prefix = args.data_prefix
    split_path = _resolve_split_path(args)
    print(f"[split] using {split_path}")

    with open(split_path, "rb") as f:
        d_train = pickle.load(f)[0]['train']

    with open(split_path, "rb") as f2:
        d_test = pickle.load(f2)[0]['test']

    img_files_train = [os.path.join(data_prefix, d_train[i][0].strip("/")) for i in list(d_train.keys())]  #list of nii.gz
    seg_files_train = [os.path.join(data_prefix, d_train[i][1].strip("/")) for i in list(d_train.keys())]
    img_files_test = [os.path.join(data_prefix, d_test[i][0].strip("/")) for i in list(d_test.keys())]  #list of nii.gz
    seg_files_test = [os.path.join(data_prefix, d_test[i][1].strip("/")) for i in list(d_test.keys())]
    # print('img_files_train:', img_files_train)
    # print('img_files_test:', img_files_test)
    # print('seg_files_test:', seg_files_test)
    # print('seg_files_train:', seg_files_train)
    for i in range(len(img_files_train)):
        img_path = img_files_train[i]
        seg_path = seg_files_train[i]
        img = nib.load(img_path).get_fdata().astype(np.float32).transpose(args.spatial_index)
        seg = nib.load(seg_path).get_fdata().astype(np.float32).transpose(args.spatial_index)
        # Sanity-check that the image and segmentation shapes match; warn if they don't
        assert img.shape == seg.shape, f"shape mismatch: {img.shape} vs {seg.shape}"
        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0
        if args.target_class is not None:
            seg = (seg == args.target_class).astype(np.float32)
        # Clip by intensity_range, then min-max normalize the image into [0, 1]
        img = np.clip(img, args.intensity_range[0], args.intensity_range[1])
        img = (img - args.intensity_range[0]) / (args.intensity_range[1] - args.intensity_range[0])
        if args.data == 'colon':  # case_name = filename without the .nii.gz suffix
            case_name = os.path.basename(img_path).split('.')[0]
        else:  # case_name = the second-to-last directory name in the path
            case_name = os.path.basename(os.path.dirname(img_path))
        image_case_folder = os.path.join(args.save_folder, 'train', 'CT', case_name)
        seg_case_folder = os.path.join(args.save_folder, 'train', 'Mask', case_name)
        os.makedirs(image_case_folder, exist_ok=True)
        os.makedirs(seg_case_folder, exist_ok=True)
        for j in range(img.shape[0]):
            img_slice = img[j]
            seg_slice = seg[j]
            # Save only slices that contain foreground voxels as PNG; filename is the slice index, zero-padded to 3 digits
            if seg_slice.sum() != 0:
                plt.imsave(os.path.join(image_case_folder, f'{j:03d}.png'), img_slice, cmap='gray')
                plt.imsave(os.path.join(seg_case_folder, f'{j:03d}.png'), seg_slice, cmap='gray')
        print(f"case {case_name} done!")
        
    for i in range(len(img_files_test)):
        img_path = img_files_test[i]
        seg_path = seg_files_test[i]
        img = nib.load(img_path).get_fdata().astype(np.float32).transpose(args.spatial_index)
        seg = nib.load(seg_path).get_fdata().astype(np.float32).transpose(args.spatial_index)
        # Sanity-check that the image and segmentation shapes match; warn if they don't
        assert img.shape == seg.shape, f"shape mismatch: {img.shape} vs {seg.shape}"
        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0
        if args.target_class is not None:
            seg = (seg == args.target_class).astype(np.float32)
        # Clip by intensity_range, then min-max normalize the image into [0, 1]
        img = np.clip(img, args.intensity_range[0], args.intensity_range[1])
        img = (img - args.intensity_range[0]) / (args.intensity_range[1] - args.intensity_range[0])
        # print('img:', img.shape, img.max(), img.min())
        if args.data == 'colon':
            case_name = os.path.basename(img_path).split('.')[0]
        else:
            case_name = os.path.basename(os.path.dirname(img_path))
        image_case_folder = os.path.join(args.save_folder, 'test', 'CT', case_name)
        seg_case_folder = os.path.join(args.save_folder, 'test', 'Mask', case_name)
        os.makedirs(image_case_folder, exist_ok=True)
        os.makedirs(seg_case_folder, exist_ok=True)
        cnt = 0
        for j in range(img.shape[0]):
            img_slice = img[j]
            seg_slice = seg[j]
            # Save only slices that contain foreground voxels as PNG; filename is the slice index, zero-padded to 3 digits
            if seg_slice.sum() != 0:
                cnt += 1
                plt.imsave(os.path.join(image_case_folder, f'{j:03d}.png'), img_slice, cmap='gray')
                plt.imsave(os.path.join(seg_case_folder, f'{j:03d}.png'), seg_slice, cmap='gray')
                # imageio.imwrite(os.path.join(image_case_folder, f"{j:05d}.png"), img_slice.astype(np.uint8)*255)
                # print('img_slice:', img_slice.shape, img_slice.max(), img_slice.min())
                # imageio.imwrite(os.path.join(seg_case_folder, f"{j:05d}.png"), seg_slice.astype(np.uint8)*255)
        print(f"case {case_name} done!")
        # print(f"case {case_name} has {cnt} slices with foreground!")
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument(
        "--data_prefix",   #folder to save checkpoints
        default='',
        type=str,
    )
    parser.add_argument(
        "--save_folder",   #folder to save datasets
        default="",
        type=str,
    )
    parser.add_argument(
        "--split_pkl",
        default="",
        type=str,
        help=(
            "Optional path to split.pkl. If omitted, look first in "
            "<data_prefix>/split.pkl and otherwise fall back to the bundled "
            "data_preparation/splits/<dataset>/split.pkl shipped with this "
            "repository."
        ),
    )
    parser.add_argument(
        "--intensity_range",
        default=(-45,227),
        type=tuple,
    )
    parser.add_argument(
        "--global_mean",
        default=60.34786,
        type=float,
    )
    parser.add_argument(
        "--global_std",
        default=50.526666,
        type=float,
    )
    parser.add_argument(   # index used to convert to DHW
        "--spatial_index",
        default=[0,1,2],
        type=list,
    )
    parser.add_argument(
        "--target_class",
        default=2,
        type=int,
    )
    args = parser.parse_args()
    # Default save folder if user didn't pass one explicitly.
    if not args.save_folder:
        args.save_folder = os.path.join('./2D_data', args.data)
    # Per-dataset preprocessing parameters: intensity clip range, foreground
    # mean/std, axis order, target class id, and the canonical raw-data
    # subdirectory name. Intensity stats are the 0.5 / 99.5 percentiles of
    # foreground voxels reported in the paper supplement and are baked in so
    # the user does not need to recompute them. The data_prefix default is
    # only used when the user does not pass --data_prefix on the CLI.
    if args.data == 'kits':
        args.intensity_range = (-52, 269)
        args.global_mean = 60.514008
        args.global_std = 55.836348
        args.spatial_index = [0, 1, 2]
        args.target_class = 2
        if not args.data_prefix:
            args.data_prefix = 'kits23'
    elif args.data == 'pancreas':
        args.intensity_range = (-42, 195)
        args.global_mean = 71.919696
        args.global_std = 57.146912
        args.spatial_index = [2, 1, 0]
        args.target_class = 2
        if not args.data_prefix:
            args.data_prefix = 'Task03_Pancreas'
    elif args.data == 'lits':
        args.intensity_range = (-46, 164)
        args.global_mean = 60.456020
        args.global_std = 40.840413
        args.spatial_index = [2, 1, 0]
        args.target_class = 2
        if not args.data_prefix:
            args.data_prefix = 'Task01_LITS17'
    elif args.data == 'colon':
        args.intensity_range = (-30, 166)
        args.global_mean = 64.836747
        args.global_std = 32.622727
        args.spatial_index = [2, 1, 0]
        args.target_class = 1
        if not args.data_prefix:
            args.data_prefix = 'Task10_Colon'
    data_extraction(args)

