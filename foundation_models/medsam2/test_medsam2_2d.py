# -*- coding: utf-8 -*-
"""
MedSAM2 2D test script.
Evaluate the performance of the trained model.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import cv2
from skimage import io

# Add the MedSAM2 directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.dataset.medsam2_2d_dataset import MedSAM2_2D_Dataset
from training.model.medsam2_with_trace import MedSAM2_with_TRACE
from training.model.trace import TRACE
from torch.utils.data import DataLoader


def compute_metrics(pred, target, smooth=1e-6):
    """
    Compute IoU and Dice scores.
    
    Args:
        pred: predicted binary mask, shape (H, W) or (1, H, W)
        target: ground-truth binary mask, shape (H, W) or (1, H, W)
        smooth: smoothing constant
    
    Returns:
        iou: IoU score
        dice: Dice score
    """
    # Ensure input is a 1-D array
    if len(pred.shape) > 2:
        pred = pred.flatten()
    else:
        pred = pred.flatten()
    
    if len(target.shape) > 2:
        target = target.flatten()
    else:
        target = target.flatten()
    
    # Ensure mask is binary
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    # Compute intersection
    intersection = np.sum(pred * target)
    
    # Compute total
    total = np.sum(pred) + np.sum(target)
    
    # Compute union
    union = total - intersection
    
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Compute Dice coefficient
    dice = (2. * intersection + smooth) / (total + smooth)
    
    return iou, dice


def get_image_mask_paths(root_dir):
    """Collect all image and mask paths."""
    image_paths = []
    mask_paths = []
    ct_dir = os.path.join(root_dir, "CT")
    
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


@torch.no_grad()
def medsam2_inference(image_predictor, img_rgb, box):
    """
    Run MedSAM2 inference.
    
    Args:
        image_predictor: SAM2ImagePredictor instance
        img_rgb: RGB image in HWC format with pixel values in [0, 255]
        box: bounding box [x0, y0, x1, y1]
    
    Returns:
        medsam_seg: predicted binary mask (H, W)
    """
    # Set the image
    image_predictor.set_image(img_rgb)
    
    # Predict the mask
    box_array = np.array(box).reshape(1, 4) if len(box.shape) == 1 else box
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_array,
        multimask_output=False,
        return_logits=False,
        normalize_coords=True,
    )
    
    # Take the first mask (multimask_output=False)
    medsam_seg = (masks[0] > 0.0).astype(np.uint8)
    return medsam_seg


def eval_model_trace(model, dataloader, device, image_size=512):
    """Evaluate the MedSAM2 + TRACE wrapper model via the standard
    MedSAM2_2D_Dataset DataLoader. Returns (avg_iou, avg_dice)."""
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            bboxes = batch['bbox'].cpu().numpy()
            if 'ref_image' not in batch or 'ref_gt' not in batch:
                raise ValueError("Dataset must return 'ref_image' and 'ref_gt' for the TRACE variant")
            ref_images = batch['ref_image'].to(device)
            ref_gts = batch['ref_gt'].to(device)
            outputs = model(images, bboxes, ref_images, ref_gts, image_size=image_size)
            pred_masks = torch.sigmoid(outputs['final'])
            pred_np = pred_masks.cpu().numpy()
            target_np = masks.cpu().numpy()
            for i in range(pred_np.shape[0]):
                iou, dice = compute_metrics(pred_np[i, 0], target_np[i, 0])
                total_iou += iou
                total_dice += dice
                num_samples += 1
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0
    return avg_iou, avg_dice


def eval_model(model, image_paths, mask_paths, annotation_path, device, image_size=512):
    """
    Evaluate model performance.
    
    Args:
        model: MedSAM2 model
        image_paths: list of image paths
        mask_paths: list of mask paths
        annotation_path: path to the annotation dict
        device: torch device
        image_size: model input size
    
    Returns:
        avg_iou: mean IoU
        avg_dice: mean Dice
    """
    model.eval()
    
    # Load the annotation dict
    ref_map = {}
    if annotation_path and os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            ref_map = json.load(f)
    
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    image_predictor = SAM2ImagePredictor(model)
    
    for i in tqdm(range(len(image_paths)), desc="Evaluating"):
        image_path = image_paths[i]
        mask_path = mask_paths[i]
        
        # Read the image
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
        
        # Make sure the image is uint8 in [0, 255]
        if img_3c.dtype != np.uint8:
            img_3c = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            ) * 255.0
            img_3c = img_3c.astype(np.uint8)
        
        H, W, _ = img_3c.shape
        
        # Read the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Cannot read mask {mask_path}, skipping...")
            continue
        
        # Resolve the reference-mask path
        case_num = os.path.basename(os.path.dirname(mask_path))
        ref_mask_path = None
        
        if case_num in ref_map:
            info = ref_map[case_num]
            if isinstance(info, dict):
                ref_mask_path = info.get("mask_path")
                if ref_mask_path and not os.path.exists(ref_mask_path):
                    ref_mask_path = None
        
        # Fall back to the current mask if the reference is missing
        if ref_mask_path is None or not os.path.exists(ref_mask_path):
            ref_mask_path = mask_path
        
        # Extract the bounding box from the reference mask
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            print(f"Warning: Cannot read ref_mask {ref_mask_path}, using current mask...")
            ref_mask = mask
        
        # Resize ref_mask to image_size
        ref_mask_resized = cv2.resize(ref_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        y_indices, x_indices = np.where(ref_mask_resized > 127)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Fall back to the current mask if the reference mask is empty
            mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            y_indices, x_indices = np.where(mask_resized > 127)
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Warning: Empty mask for {mask_path}, skipping...")
                continue
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Map bbox back from image_size scale to the original image scale
        x_min = int(x_min * W / image_size)
        x_max = int(x_max * W / image_size)
        y_min = int(y_min * H / image_size)
        y_max = int(y_max * H / image_size)
        
        box_prompt = np.array([x_min, y_min, x_max, y_max])
        
        # Run MedSAM2 inference
        medsam_seg = medsam2_inference(image_predictor, img_3c, box_prompt)
        
        # Resize the predicted mask to the original mask size
        if medsam_seg.shape != mask.shape:
            medsam_seg = cv2.resize(medsam_seg, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize the mask
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Compute metrics
        iou, dice = compute_metrics(medsam_seg, mask_binary)
        
        total_iou += iou
        total_dice += dice
        num_samples += 1
    
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0
    
    return avg_iou, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Test MedSAM2 on 2D medical images")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["kits", "pancreas", "lits", "colon"],
        help="Dataset name"
    )
    parser.add_argument(
        "--ref_type",
        type=str,
        default="middle",
        choices=["middle", "neighbor"],
        help="Reference slice type: middle or neighbor"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path (relative to MedSAM2 root or absolute path)"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="Model config file (relative to MedSAM2 root)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for testing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (used by the TRACE eval DataLoader)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Data-loading workers (used by the TRACE eval DataLoader)"
    )
    parser.add_argument(
        "--use_trace",
        action="store_true",
        help="Evaluate MedSAM2 + TRACE; default is baseline MedSAM2."
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="Path to the trained MedSAM2 + TRACE checkpoint (only used with --use_trace)."
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=3,
        help="Refinement iterations for TRACE (default: 3, only used with --use_trace)."
    )

    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Build the model
    medsam2_root = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve the checkpoint path
    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(medsam2_root, args.checkpoint)
    
    print(f"Loading model from {checkpoint_path}")
    print(f"Using config: {args.cfg}")
    
    sam2_model = build_sam2(
        config_file=args.cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval"
    )

    if args.use_trace:
        if not args.model_ckpt:
            raise ValueError("--use_trace requires --model_ckpt to point at a trained MedSAM2 + TRACE checkpoint.")
        class _RefinementConfig:
            classifier = None
        refinement_module = TRACE(_RefinementConfig(), img_size=args.image_size, pretrained=True)
        model = MedSAM2_with_TRACE(
            sam2_model=sam2_model,
            refinement=refinement_module,
            refine_iters=args.refine_iters,
            detach_between_iters=True,
        )
        print(f"Loading trained MedSAM2 + TRACE checkpoint from {args.model_ckpt}")
        trace_ckpt = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(trace_ckpt.get("model", trace_ckpt), strict=False)
        model = model.to(device)
        model.eval()
        dataset_dir = os.path.join("./2D_data", args.data)
        test_dataset = MedSAM2_2D_Dataset(
            base_dir=dataset_dir,
            mode="test",
            ref_type=args.ref_type,
            image_size=args.image_size,
            bbox_shift=0,
            use_augmentation=False,
        )
        print(f"Test samples: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"\n{'='*50}\nEvaluating MedSAM2 + TRACE on {args.data} ({args.ref_type})\n{'='*50}")
        avg_iou, avg_dice = eval_model_trace(model, test_dataloader, device, image_size=args.image_size)
        num_samples_for_log = len(test_dataset)
    else:
        # Baseline path: optionally overlay a finetune checkpoint onto build_sam2's weights.
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model" in checkpoint:
                sam2_model.load_state_dict(checkpoint["model"], strict=False)
                print("Loaded model weights from training checkpoint")
        except Exception as e:
            print(f"Warning: Could not load checkpoint as training checkpoint: {e}")
            print("Using checkpoint as pretrained model")

        dataset_dir = os.path.join("./2D_data", args.data, "test")
        image_paths, mask_paths = get_image_mask_paths(dataset_dir)
        print(f"Test samples: {len(image_paths)}")

        if args.ref_type == 'middle':
            annotation_path = os.path.join(dataset_dir, "annotation_dict_middle.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotation_dict_neighbor.json")

        print(f"\n{'='*50}\nEvaluating baseline MedSAM2 on {args.data} ({args.ref_type})\n{'='*50}")
        avg_iou, avg_dice = eval_model(
            sam2_model,
            image_paths,
            mask_paths,
            annotation_path,
            device,
            image_size=args.image_size,
        )
        num_samples_for_log = len(image_paths)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"{'='*50}\n")
    
    # Save the results
    results = {
        "dataset": args.data,
        "ref_type": args.ref_type,
        "use_trace": bool(args.use_trace),
        "checkpoint": checkpoint_path,
        "model_ckpt": args.model_ckpt if args.use_trace else None,
        "avg_iou": float(avg_iou),
        "avg_dice": float(avg_dice),
        "num_samples": int(num_samples_for_log),
    }

    results_dir = os.path.join(medsam2_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "with_TRACE" if args.use_trace else "baseline"
    results_file = os.path.join(results_dir, f"results_{args.data}_{args.ref_type}_{suffix}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
