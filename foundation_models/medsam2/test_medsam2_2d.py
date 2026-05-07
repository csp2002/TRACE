# -*- coding: utf-8 -*-
"""
MedSAM2 2D测试脚本
用于评估训练后的模型性能
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

# 添加MedSAM2目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.dataset.medsam2_2d_dataset import MedSAM2_2D_Dataset
from torch.utils.data import DataLoader


def compute_metrics(pred, target, smooth=1e-6):
    """
    计算IoU和Dice系数
    
    Args:
        pred: 预测mask，binary mask (H, W) 或 (1, H, W)
        target: 真实mask，binary mask (H, W) 或 (1, H, W)
        smooth: 平滑参数
    
    Returns:
        iou: IoU值
        dice: Dice系数
    """
    # 确保输入为一维数组
    if len(pred.shape) > 2:
        pred = pred.flatten()
    else:
        pred = pred.flatten()
    
    if len(target.shape) > 2:
        target = target.flatten()
    else:
        target = target.flatten()
    
    # 确保是binary mask
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    # 计算交集
    intersection = np.sum(pred * target)
    
    # 计算总数
    total = np.sum(pred) + np.sum(target)
    
    # 计算并集
    union = total - intersection
    
    # 计算 IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # 计算 Dice 系数
    dice = (2. * intersection + smooth) / (total + smooth)
    
    return iou, dice


def get_image_mask_paths(root_dir):
    """获取所有图像和mask路径"""
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
    使用 MedSAM2 进行推理
    
    Args:
        image_predictor: SAM2ImagePredictor 实例
        img_rgb: RGB 格式的图像，HWC 格式，像素值在 [0, 255]
        box: 边界框，格式为 [x0, y0, x1, y1]
    
    Returns:
        medsam_seg: 分割结果，binary mask (H, W)
    """
    # 设置图像
    image_predictor.set_image(img_rgb)
    
    # 预测掩码
    box_array = np.array(box).reshape(1, 4) if len(box.shape) == 1 else box
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_array,
        multimask_output=False,
        return_logits=False,
        normalize_coords=True,
    )
    
    # 获取第一个掩码（因为 multimask_output=False）
    medsam_seg = (masks[0] > 0.0).astype(np.uint8)
    return medsam_seg


def eval_model(model, image_paths, mask_paths, annotation_path, device, image_size=512):
    """
    评估模型性能
    
    Args:
        model: MedSAM2模型
        image_paths: 图像路径列表
        mask_paths: mask路径列表
        annotation_path: annotation字典路径
        device: 设备
        image_size: 图像尺寸
    
    Returns:
        avg_iou: 平均IoU
        avg_dice: 平均Dice
    """
    model.eval()
    
    # 加载annotation字典
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
        
        # 读取图像
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np
        
        # 确保图像是uint8格式，像素值在[0, 255]
        if img_3c.dtype != np.uint8:
            img_3c = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            ) * 255.0
            img_3c = img_3c.astype(np.uint8)
        
        H, W, _ = img_3c.shape
        
        # 读取mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Cannot read mask {mask_path}, skipping...")
            continue
        
        # 获取参考mask路径
        case_num = os.path.basename(os.path.dirname(mask_path))
        ref_mask_path = None
        
        if case_num in ref_map:
            info = ref_map[case_num]
            if isinstance(info, dict):
                ref_mask_path = info.get("mask_path")
                if ref_mask_path and not os.path.exists(ref_mask_path):
                    ref_mask_path = None
        
        # 如果参考mask不存在，使用当前mask
        if ref_mask_path is None or not os.path.exists(ref_mask_path):
            ref_mask_path = mask_path
        
        # 从参考mask中提取bounding box
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            print(f"Warning: Cannot read ref_mask {ref_mask_path}, using current mask...")
            ref_mask = mask
        
        # 调整ref_mask大小到image_size
        ref_mask_resized = cv2.resize(ref_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        y_indices, x_indices = np.where(ref_mask_resized > 127)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            # 如果参考mask为空，使用当前mask
            mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            y_indices, x_indices = np.where(mask_resized > 127)
            if len(y_indices) == 0 or len(x_indices) == 0:
                print(f"Warning: Empty mask for {mask_path}, skipping...")
                continue
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 将bbox从image_size尺度映射回原始图像尺度
        x_min = int(x_min * W / image_size)
        x_max = int(x_max * W / image_size)
        y_min = int(y_min * H / image_size)
        y_max = int(y_max * H / image_size)
        
        box_prompt = np.array([x_min, y_min, x_max, y_max])
        
        # 使用MedSAM2进行推理
        medsam_seg = medsam2_inference(image_predictor, img_3c, box_prompt)
        
        # 调整预测mask大小到原始mask大小
        if medsam_seg.shape != mask.shape:
            medsam_seg = cv2.resize(medsam_seg, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 二值化mask
        mask_binary = (mask > 127).astype(np.uint8)
        
        # 计算指标
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
        choices=["kits", "pancreas", "lits", "colon", "local"],
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
        help="Batch size (not used in inference, kept for compatibility)"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 构建模型
    medsam2_root = os.path.dirname(os.path.abspath(__file__))
    
    # 处理checkpoint路径
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
    
    # 如果是训练checkpoint，需要加载state_dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model" in checkpoint:
            sam2_model.load_state_dict(checkpoint["model"], strict=False)
            print("Loaded model weights from training checkpoint")
    except Exception as e:
        print(f"Warning: Could not load checkpoint as training checkpoint: {e}")
        print("Using checkpoint as pretrained model")
    
    # 创建数据集路径
    dataset_dir = os.path.join("./2D_data", args.data, "test")
    
    # 获取图像和mask路径
    image_paths, mask_paths = get_image_mask_paths(dataset_dir)
    
    print(f"Test samples: {len(image_paths)}")
    
    # 获取annotation路径
    if args.ref_type == 'middle':
        annotation_path = os.path.join(dataset_dir, "annotation_dict_middle.json")
    else:
        annotation_path = os.path.join(dataset_dir, "annotation_dict_neighbor.json")
    
    # 评估模型
    print(f"\n{'='*50}")
    print(f"Evaluating on {args.data} dataset (ref_type: {args.ref_type})")
    print(f"{'='*50}")
    
    avg_iou, avg_dice = eval_model(
        sam2_model,
        image_paths,
        mask_paths,
        annotation_path,
        device,
        image_size=args.image_size
    )
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"{'='*50}\n")
    
    # 保存结果
    results = {
        "dataset": args.data,
        "ref_type": args.ref_type,
        "checkpoint": checkpoint_path,
        "avg_iou": float(avg_iou),
        "avg_dice": float(avg_dice),
        "num_samples": len(image_paths)
    }
    
    results_dir = os.path.join(medsam2_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{args.data}_{args.ref_type}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
