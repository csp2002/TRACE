# -*- coding: utf-8 -*-
"""
MedSAM2 2D test script — TRACE variant
Evaluate the trained MedSAM2 + TRACE model.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

# 添加MedSAM2目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from training.dataset.medsam2_2d_dataset import MedSAM2_2D_Dataset
from training.model.medsam2_with_trace import MedSAM2_with_TRACE
from training.model.trace import TRACE
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
    # 展平
    pred = pred.flatten()
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


def eval_model(model, dataloader, device, image_size=512):
    """
    评估模型性能
    
    Args:
        model: MedSAM2 + TRACE model
        dataloader: 数据加载器
        device: 设备
        image_size: 图像尺寸
    
    Returns:
        avg_iou: 平均IoU
        avg_dice: 平均Dice
    """
    model.eval()
    
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)  # (B, 3, H, W)
            masks = batch['mask'].to(device)  # (B, 1, H, W)
            bboxes = batch['bbox'].cpu().numpy()  # (B, 4)
            
            # extra data needed by the TRACE variant
            if 'ref_image' not in batch or 'ref_gt' not in batch:
                raise ValueError("Dataset must return 'ref_image' and 'ref_gt' for the TRACE variant")
            
            ref_images = batch['ref_image'].to(device)  # (B, 3, H, W)
            ref_gts = batch['ref_gt'].to(device)  # (B, 1, H, W)
            
            # 前向传播
            outputs = model(
                images, bboxes, ref_images, ref_gts, image_size=image_size
            )
            
            # 获取最终预测
            pred_logits = outputs['final']  # (B, 1, H, W)
            pred_masks = torch.sigmoid(pred_logits)  # (B, 1, H, W)
            
            # 转换为numpy并计算指标
            pred_np = pred_masks.cpu().numpy()
            target_np = masks.cpu().numpy()
            
            batch_size = pred_np.shape[0]
            for i in range(batch_size):
                iou, dice = compute_metrics(pred_np[i, 0], target_np[i, 0])
                total_iou += iou
                total_dice += dice
                num_samples += 1
    
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0
    
    return avg_iou, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Test MedSAM2 + TRACE on 2D medical images")
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
        "--cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="Model config file (relative to MedSAM2 root)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/MedSAM2_latest.pt",
        help="Pretrained MedSAM2 checkpoint path"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to trained MedSAM2 + TRACE checkpoint"
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
        help="Image size"
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=3,
        help="Number of refinement iterations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 构建MedSAM2基础模型
    medsam2_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(medsam2_root, args.checkpoint)
    
    print(f"Loading MedSAM2 model from {checkpoint_path}")
    sam2_model = build_sam2(
        config_file=args.cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval"
    )
    
    # 创建refinement模块
    class RefinementConfig:
        def __init__(self):
            self.classifier = None
    
    refinement_config = RefinementConfig()
    refinement_module = TRACE(
        refinement_config, 
        img_size=args.image_size, 
        pretrained=True
    )
    
    # Build MedSAM2 + TRACE model
    model = MedSAM2_with_TRACE(
        sam2_model=sam2_model,
        refinement=refinement_module,
        refine_iters=args.refine_iters,
        detach_between_iters=True
    )
    
    # 加载训练好的模型权重
    print(f"Loading trained model from {args.model_ckpt}")
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    
    # 处理checkpoint格式
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 创建测试数据集
    dataset_dir = os.path.join("./2D_data", args.data)
    test_dataset = MedSAM2_2D_Dataset(
        base_dir=dataset_dir,
        mode="test",
        ref_type=args.ref_type,
        image_size=args.image_size,
        bbox_shift=0,  # 测试时不添加扰动
        use_augmentation=False  # 测试时不使用数据增强
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 评估模型
    print("Starting evaluation...")
    avg_iou, avg_dice = eval_model(
        model, test_dataloader, device, image_size=args.image_size
    )
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
