# -*- coding: utf-8 -*-
"""
MedSAM2 2D training script — TRACE variant
包含 adds-on refinement module + iterative refinement + deep supervision
使用middle或neighbor slice的GT Mask提取box prompt进行训练
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import shutil
import json
import matplotlib.pyplot as plt

# 添加MedSAM2目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from training.dataset.medsam2_2d_dataset import MedSAM2_2D_Dataset
from training.model.medsam2_with_trace import MedSAM2_with_TRACE
from training.model.trace import TRACE
from training.loss_fns import dice_loss, sigmoid_focal_loss

# 设置随机种子
torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2023)


def compute_loss_with_trace(preds_all, gt_masks, num_objects, weight_dict=None):
    """
    Compute the TRACE-variant loss（deep supervision：所有迭代的输出都参与loss计算）
    使用MedSAM2的loss函数：focal loss + dice loss，权重比例20:1
    
    Args:
        preds_all: list of (B, 1, H, W) 所有迭代的mask logits
        gt_masks: (B, 1, H, W) 真实mask [0, 1]
        num_objects: 对象数量（batch size）
        weight_dict: loss权重字典
    
    Returns:
        total_loss, focal_loss, dice_loss_val
    """
    if weight_dict is None:
        weight_dict = {
            "loss_mask": 20.0,  # focal loss权重（MedSAM2论文：20）
            "loss_dice": 1.0,   # dice loss权重（MedSAM2论文：1）
        }
    
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    
    # 对所有迭代的输出计算loss（deep supervision）
    for pred_logits in preds_all:
        # pred_logits: (B, 1, H, W) - mask logits
        # gt_masks: (B, 1, H, W) - ground truth masks [0, 1]
        # 注意：MedSAM2的loss函数期望输入是logits（未sigmoid），target是[0,1]
        
        # 确保尺寸匹配
        assert pred_logits.shape == gt_masks.shape, f"Shape mismatch: pred_logits {pred_logits.shape} vs gt_masks {gt_masks.shape}"
        
        # 计算focal loss（MedSAM2的参数：alpha=0.25, gamma=2.0）
        focal_loss_val = sigmoid_focal_loss(
            pred_logits,
            gt_masks,
            num_objects,
            alpha=0.25,
            gamma=2.0,
            loss_on_multimask=False,
        )  # 返回的是已经除以num_objects的值
        
        # 计算dice loss
        # dice_loss 在 loss_on_multimask=False 时需要 inputs 和 targets 都是 (B, 1, H, W)
        # 但函数内部会 flatten inputs，所以需要确保 targets 也匹配
        # 修复：在调用前确保 targets 也被正确 flatten
        dice_loss_val = dice_loss(
            pred_logits,
            gt_masks,
            num_objects,
            loss_on_multimask=False,
        )  # 返回的是已经除以num_objects的值
        
        total_focal_loss += focal_loss_val
        total_dice_loss += dice_loss_val
    
    # 平均所有迭代的loss
    num_iters = len(preds_all)
    avg_focal_loss = total_focal_loss / num_iters
    avg_dice_loss = total_dice_loss / num_iters
    
    # 计算加权总loss（使用sum()而不是mean()，因为loss函数已经除以了num_objects）
    total_loss = (
        avg_focal_loss * weight_dict["loss_mask"]
        + avg_dice_loss * weight_dict["loss_dice"]
    )
    
    return total_loss, avg_focal_loss, avg_dice_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch, use_amp=False, image_size=512):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        images = batch['image'].to(device)  # (B, 3, H, W), float, normalized
        masks = batch['mask'].to(device)  # (B, 1, H, W), float [0, 1]
        bboxes = batch['bbox'].cpu().numpy()  # (B, 4), numpy [x_min, y_min, x_max, y_max]
        
        # extra data needed by the TRACE variant
        if 'ref_image' not in batch or 'ref_gt' not in batch:
            raise ValueError("Dataset must return 'ref_image' and 'ref_gt' for the TRACE variant")
        
        ref_images = batch['ref_image'].to(device)  # (B, 3, H, W), float, normalized
        ref_gts = batch['ref_gt'].to(device)  # (B, 1, H, W), float [0, 1]
        batch_size = images.shape[0]
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # 前向传播
                outputs = model(
                    images, bboxes, ref_images, ref_gts, image_size=image_size
                )
                
                # 计算loss（deep supervision）
                num_objects = float(batch_size)
                total_loss_val, focal_loss, dice_loss_val = compute_loss_with_trace(
                    outputs['iters'], masks, num_objects
                )
            
            scaler.scale(total_loss_val).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = model(
                images, bboxes, ref_images, ref_gts, image_size=image_size
            )
            
            # 计算loss（deep supervision）
            num_objects = float(batch_size)
            total_loss_val, focal_loss, dice_loss_val = compute_loss_with_trace(
                outputs['iters'], masks, num_objects
            )
            
            total_loss_val.backward()
            optimizer.step()
        
        total_loss += total_loss_val.item()
        total_focal_loss += focal_loss.item()
        total_dice_loss += dice_loss_val.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss_val.item():.4f}',
            'focal': f'{focal_loss.item():.4f}',
            'dice': f'{dice_loss_val.item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_focal = total_focal_loss / num_batches
    avg_dice = total_dice_loss / num_batches
    
    return avg_loss, avg_focal, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train MedSAM2 + TRACE on 2D medical images")
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
        help="Pretrained checkpoint path (relative to MedSAM2 root)"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./work_dir",
        help="Working directory to save models and logs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8 as in MedSAM2 paper)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=70,
        help="Number of training epochs (default: 70)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for non-image-encoder components"
    )
    parser.add_argument(
        "--vision_lr",
        type=float,
        default=3e-5,
        help="Learning rate for image encoder"
    )
    parser.add_argument(
        "--refinement_lr",
        type=float,
        default=5e-5,
        help="Learning rate for refinement module (default: 5e-5, same as other components)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for training"
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=3,
        help="Number of refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--freeze_sam2",
        action="store_true",
        help="Freeze SAM2 model, only train refinement module"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    task_name = f"MedSAM2-2D-with_TRACE-{args.data}-{args.ref_type}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(args.work_dir, task_name, run_id)
    os.makedirs(model_save_path, exist_ok=True)
    
    # 保存训练脚本
    shutil.copyfile(__file__, os.path.join(model_save_path, "train_script.py"))
    
    # 保存参数
    with open(os.path.join(model_save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device(args.device)
    
    # 构建MedSAM2基础模型
    medsam2_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(medsam2_root, args.checkpoint)
    
    print(f"Loading MedSAM2 model from {checkpoint_path}")
    print(f"Using config: {args.cfg}")
    
    sam2_model = build_sam2(
        config_file=args.cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="train"
    )
    
    # 创建refinement模块
    # 创建一个简单的config对象
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
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 冻结SAM2（如果需要）
    if args.freeze_sam2:
        for param in model.sam2_model.parameters():
            param.requires_grad = False
        for param in model.refinement.parameters():
            param.requires_grad = True
        print("Frozen SAM2 model, only training refinement module")
    
    # 创建数据集
    dataset_dir = os.path.join("./2D_data", args.data)
    train_dataset = MedSAM2_2D_Dataset(
        base_dir=dataset_dir,
        mode="train",
        ref_type=args.ref_type,
        image_size=args.image_size,
        bbox_shift=10
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 优化器（分层学习率）
    image_encoder_params = []
    refinement_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'image_encoder' in name or 'sam2_model.image_encoder' in name:
            image_encoder_params.append(param)
        elif 'refinement' in name:
            refinement_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组
    param_groups = []
    if image_encoder_params:
        param_groups.append({
            'params': image_encoder_params, 
            'lr': args.vision_lr, 
            'weight_decay': args.weight_decay
        })
    if refinement_params:
        param_groups.append({
            'params': refinement_params, 
            'lr': args.refinement_lr, 
            'weight_decay': args.weight_decay
        })
    if other_params:
        param_groups.append({
            'params': other_params, 
            'lr': args.lr, 
            'weight_decay': args.weight_decay
        })
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),  # MedSAM2: β1=0.9, β2=0.999
        weight_decay=args.weight_decay
    )
    
    print(f"Image encoder parameters: {sum(p.numel() for p in image_encoder_params) / 1e6:.2f}M")
    print(f"Refinement parameters: {sum(p.numel() for p in refinement_params) / 1e6:.2f}M")
    print(f"Other parameters: {sum(p.numel() for p in other_params) / 1e6:.2f}M")
    print(f"Vision LR: {args.vision_lr}, Refinement LR: {args.refinement_lr}, Other LR: {args.lr}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get("epoch", 0) + 1
            model.load_state_dict(checkpoint.get("model", checkpoint))
            optimizer.load_state_dict(checkpoint.get("optimizer", {}))
            print(f"Resumed from epoch {start_epoch}")
    
    model = model.to(device)
    
    # 训练循环
    losses = []
    focal_losses = []
    dice_losses = []
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*50}")
        
        avg_loss, avg_focal, avg_dice = train_one_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device, 
            epoch,
            use_amp=args.use_amp,
            image_size=args.image_size
        )
        
        losses.append(avg_loss)
        focal_losses.append(avg_focal)
        dice_losses.append(avg_dice)
        
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Focal: {avg_focal:.4f}, Dice: {avg_dice:.4f}")
        
        # 保存checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "focal_loss": avg_focal,
            "dice_loss": avg_dice,
        }
        
        # 保存最新模型
        torch.save(checkpoint, os.path.join(model_save_path, "latest.pth"))
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(model_save_path, "best.pth"))
            print(f"Saved best model (loss: {best_loss:.4f})")
        
        # 绘制loss曲线
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Total Loss")
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(focal_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Focal Loss")
        plt.title("Focal Loss")
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(dice_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Dice Loss")
        plt.title("Dice Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, "loss_curves.png"), dpi=150)
        plt.close()
        
        print(f"Loss curves saved to {os.path.join(model_save_path, 'loss_curves.png')}")


if __name__ == "__main__":
    main()
