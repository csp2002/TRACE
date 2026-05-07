# -*- coding: utf-8 -*-
"""
MedSAM2 2D训练脚本
使用middle或neighbor slice的GT Mask提取box prompt进行训练
直接使用MedSAM2模型进行训练，使用MedSAM2的loss函数和transforms
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
from training.loss_fns import dice_loss, sigmoid_focal_loss

# 设置随机种子
torch.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2023)


def forward_with_box(model, images, boxes, image_size=512, multimask_output=True):
    """
    使用box prompt进行前向传播（直接调用模型的内部方法）
    
    Args:
        model: MedSAM2模型
        images: (B, 3, H, W) 图像tensor，已归一化（ImageNet mean/std）
        boxes: (B, 4) box坐标 [x_min, y_min, x_max, y_max]，在image_size尺度下
        image_size: 图像尺寸
        multimask_output: 是否输出多个mask
    
    Returns:
        high_res_multimasks, ious, low_res_masks, high_res_masks
    """
    B = images.shape[0]
    device = images.device
    
    # 1. 获取图像特征
    backbone_out = model.forward_image(images)
    _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
    
    # 2. 准备backbone_features（用于_forward_sam_heads）
    # vision_feats[-1]的格式是 (HW, B, C)，需要转换为 (B, C, H, W)
    H, W = feat_sizes[-1]
    C = model.hidden_dim
    # vision_feats[-1]: (HW, B, C) -> (B, C, H, W)
    backbone_features = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)  # (B, C, H, W)
    
    # 3. 准备high-res features（如果需要）
    if model.use_high_res_features_in_sam and len(vision_feats) > 1:
        # vision_feats[i]: (HW, B, C) -> (B, C, H, W)
        # 参考_track_step中的实现
        high_res_features = [
            vision_feats[i].permute(1, 2, 0).view(vision_feats[i].size(1), vision_feats[i].size(2), *feat_sizes[i])
            for i in range(len(vision_feats) - 1)
        ]
    else:
        high_res_features = None
    
    # 4. 准备box prompt
    # boxes格式: [x_min, y_min, x_max, y_max]
    # prompt_encoder期望的boxes格式是 (B, 2, 2): [[x_min, y_min], [x_max, y_max]]
    boxes_torch = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if boxes_torch.dim() == 2 and boxes_torch.shape[1] == 4:
        boxes_torch = boxes_torch.reshape(B, 2, 2)  # (B, 2, 2)
    
    # 5. 调用prompt encoder（传入boxes）
    # points可以直接传入None，prompt_encoder内部会处理
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=None,  # 没有point prompt
        boxes=boxes_torch,  # 传入boxes
        masks=None,
    )
    
    # 6. 调用mask decoder
    (
        low_res_multimasks,
        ious,
        sam_output_tokens,
        object_score_logits,
    ) = model.sam_mask_decoder(
        image_embeddings=backbone_features,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    

    # 7. 处理object scores（如果需要）
    if model.pred_obj_scores:
    #     print("pred_obj_scores:", model.pred_obj_scores,
    #   "obj_logit mean/min/max:",
    #   object_score_logits.mean().item(),
    #   object_score_logits.min().item(),
    #   object_score_logits.max().item())

        is_obj_appearing = object_score_logits > 0
        NO_OBJ_SCORE = -1e10
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            torch.tensor(NO_OBJ_SCORE, device=device, dtype=low_res_multimasks.dtype),
        )
    
    # 8. 上采样到原始图像尺寸
    low_res_multimasks = low_res_multimasks.float()
    high_res_multimasks = F.interpolate(
        low_res_multimasks,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    
    # 9. 选择最佳mask（如果multimask_output=True）
    if multimask_output:
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(B, device=device)
        

        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
        high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
    else:
        low_res_masks = low_res_multimasks
        high_res_masks = high_res_multimasks
    
    return high_res_multimasks, ious, low_res_masks, high_res_masks


def compute_loss(pred_masks, pred_ious, gt_masks, num_objects, weight_dict=None):
    """
    计算loss（focal loss + dice loss，论文配置：20:1）
    
    Args:
        pred_masks: (B, M, H, W) 预测mask logits（multimask输出）
        pred_ious: (B, M) 预测IoU（未使用，仅保留接口兼容性）
        gt_masks: (B, 1, H, W) 真实mask [0, 1]
        num_objects: 对象数量（batch size）
        weight_dict: loss权重字典
    
    Returns:
        total_loss, focal_loss, dice_loss_val
    """
   

    if weight_dict is None:
        weight_dict = {
            "loss_mask": 20.0,  # focal loss权重（论文：20）
            "loss_dice": 1.0,   # dice loss权重（论文：1）
        }
    
    # 扩展gt_masks以匹配multimask输出
    gt_masks_expanded = gt_masks.expand_as(pred_masks)  # (B, M, H, W)
    
    # 计算multimask的loss
    loss_multimask = sigmoid_focal_loss(
        pred_masks,
        gt_masks_expanded,
        num_objects,
        alpha=0.25,
        gamma=2.0,
        loss_on_multimask=True,
    )  # (B, M)
    
    loss_multidice = dice_loss(
        pred_masks,
        gt_masks_expanded,
        num_objects,
        loss_on_multimask=True,
    )  # (B, M)
    
    # 选择loss最小的mask进行反向传播（基于focal+dice组合）
    if loss_multimask.size(1) > 1:
        loss_combo = (
            loss_multimask * weight_dict["loss_mask"]
            + loss_multidice * weight_dict["loss_dice"]
        )
        best_loss_inds = torch.argmin(loss_combo, dim=-1)
        batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
        loss_mask = loss_multimask[batch_inds, best_loss_inds]
        loss_dice_val = loss_multidice[batch_inds, best_loss_inds]
    else:
        loss_mask = loss_multimask.squeeze(1)
        loss_dice_val = loss_multidice.squeeze(1)
    
    # 计算加权总loss（仅focal loss和dice loss）
    # 注意：loss_multimask和loss_multidice已经除以了num_objects
    # 所以这里使用sum()而不是mean()（与MedSAM2原始实现一致）
    total_loss = (
        loss_mask.sum() * weight_dict["loss_mask"]
        + loss_dice_val.sum() * weight_dict["loss_dice"]
    )
    
    return total_loss, loss_mask.sum(), loss_dice_val.sum()


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
        
        batch_size = images.shape[0]
        optimizer.zero_grad()
        
        # bboxes已经在image_size尺度下
        boxes_array = bboxes  # (B, 4)
        
        if use_amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # 前向传播
                

                high_res_multimasks, ious, _, _ = forward_with_box(
                    model, images, boxes_array, image_size=image_size, multimask_output=True
                )
                
                # 计算loss
                num_objects = float(batch_size)
                total_loss_val, focal_loss, dice_loss_val = compute_loss(
                    high_res_multimasks, ious, masks, num_objects
                )
                H, W = masks.shape[-2], masks.shape[-1]
                # print("focal(raw)=", focal_loss.item(), "focal*HW=", focal_loss.item() * H * W)

            
            scaler.scale(total_loss_val).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # print("images.shape:", images.shape,"max:", images.max(), "min:", images.min())
            # print("masks.shape:", masks.shape,"max:", masks.max(), "min:", masks.min())
            # print("boxes_array.shape:", boxes_array.shape,"max:", boxes_array.max(), "min:", boxes_array.min())
            # 前向传播
            high_res_multimasks, ious, _, _ = forward_with_box(
                model, images, boxes_array, image_size=image_size, multimask_output=True
            )
            # print("high_res_multimasks.shape:", high_res_multimasks.shape,"max:", high_res_multimasks.max(), "min:", high_res_multimasks.min())
            # 计算loss
            num_objects = float(batch_size)
            total_loss_val, focal_loss, dice_loss_val = compute_loss(
                high_res_multimasks, ious, masks, num_objects
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
            # 'p_focal': f'{focal_loss.item():.3e}',

            'focal': f'{focal_loss.item():.4f}',
            'dice': f'{dice_loss_val.item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_focal = total_focal_loss / num_batches
    avg_dice = total_dice_loss / num_batches
    
    return avg_loss, avg_focal, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train MedSAM2 on 2D medical images")
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
        help="Batch size (default: 8 as in paper)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=70,
        help="Number of training epochs (default: 70 as in paper)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for non-image-encoder components (default: 5e-5 as in paper)"
    )
    parser.add_argument(
        "--vision_lr",
        type=float,
        default=3e-5,
        help="Learning rate for image encoder (default: 3e-5 as in paper)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01 as in paper)"
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
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    task_name = f"MedSAM2-2D-{args.data}-{args.ref_type}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(args.work_dir, task_name, run_id)
    os.makedirs(model_save_path, exist_ok=True)
    
    # 保存训练脚本
    shutil.copyfile(__file__, os.path.join(model_save_path, "train_script.py"))
    
    # 保存参数
    with open(os.path.join(model_save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device(args.device)
    
    # 构建模型
    medsam2_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(medsam2_root, args.checkpoint)
    
    print(f"Loading model from {checkpoint_path}")
    print(f"Using config: {args.cfg}")
    
    sam2_model = build_sam2(
        config_file=args.cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="train"
    )
    
    # 打印模型参数
    total_params = sum(p.numel() for p in sam2_model.parameters())
    trainable_params = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 创建数据集
    dataset_dir = os.path.join("./2D_data", args.data)
    train_dataset = MedSAM2_2D_Dataset(
        base_dir=dataset_dir,
        mode="train",
        ref_type=args.ref_type,
        image_size=args.image_size,
        bbox_shift=10  # Paper: 0-10 pixels
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
    
    # 优化器（分层学习率：image_encoder使用3e-5，其他组件使用5e-5）
    # 分离image_encoder参数和其他参数
    image_encoder_params = []
    other_params = []
    
    for name, param in sam2_model.named_parameters():
        if 'image_encoder' in name:
            image_encoder_params.append(param)
        else:
            other_params.append(param)
    
    # 创建参数组
    param_groups = [
        {'params': image_encoder_params, 'lr': args.vision_lr, 'weight_decay': args.weight_decay},
        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),  # Paper: β1=0.9, β2=0.999
        weight_decay=args.weight_decay
    )
    
    print(f"Image encoder parameters: {sum(p.numel() for p in image_encoder_params) / 1e6:.2f}M")
    print(f"Other parameters: {sum(p.numel() for p in other_params) / 1e6:.2f}M")
    print(f"Image encoder LR: {args.vision_lr}, Other LR: {args.lr}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get("epoch", 0) + 1
            sam2_model.load_state_dict(checkpoint.get("model", checkpoint))
            optimizer.load_state_dict(checkpoint.get("optimizer", {}))
            print(f"Resumed from epoch {start_epoch}")
    
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
            sam2_model, 
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
            "model": sam2_model.state_dict(),
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
        plt.figure(figsize=(12, 4))
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
