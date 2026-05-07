# -*- coding: utf-8 -*-
"""
MedSAM2 Video模式测试脚本
将每个病人的所有2D图片当作一个video，使用middle slice的GT提取box prompt进行分割
最后计算每个slice的2D IoU和Dice并在数据集上取平均
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import json
import cv2
from skimage import io
import tempfile
import shutil

# 添加MedSAM2目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2_video_predictor


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


def extract_box_from_mask(mask, image_size=512):
    """
    从mask中提取bounding box
    
    Args:
        mask: mask数组 (H, W)，值在[0, 255]或[0, 1]
        image_size: 图像尺寸，用于调整mask大小
    
    Returns:
        box: bounding box [x_min, y_min, x_max, y_max]（原始尺寸）
    """
    # 确保mask是binary
    if mask.max() > 1.0:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = (mask > 0.5).astype(np.uint8)
    
    H, W = mask_binary.shape
    
    # 调整mask大小到image_size
    mask_resized = cv2.resize(mask_binary.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    y_indices, x_indices = np.where(mask_resized > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果mask为空，返回整个图像
        return np.array([0, 0, W, H])
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # 将bbox从image_size尺度映射回原始图像尺度
    x_min = int(x_min * W / image_size)
    x_max = int(x_max * W / image_size)
    y_min = int(y_min * H / image_size)
    y_max = int(y_max * H / image_size)
    
    # 确保bbox在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    
    box = np.array([x_min, y_min, x_max, y_max])
    return box


def organize_patient_slices(root_dir, case_num):
    """
    组织某个病人的所有slice图片路径
    
    Args:
        root_dir: 数据集根目录（包含CT和Mask子目录）
        case_num: 病人编号
    
    Returns:
        slice_paths: list of (image_path, mask_path, slice_filename) tuples，按slice编号排序
    """
    ct_dir = os.path.join(root_dir, "CT", case_num)
    mask_dir = os.path.join(root_dir, "Mask", case_num)
    
    if not os.path.isdir(ct_dir):
        return []
    
    slice_paths = []
    for filename in os.listdir(ct_dir):
        if not os.path.isfile(os.path.join(ct_dir, filename)):
            continue
        
        image_path = os.path.join(ct_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        if os.path.exists(mask_path):
            slice_paths.append((image_path, mask_path, filename))
    
    # 按文件名排序（假设文件名包含slice编号）
    slice_paths.sort(key=lambda x: x[2])
    
    return slice_paths


def create_temp_video_dir(slice_paths, temp_dir, start_idx=0):
    """
    创建临时目录，将slice图片复制为video frames（命名为00000.jpg, 00001.jpg等）
    
    Args:
        slice_paths: list of (image_path, mask_path, slice_filename) tuples
        temp_dir: 临时目录路径
        start_idx: 起始索引（用于frame编号）
    
    Returns:
        video_dir: 创建的video目录路径
        slice_order: 从filename到frame_idx的映射
    """
    os.makedirs(temp_dir, exist_ok=True)
    video_dir = temp_dir
    
    slice_order = {}
    for local_idx, (image_path, _, filename) in enumerate(slice_paths):
        # 生成frame文件名（5位数字，从start_idx开始）
        frame_name = f"{start_idx + local_idx:05d}.jpg"
        frame_path = os.path.join(video_dir, frame_name)
        
        # 读取图像并转换为RGB
        img = io.imread(image_path)
        if len(img.shape) == 2:
            img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img_3c = img[:, :, :3]
        else:
            img_3c = img
        
        # 确保是uint8格式
        if img_3c.dtype != np.uint8:
            img_3c = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            ) * 255.0
            img_3c = img_3c.astype(np.uint8)
        
        # 保存为JPEG（video predictor期望JPEG格式）
        io.imsave(frame_path, img_3c)
        
        slice_order[filename] = start_idx + local_idx
    
    return video_dir, slice_order


@torch.no_grad()
def eval_model_video(model, root_dir, annotation_path, device, image_size=512):
    """
    使用video模式评估模型性能
    
    Args:
        model: MedSAM2 video predictor模型
        root_dir: 数据集根目录
        annotation_path: annotation字典路径（包含middle slice信息）
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
    
    # 按case组织所有slice
    ct_dir = os.path.join(root_dir, "CT")
    case_nums = [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))]
    case_nums.sort()
    
    total_iou = 0.0
    total_dice = 0.0
    num_slices = 0
    
    # 创建临时目录用于存放video frames
    temp_base_dir = tempfile.mkdtemp(prefix="medsam2_video_")
    
    try:
        for case_num in tqdm(case_nums, desc="Processing patients"):
            # 组织该病人的所有slice
            slice_paths = organize_patient_slices(root_dir, case_num)
            if len(slice_paths) == 0:
                continue
            
            # 获取middle slice的mask路径（必须从annotation_dict_middle.json获取）
            middle_mask_path = None
            if case_num in ref_map:
                info = ref_map[case_num]
                if isinstance(info, dict):
                    middle_mask_path = info.get("mask_path")
            
            # 如果找不到middle mask，跳过这个case（必须使用middle slice的GT）
            if middle_mask_path is None or not os.path.exists(middle_mask_path):
                print(f"Warning: Middle slice mask not found for case {case_num}, skipping...")
                continue
            
            # 从middle mask提取box
            middle_mask = cv2.imread(middle_mask_path, cv2.IMREAD_GRAYSCALE)
            if middle_mask is None:
                print(f"Warning: Cannot read middle mask {middle_mask_path}, skipping...")
                continue
            
            # 找到middle slice在slice_paths中的索引
            middle_filename = os.path.basename(middle_mask_path)
            middle_idx = None
            for idx, (_, _, filename) in enumerate(slice_paths):
                if filename == middle_filename:
                    middle_idx = idx
                    break
            
            if middle_idx is None:
                print(f"Warning: Middle slice {middle_filename} not found in slice_paths for case {case_num}, skipping...")
                continue
            
            # 读取middle slice图像以获取原始尺寸
            middle_image_path = middle_mask_path.replace('Mask', 'CT')
            middle_img = io.imread(middle_image_path)
            if len(middle_img.shape) == 2:
                H, W = middle_img.shape
            else:
                H, W = middle_img.shape[:2]
            
            # 提取box（在原始图像尺寸下）
            box_prompt = extract_box_from_mask(middle_mask, image_size)
            
            # 将slice_paths分为两部分（以middle slice为界）
            # 第一部分：从第一个slice到middle slice（包括middle）
            part1_slices = slice_paths[:middle_idx + 1]
            # 第二部分：从middle slice到最后一个slice（包括middle）
            part2_slices = slice_paths[middle_idx:]
            
            # 用于存储所有预测结果，key是原始slice_paths中的索引
            all_masks = {}
            
            try:
                # ========== 处理第一部分：从第一个slice到middle slice ==========
                if len(part1_slices) > 0:
                    temp_video_dir_part1 = os.path.join(temp_base_dir, f"{case_num}_part1")
                    video_dir_part1, slice_order_part1 = create_temp_video_dir(part1_slices, temp_video_dir_part1, start_idx=0)
                    
                    # middle slice在第一部分中的frame索引（应该是最后一个）
                    middle_frame_idx_part1 = len(part1_slices) - 1
                    
                    # 初始化video inference state
                    inference_state_part1 = model.init_state(
                        video_path=video_dir_part1,
                        async_loading_frames=False
                    )
                    
                    # 在middle frame（第一部分的最后一个frame）添加box prompt
                    obj_id = 0
                    model.add_new_points_or_box(
                        inference_state=inference_state_part1,
                        frame_idx=middle_frame_idx_part1,
                        obj_id=obj_id,
                        box=box_prompt.astype(np.float32),
                        normalize_coords=True,
                    )
                    
                    # 反向传播（从middle到第一个slice）
                    for frame_idx, obj_ids, video_res_masks in model.propagate_in_video(
                        inference_state_part1,
                        start_frame_idx=middle_frame_idx_part1,
                        reverse=True
                    ):
                        mask_np = video_res_masks[obj_id, 0].cpu().numpy()
                        # 将frame_idx映射回原始slice_paths的索引
                        # part1_slices中的frame_idx对应slice_paths中的索引
                        original_idx = frame_idx  # 因为part1_slices是从slice_paths[0:middle_idx+1]取的
                        all_masks[original_idx] = mask_np
                
                # ========== 处理第二部分：从middle slice到最后一个slice ==========
                if len(part2_slices) > 0:
                    temp_video_dir_part2 = os.path.join(temp_base_dir, f"{case_num}_part2")
                    video_dir_part2, slice_order_part2 = create_temp_video_dir(part2_slices, temp_video_dir_part2, start_idx=0)
                    
                    # middle slice在第二部分中的frame索引（应该是第一个）
                    middle_frame_idx_part2 = 0
                    
                    # 初始化video inference state
                    inference_state_part2 = model.init_state(
                        video_path=video_dir_part2,
                        async_loading_frames=False
                    )
                    
                    # 在middle frame（第二部分的第一个frame）添加box prompt
                    obj_id = 0
                    model.add_new_points_or_box(
                        inference_state=inference_state_part2,
                        frame_idx=middle_frame_idx_part2,
                        obj_id=obj_id,
                        box=box_prompt.astype(np.float32),
                        normalize_coords=True,
                    )
                    
                    # 正向传播（从middle到最后一个slice）
                    for frame_idx, obj_ids, video_res_masks in model.propagate_in_video(
                        inference_state_part2,
                        start_frame_idx=middle_frame_idx_part2,
                        reverse=False
                    ):
                        mask_np = video_res_masks[obj_id, 0].cpu().numpy()
                        # 将frame_idx映射回原始slice_paths的索引
                        # part2_slices中的frame_idx对应slice_paths中的索引
                        original_idx = middle_idx + frame_idx  # 因为part2_slices是从slice_paths[middle_idx:]取的
                        # 如果middle slice已经在第一部分处理过，使用第二部分的预测（覆盖）
                        all_masks[original_idx] = mask_np
                
                # 对每个slice计算指标（middle slice只计算一次）
                processed_middle = False
                for original_idx, (image_path, mask_path, filename) in enumerate(slice_paths):
                    if original_idx not in all_masks:
                        continue
                    
                    # 如果是middle slice，只计算一次
                    if original_idx == middle_idx:
                        if processed_middle:
                            continue
                        processed_middle = True
                    
                    # 读取GT mask
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is None:
                        continue
                    
                    # 获取预测mask
                    pred_mask = all_masks[original_idx]
                    
                    # 调整预测mask大小到GT mask大小
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(
                            pred_mask.astype(np.float32),
                            (gt_mask.shape[1], gt_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # 二值化
                    pred_binary = (pred_mask > 0.5).astype(np.uint8)
                    gt_binary = (gt_mask > 127).astype(np.uint8)
                    
                    # 计算指标
                    iou, dice = compute_metrics(pred_binary, gt_binary)
                    
                    total_iou += iou
                    total_dice += dice
                    num_slices += 1
                
            except Exception as e:
                print(f"Error processing case {case_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            finally:
                # 清理临时目录
                temp_video_dir_part1 = os.path.join(temp_base_dir, f"{case_num}_part1")
                temp_video_dir_part2 = os.path.join(temp_base_dir, f"{case_num}_part2")
                if os.path.exists(temp_video_dir_part1):
                    shutil.rmtree(temp_video_dir_part1, ignore_errors=True)
                if os.path.exists(temp_video_dir_part2):
                    shutil.rmtree(temp_video_dir_part2, ignore_errors=True)
    
    finally:
        # 清理临时基目录
        if os.path.exists(temp_base_dir):
            shutil.rmtree(temp_base_dir, ignore_errors=True)
    
    avg_iou = total_iou / num_slices if num_slices > 0 else 0.0
    avg_dice = total_dice / num_slices if num_slices > 0 else 0.0
    
    return avg_iou, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Test MedSAM2 on 2D medical images using video mode")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["kits", "pancreas", "lits", "colon", "local"],
        help="Dataset name"
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
    
    sam2_model = build_sam2_video_predictor(
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
    
    print(f"Test dataset: {dataset_dir}")
    
    # 获取annotation路径
    annotation_path = os.path.join(dataset_dir, "annotation_dict_middle.json")
    
    # 评估模型
    print(f"\n{'='*50}")
    print(f"Evaluating on {args.data} dataset using video mode")
    print(f"{'='*50}")
    
    avg_iou, avg_dice = eval_model_video(
        sam2_model,
        dataset_dir,
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
        "mode": "video",
        "checkpoint": checkpoint_path,
        "avg_iou": float(avg_iou),
        "avg_dice": float(avg_dice),
    }
    
    results_dir = os.path.join(medsam2_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{args.data}_video.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
