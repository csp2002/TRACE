# -*- coding: utf-8 -*-
#csp: MedSAM2 [fixed] neighbor slice box prompt
"""
usage example:
python MedSAM2_neighbor_infer.py --data kits

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import io, transform
import argparse
import cv2
import json


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )



def compute_metrics(pred, target, smooth=1e-6):
    # 确保输入为一维数组
    pred = pred.flatten()
    target = target.flatten()
    
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
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(root_dir, "CT")
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

@torch.no_grad()
def medsam2_inference(image_predictor, img_rgb, box):
    """
    使用 MedSAM2 进行推理
    
    Args:
        image_predictor: SAM2ImagePredictor 实例
        img_rgb: RGB 格式的图像，HWC 格式，像素值在 [0, 255]
        box: 边界框，格式为 [x0, y0, x1, y1]
    
    Returns:
        medsam_seg: 分割结果，uint8 格式的二进制掩码
    """
    # 设置图像
    image_predictor.set_image(img_rgb)
    
    # 预测掩码
    # box 格式为 [x0, y0, x1, y1]，需要转换为 (1, 4) 的形状
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

def eval(image_paths, mask_paths, image_predictor, save_dir, annotation_path):
    total_iou = 0.0
    total_dice = 0.0
    with open(annotation_path, 'r') as f:
        annotation_dict = json.load(f)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        mask_path =mask_paths[i]
        # print("image:"+image_path)
        # print("mask:"+mask_path)
        img_np = io.imread(image_path)
        # print("img_np:"+str(img_np.shape))
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2]==4:
            img_3c = img_np[:,:,:3]
        else:
            img_3c = img_np
        
        # 确保图像是 uint8 格式，像素值在 [0, 255]
        if img_3c.dtype != np.uint8:
            img_3c = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            ) * 255.0
            img_3c = img_3c.astype(np.uint8)
        
        H, W, _ = img_3c.shape
        
        # 读取掩码
        mask = cv2.imread(mask_path)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        case_num, slice_num = mask_path.split('/')[-2:]
        # folder_name = os.path.dirname(mask_path)
        
        # ========== [CHANGED] neighbor JSON per-slice lookup ==========
        if case_num not in annotation_dict:
            print(f"[WARN] Patient '{case_num}' not found in neighbor json: {annotation_path}. Will fall back to current slice.")
            ref_image_path = None
            ref_mask_path = None
        elif slice_num not in annotation_dict[case_num]:
            print(f"[WARN] Slice '{slice_num}' not found in neighbor json for patient '{case_num}'. Will fall back to current slice.")
            ref_image_path = None
            ref_mask_path = None
        else:
            entry = annotation_dict[case_num][slice_num]
            ref_mask_path = entry.get("ref_mask_path", None)
            ref_image_path = entry.get("ref_ct_path", None)
            
            if (not ref_mask_path) or (not os.path.exists(ref_mask_path)):
                print(f"[WARN] neighbor-slice mask not found: {ref_mask_path}. Will fall back to current slice.")
                ref_mask_path = None
            if (not ref_image_path) or (not os.path.exists(ref_image_path)):
                print(f"[WARN] neighbor-slice image not found: {ref_image_path}. Will fall back to current slice.")
                ref_image_path = None
        
        # 从参考掩码中提取边界框
        if ref_mask_path and os.path.exists(ref_mask_path):
            annotation_prompt = cv2.imread(ref_mask_path)
            if len(annotation_prompt.shape) == 3:
                annotation_prompt = cv2.cvtColor(annotation_prompt, cv2.COLOR_BGR2GRAY)
            annotation_prompt = cv2.resize(annotation_prompt, (W, H), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(annotation_prompt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_prompt = np.array([x, y, x+w, y+h])
            else:
                print("Error! Cannot find the rectangle of the annotation:"+ ref_mask_path)
                continue
        else:
            # 如果参考掩码不存在，使用当前掩码
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_prompt = np.array([x, y, x+w, y+h])
            else:
                print("Error! Cannot find the rectangle of the mask:"+ mask_path)
                continue

        # 使用 MedSAM2 进行推理
        medsam_seg = medsam2_inference(image_predictor, img_3c, box_prompt)
        # print("res:"+str(medsam_seg.shape)+str(medsam_seg.max())+str(medsam_seg.min()))  #(512,512) 0 - 1
        # print("mask:"+str(mask.shape)+str(mask.max())+str(mask.min()))  #(512,512) 0 - 255
        iou, dice = compute_metrics(medsam_seg, mask.astype(bool))
        # print("iou:"+str(iou))
        # print("dice:"+str(dice))
        # raise Exception
        # io.imsave(
        #     join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
        #     medsam_seg,
        #     check_contrast=False,
        # )
        total_iou += iou
        total_dice += dice

        # parent_dir = mask_path.split('/')[0]
        # print('save_dir:'+save_dir)
        # print('mask_path:'+mask_path)

        # result_mask_path = mask_path.replace('.', save_dir)
        # # print("result_mask_path:"+result_mask_path)
        # # raise Exception
        # result_mask_folder = os.path.dirname(result_mask_path)
        # os.makedirs(result_mask_folder, exist_ok=True)
        # cv2.imwrite(result_mask_path, medsam_seg * 255)
        # print("Masks have been saved to:", result_mask_path)
        # raise Exception
    avg_iou = total_iou / len(image_paths)
    avg_dice = total_dice / len(image_paths)
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")



if __name__ == "__main__":
    device = "cuda"
    
    # MedSAM2 配置和模型路径
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon","local"]
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help='model config',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/MedSAM2_latest.pt",
        help='checkpoint path',
    )
    args = parser.parse_args()
    
    # 构建 MedSAM2 模型
    sam2_model = build_sam2(
        config_file=args.cfg,
        ckpt_path=args.checkpoint,
        device=device,
        mode="eval"
    )
    image_predictor = SAM2ImagePredictor(sam2_model)
    
    dataset_dir = os.path.join('./2D_data',args.data,'test')
    # image_paths, mask_paths = get_image_mask_paths_TCIA(dataset_dir)
    # image_paths, mask_paths = get_image_mask_paths_local(dataset_dir)
    # image_paths, mask_paths = get_image_mask_paths_openkbp(dataset_dir)
    # image_paths, mask_paths, _, _ = get_image_mask_paths_kits(dataset_dir)
    # image_paths, mask_paths, _, _ = get_image_mask_paths_lits(dataset_dir)
    image_paths, mask_paths = get_image_mask_paths(dataset_dir)
    save_dir = 'MedSAM2_box_results'
    annotation_path = os.path.join('./2D_data',args.data, 'test','annotation_dict_neighbor.json')
    eval(image_paths, mask_paths, image_predictor, save_dir, annotation_path)

    

# # %% visualize results
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img_3c)
# show_box(box_np[0], ax[0])
# ax[0].set_title("Input Image and Bounding Box")
# ax[1].imshow(img_3c)
# show_mask(medsam_seg, ax[1])
# show_box(box_np[0], ax[1])
# ax[1].set_title("MedSAM2 Segmentation")
# plt.show()
