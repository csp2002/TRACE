# -*- coding: utf-8 -*-
#csp: MedSAM [fixed] middle slice box prompt
"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import cv2
import argparse
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


def get_image_mask_paths_TCIA(root_dir):
    image_paths = []
    mask_paths = []
        
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
            
        for organ in os.listdir(patient_path):
            if 'MR' not in organ:   #delete '*skull'
                organ_folder = os.path.join(patient_path, organ)
                for slicename in os.listdir(organ_folder):
                    mask_path = os.path.join(organ_folder, slicename)
                    ct_folder = str(patient) + '_MR_t1'
                    ct_path = os.path.join(patient_path,ct_folder,slicename)
                        
                    if os.path.exists(mask_path) and os.path.exists(ct_path):
                        image_paths.append(ct_path)
                        mask_paths.append(mask_path)
                            # print("image:"+ ct_path)
                            # print("masks:"+ mask_path)
                    else:
                        print("Cannot find path:" + ct_path + " or:" +mask_path)
    if len(image_paths)==len(mask_paths):
        return image_paths, mask_paths
    else:
        print("The length of images and masks are not equal!")


def get_image_mask_paths_local(root_dir):
    image_paths = []
    mask_paths = []
    ct_dir = os.path.join(root_dir, "CT")
    mask_dir = os.path.join(root_dir, "Mask")
    
    for patient_folder in os.listdir(ct_dir):
        patient_ct_folder = os.path.join(ct_dir, patient_folder)
        patient_mask_folder = os.path.join(mask_dir, patient_folder)
        
        for organ_folder in os.listdir(patient_ct_folder):
            if '*' not in organ_folder:   #delete '*skull'
                organ_ct_folder = os.path.join(patient_ct_folder, organ_folder)
                organ_mask_folder = os.path.join(patient_mask_folder, organ_folder)
                
                for ct_filename in os.listdir(organ_ct_folder):
                    ct_path = os.path.join(organ_ct_folder, ct_filename)
                    mask_filename = ct_filename.replace('ct', 'mask')
                    mask_path = os.path.join(organ_mask_folder, mask_filename)
                    
                    if os.path.exists(mask_path):
                        image_paths.append(ct_path)
                        mask_paths.append(mask_path)
    
    return image_paths, mask_paths

def get_image_mask_paths_openkbp(root_dir):
    image_paths = []
    mask_paths = []
    for patient_id in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient_id)
        ct_dir = os.path.join(patient_dir, 'CT')
        ptv_dir = os.path.join(patient_dir, 'PTV70')
        
        if not os.path.isdir(ct_dir) or not os.path.isdir(ptv_dir):
            continue
        
        ct_slices = sorted(os.listdir(ct_dir))
        ptv_slices = sorted(os.listdir(ptv_dir))
        
        assert len(ct_slices) == len(ptv_slices), f"Slice count mismatch in {patient_id}"
        
        for ct_slice, ptv_slice in zip(ct_slices, ptv_slices):
            ct_path = os.path.join(ct_dir, ct_slice)
            ptv_path = os.path.join(ptv_dir, ptv_slice)
            image_paths.append(ct_path)
            mask_paths.append(ptv_path)
    return image_paths, mask_paths

def get_image_mask_paths_kits(root_dir):
    image_paths = []
    mask_paths = []
    ct_dirs = []
    mask_dirs = []
    current_dir = os.getcwd()   
    root_dir = os.path.join(current_dir, root_dir)
    for patient_id in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient_id)
        ct_dir = os.path.join(patient_dir, 'CT')
        mask_dir = os.path.join(patient_dir, 'Mask')
        ct_dirs.append(ct_dir)
        mask_dirs.append(mask_dir)
        
        
        
        ct_slices = sorted(os.listdir(ct_dir),key=lambda x: int(x.split('.')[0]))
        mask_slices = sorted(os.listdir(mask_dir),key=lambda x: int(x.split('.')[0]))
        
        assert len(ct_slices) == len(mask_slices), f"Slice count mismatch in {patient_id}"
        
        for ct_slice, mask_slice in zip(ct_slices, mask_slices):
            ct_path = os.path.join(ct_dir, ct_slice)
            mask_path = os.path.join(mask_dir, mask_slice)
            image_paths.append(ct_path)
            mask_paths.append(mask_path)
    return image_paths, mask_paths, ct_dirs, mask_dirs


def get_image_mask_paths_lits(root_dir):
    image_paths = []
    mask_paths = []
    ct_dirs = []
    mask_dirs = []
    current_dir = os.getcwd()   
    root_dir = os.path.join(current_dir, root_dir)
    for patient_id in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient_id)
        ct_dir = os.path.join(patient_dir, 'CT')
        mask_dir = os.path.join(patient_dir, 'Mask')
        ct_dirs.append(ct_dir)
        mask_dirs.append(mask_dir)
        
        
        
        ct_slices = sorted(os.listdir(ct_dir),key=lambda x: int(x.split('.')[0]))
        mask_slices = sorted(os.listdir(mask_dir),key=lambda x: int(x.split('.')[0]))
        
        assert len(ct_slices) == len(mask_slices), f"Slice count mismatch in {patient_id}"
        
        for ct_slice, mask_slice in zip(ct_slices, mask_slices):
            ct_path = os.path.join(ct_dir, ct_slice)
            mask_path = os.path.join(mask_dir, mask_slice)
            image_paths.append(ct_path)
            mask_paths.append(mask_path)
    return image_paths, mask_paths, ct_dirs, mask_dirs

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
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def eval(image_paths, mask_paths, medsam_model, save_dir, annotation_path):
    total_iou = 0.0
    total_dice = 0.0

    with open(annotation_path, 'r') as f:
        annotation_dict = json.load(f)

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        mask_path = mask_paths[i]

        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_3c = img_np[:, :, :3]
        else:
            img_3c = img_np

        H, W, _ = img_3c.shape
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        mask = cv2.imread(mask_path)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        case_num, slice_num = mask_path.split('/')[-2:]   # patient, filename

        # ========== [CHANGED] neighbor JSON per-slice lookup ==========
        if case_num not in annotation_dict:
            raise KeyError(f"[FATAL] Patient '{case_num}' not found in neighbor json: {annotation_path}")

        if slice_num not in annotation_dict[case_num]:
            raise KeyError(
                f"[FATAL] Slice '{slice_num}' not found in neighbor json for patient '{case_num}'.\n"
                f"  mask_path={mask_path}\n"
                f"  json={annotation_path}"
            )

        entry = annotation_dict[case_num][slice_num]
        ref_mask_path  = entry.get("ref_mask_path", None)
        ref_image_path = entry.get("ref_ct_path", None)

        if (not ref_mask_path) or (not os.path.exists(ref_mask_path)):
            raise FileNotFoundError(
                f"[FATAL] ref_mask_path missing or not exists.\n"
                f"  patient={case_num}, slice={slice_num}\n"
                f"  ref_mask_path={ref_mask_path}\n"
                f"  json={annotation_path}"
            )
        if (not ref_image_path) or (not os.path.exists(ref_image_path)):
            raise FileNotFoundError(
                f"[FATAL] ref_ct_path missing or not exists.\n"
                f"  patient={case_num}, slice={slice_num}\n"
                f"  ref_ct_path={ref_image_path}\n"
                f"  json={annotation_path}"
            )

        annotation_prompt = cv2.imread(ref_mask_path)
        if len(annotation_prompt.shape) == 3:
            annotation_prompt = cv2.cvtColor(annotation_prompt, cv2.COLOR_BGR2GRAY)

        annotation_prompt = cv2.resize(
            annotation_prompt, (mask.shape[0], mask.shape[1]), interpolation=cv2.INTER_NEAREST
        )

        contours, _ = cv2.findContours(annotation_prompt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            box_prompt = np.array([[x, y, x + w, y + h]])
        else:
            raise RuntimeError(
                f"[FATAL] Cannot find contour for reference mask -> bbox.\n"
                f"  ref_mask_path={ref_mask_path}\n"
                f"  ref_ct_path={ref_image_path}"
            )

        box_1024 = box_prompt / np.array([W, H, W, H]) * 1024

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        medsam_seg = medsam_inference(medsam_model, image_embedding[0], box_1024, H, W)

        iou, dice = compute_metrics(medsam_seg, mask.astype(bool))
        total_iou += iou
        total_dice += dice

    avg_iou = total_iou / len(image_paths)
    avg_dice = total_dice / len(image_paths)
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")




if __name__ == "__main__":
    device = "cuda"
    medsam_model = sam_model_registry["vit_b"](checkpoint="MedSAM/medsam_vit_b.pth")
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    # raise Exception
    # dataset_dir = 'TCIA_Seg_testset'
    # dataset_dir = 'local_Seg_SMet_testset'
    # dataset_dir = 'openkbp_seg/test-pats'
    # dataset_dir = 'kits19/seg_testset2'
    # dataset_dir = 'LiTS/seg_testset2'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon","local"]
    )
    args = parser.parse_args()
    dataset_dir = os.path.join('./2D_data',args.data,'test')
    # image_paths, mask_paths = get_image_mask_paths_TCIA(dataset_dir)
    # image_paths, mask_paths = get_image_mask_paths_local(dataset_dir)
    # image_paths, mask_paths = get_image_mask_paths_openkbp(dataset_dir)
    # image_paths, mask_paths, _, _ = get_image_mask_paths_kits(dataset_dir)
    # image_paths, mask_paths, _, _ = get_image_mask_paths_lits(dataset_dir)
    image_paths, mask_paths = get_image_mask_paths(dataset_dir)
    save_dir = 'MedSAM_box_results'
    annotation_path = os.path.join('./2D_data',args.data, 'test','annotation_dict_neighbor.json')
    eval(image_paths, mask_paths, medsam_model, save_dir, annotation_path)

    

# # %% visualize results
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img_3c)
# show_box(box_np[0], ax[0])
# ax[0].set_title("Input Image and Bounding Box")
# ax[1].imshow(img_3c)
# show_mask(medsam_seg, ax[1])
# show_box(box_np[0], ax[1])
# ax[1].set_title("MedSAM Segmentation")
# plt.show()
