import os, json
import sys
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2
from skimage import io, transform

import argparse
import matplotlib.pyplot as plt

from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import TransUNet_ours as TransUNet_ours
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .networks.medformer import MedFormer,MedFormer_ours
from .networks.attention_unet import AttentionUNet,AttentionUNet_ours
from .networks.unetpp import UNetPlusPlus,UNetPlusPlus_ours
from .networks.swin_unet import SwinUnet,SwinUnet_ours
from .networks.swin_unet import SwinUnet_config
from .networks.FAT_Net import FAT_Net,FATNet_ours
from .networks.H2Former import res34_swin_MS,H2Former_ours

import torch.nn as nn


# ====================== utils =============================

def _resolve_ckpt(path):
    """Prefer a val-selected best_model.pth in the same directory; fall back to
    the given epoch checkpoint (mirrors tumor_seg/test.py's resolution)."""
    best = os.path.join(os.path.dirname(path), 'best_model.pth')
    return best if os.path.exists(best) else path


def dice_coef(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Binary Dice. pred/gt shape: (1,H,W), values in {0,1}."""
    inter = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    return (2 * inter + eps) / (union + eps)


def _prep_img(path: str, out: int = 224) -> np.ndarray:
    """Traditional model image preprocessing: resize to out x out, normalize to [0,1]"""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        print(f"Warning: {path} is all zeros.")
        arr = arr * 0.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=3)  # bicubic


def _prep_msk(path: str, out: int = 224) -> np.ndarray:
    """Traditional model mask preprocessing: resize to out x out, normalize to [0,1]"""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=0)  # nearest


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H,W) -> (1,1,H,W)"""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


# ====================== MedSAM utils =============================

def medsam_prep_img(path: str) -> Tuple[np.ndarray, int, int]:
    """
    MedSAM image preprocessing: resize to 1024x1024, normalize to [0,1], convert to RGB
    Returns: (img_1024_tensor, H, W) where H, W are original dimensions
    """
    img_np = io.imread(path)
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
    )  # normalize to [0, 1]
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, 1024, 1024)
    return img_1024_tensor, H, W


def medsam_extract_box_from_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Extract bounding box from mask using cv2.findContours (MedSAM style)
    Returns: box_prompt as [x, y, x+w, y+h]
    """
    if len(mask_np.shape) == 3:
        mask = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_np
    
    # Ensure binary mask
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = (mask > 0.5).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        box_prompt = np.array([[x, y, x+w, y+h]])
    else:
        # If no contour found, return full image box
        H, W = mask.shape
        box_prompt = np.array([[0, 0, W, H]])
    return box_prompt


# MedSAM wrapper class (for model1)
class MedSAM_Wrapper(nn.Module):
    """Wrapper for standard MedSAM model (model1)"""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # Ensure box_torch is (B, 1, 4) format
            if len(box_torch.shape) == 1:
                # (4,) -> (1, 1, 4)
                box_torch = box_torch.unsqueeze(0).unsqueeze(0)
            elif len(box_torch.shape) == 2:
                # (B, 4) -> (B, 1, 4)
                box_torch = box_torch[:, None, :]
            # If already (B, 1, 4), keep as is

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


# ====================== MedSAM2 utils =============================

def medsam2_prep_img(path: str) -> Tuple[np.ndarray, int, int]:
    """
    MedSAM2 image preprocessing: keep original size, convert to RGB uint8 [0,255]
    Returns: (img_rgb, H, W) where H, W are original dimensions
    """
    img_np = io.imread(path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_3c = img_np[:, :, :3]
    else:
        img_3c = img_np
    
    # Ensure uint8 format, pixel values in [0, 255]
    if img_3c.dtype != np.uint8:
        img_3c = (img_3c - img_3c.min()) / np.clip(
            img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
        ) * 255.0
        img_3c = img_3c.astype(np.uint8)
    
    H, W, _ = img_3c.shape
    return img_3c, H, W


def medsam2_extract_box_from_mask(mask_np: np.ndarray, image_size: int = 512) -> np.ndarray:
    """
    Extract bounding box from mask using np.where (MedSAM2 style)
    Returns: box_prompt as [x_min, y_min, x_max, y_max] in original image coordinates
    """
    # Ensure binary mask
    if mask_np.max() > 1.0:
        mask_binary = (mask_np > 127).astype(np.uint8)
    else:
        mask_binary = (mask_np > 0.5).astype(np.uint8)
    
    H, W = mask_binary.shape
    
    # Resize mask to image_size for box extraction
    mask_resized = cv2.resize(mask_binary.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    y_indices, x_indices = np.where(mask_resized > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        # If mask is empty, return full image box
        return np.array([0, 0, W, H])
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Map bbox from image_size scale back to original image scale
    x_min = int(x_min * W / image_size)
    x_max = int(x_max * W / image_size)
    y_min = int(y_min * H / image_size)
    y_max = int(y_max * H / image_size)
    
    # Ensure bbox is within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    
    box_prompt = np.array([x_min, y_min, x_max, y_max])
    return box_prompt


@torch.no_grad()
def medsam2_inference_direct(model, img_rgb, box, H, W, device):
    """
    MedSAM2 inference function (direct call without predictor)
    Args:
        model: SAM2Base model instance
        img_rgb: RGB image (H, W, 3), uint8, pixel values in [0, 255]
        box: bounding box [x_min, y_min, x_max, y_max] in original image coordinates
        H, W: original image dimensions
        device: device to run on
    Returns:
        medsam_seg: binary mask (H, W)
    """
    # Create transforms
    transforms = SAM2Transforms(
        resolution=model.image_size,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0
    )
    
    # Transform image
    input_image = transforms(img_rgb)  # (3, image_size, image_size)
    input_image = input_image[None, ...].to(device)  # (1, 3, image_size, image_size)
    
    # Get image embeddings
    backbone_out = model.forward_image(input_image)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
    
    # NOTE: no_mem_embed is NOT added here to stay consistent with training
    # (train_medsam2_2d.py forward_with_box does not add no_mem_embed)

    # Prepare features
    hires_size = model.image_size // 4
    bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
    ][::-1]
    image_embed = feats[-1]  # (1, C, H, W)
    high_res_feats = feats[:-1]
    
    # Transform box coordinates
    box_tensor = torch.as_tensor(box, dtype=torch.float, device=device)
    if len(box_tensor.shape) == 1:
        box_tensor = box_tensor.unsqueeze(0)  # (1, 4)
    
    # Normalize box to [0, 1] and then scale to image_size
    unnorm_box = transforms.transform_boxes(box_tensor, normalize=True, orig_hw=(H, W))  # (1, 2, 2)
    
    # Convert box to point format for prompt encoder
    box_coords = unnorm_box.reshape(-1, 2, 2)  # (1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(box_coords.size(0), 1)  # (1, 2)
    
    # Encode prompts
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels),
        boxes=None,
        masks=None,
    )
    
    # Prepare high res features
    high_res_features = [feat_level[0].unsqueeze(0) for feat_level in high_res_feats]
    
    # Decode masks
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    
    # Postprocess masks to original size
    masks = transforms.postprocess_masks(low_res_masks, (H, W))  # (1, 1, H, W)
    
    # Convert to binary mask
    medsam_seg = (masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8)  # (H, W)
    
    return medsam_seg


# ==================== core tester =========================

class DoctorSimTester:
    """
    data_root structure:
        data_root/
            └─ CT/   patientX/ 000.png, 001.png, ...
            └─ Mask/ patientX/ 000.png, 001.png, ...
    """

    def __init__(
        self,
        data_root: str,
        model1,
        model2,
        threshold: float,
        save_root: str,
        device: str = "cuda",
        img_size: int = 224,
        exclude_patients: List[str] = None,
        ai_mode: str = "B",   # 'A' or 'B'
        model_type: str = "traditional",  # 'traditional', 'medsam', 'medsam2'
    ):
        self.root = data_root
        self.m1 = model1.eval().to(device)
        self.m2 = model2.eval().to(device)
        self.th = float(threshold)
        self.save_root = save_root
        self.dev = device
        self.size = img_size
        self.exclude_patients = exclude_patients or []
        self.ai_mode = ai_mode.upper()
        self.model_type = model_type.lower()
        assert self.ai_mode in ["A", "B"], f"ai_mode must be 'A' or 'B', got {ai_mode}"
        assert self.model_type in ["traditional", "medsam", "medsam2"], f"model_type must be 'traditional', 'medsam', or 'medsam2', got {model_type}"
        
        # For MedSAM, wrap model1 with MedSAM_Wrapper
        if self.model_type == "medsam":
            # Model1 should be wrapped with MedSAM_Wrapper (will be done in model loading)
            pass

    # ---------------------------- public --------------------
    def run(self) -> Dict:
        """Traverse all volumes and summarize stats."""
        os.makedirs(self.save_root, exist_ok=True)

        ct_root = os.path.join(self.root, "CT")
        patient_dirs = sorted(
            [d for d in os.listdir(ct_root) if os.path.isdir(os.path.join(ct_root, d))]
        )

        # exclude patients
        exclude = self.exclude_patients or []
        if exclude:
            before = len(patient_dirs)
            patient_dirs = [p for p in patient_dirs if p not in set(exclude)]
            after = len(patient_dirs)
            print(f"[Mode {self.ai_mode} | th={self.th:.2f}] Excluded patients {exclude}: {before} -> {after} patients")

        all_stats = {}
        total_fix = 0
        first_manual = 0
        total_slices = 0
        total_dice_initial = 0.0
        total_dice_final = 0.0

        for pid in tqdm(patient_dirs, desc=f"Patients (Mode {self.ai_mode}, th={self.th:.2f})"):
            stat = self._process_volume(pid)
            all_stats[pid] = stat

            total_fix += stat["manual_corrections"]
            total_slices += stat["n_slices"]
            first_manual += stat["first_manual"]
            total_dice_initial += stat["avg_initial_dice"] * stat["n_slices"]
            total_dice_final += stat["avg_final_dice"] * stat["n_slices"]

        if total_slices == 0:
            avg_dice_initial = 0.0
            avg_dice_final = 0.0
        else:
            avg_dice_initial = total_dice_initial / total_slices
            avg_dice_final = total_dice_final / total_slices

        summary = {
            "ai_mode": self.ai_mode,
            "threshold": self.th,
            "excluded_patients": exclude,
            "total_manual_corrections": int(total_fix),
            "first_manual_corrections": int(first_manual),
            "total_slices": int(total_slices),
            "avg_dice_initial": float(avg_dice_initial),
            "avg_dice_final": float(avg_dice_final),
            "per_patient": all_stats,
        }

        with open(os.path.join(self.save_root, "sim_stats.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if total_slices > 0:
            rr = total_fix / total_slices
        else:
            rr = 0.0

        print(
            f"\n>>> DONE (Mode {self.ai_mode}, th={self.th:.2f}). "
            f"rejects = {total_fix} / {total_slices} ({rr:.2%}), "
            f"first_manual = {first_manual} / {len(patient_dirs)} volumes, "
            f"avg_initial_dice = {avg_dice_initial:.4f}, "
            f"avg_final_dice = {avg_dice_final:.4f} <<<\n"
        )
        print(f"Results saved to {self.save_root}/sim_stats.json")
        print("Predicted masks are saved in:", os.path.join(self.save_root, "predictions"))

        return summary

    # ------------------------ per-volume --------------------
    def _process_volume(self, pid: str) -> Dict:
        ct_dir = os.path.join(self.root, "CT", pid)
        mk_dir = os.path.join(self.root, "Mask", pid)

        # sort by integer stem, filenames like 000.png
        slices = sorted(
            [f for f in os.listdir(ct_dir) if f.lower().endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        n = len(slices)
        if n == 0:
            return {
                "n_slices": 0,
                "manual_corrections": 0,
                "avg_initial_dice": 0.0,
                "avg_final_dice": 0.0,
                "first_manual": 0,
            }

        center = n // 2  # if even, take the later one

        final_masks: List[np.ndarray] = [None] * n
        initial_masks: List[np.ndarray] = [None] * n
        accepted_status: List[bool] = [False] * n  # Track accept/reject status for each slice (for option 3)

        manual = 0
        first_manual = 0

        # ---- Mode A: every slice uses model1(img) ----
        if self.ai_mode == "A":
            for idx in range(n):
                _, acc = self._infer_slice(
                    idx=idx,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m1,
                    need_ref=False,
                    ref_idx=None,
                    final_masks=final_masks,
                    initial_masks=initial_masks,
                    accepted_status=accepted_status,
                    pid=pid,
                    slices=slices,
                )
                accepted_status[idx] = acc
                if not acc:
                    manual += 1

            # keep old meaning of first_manual: whether center slice is rejected
            # All model types now store masks at ORIGINAL resolution
            center_gt = np.array(Image.open(os.path.join(mk_dir, slices[center])).convert("L"), dtype=np.float32) / 255.0

            center_pred = initial_masks[center]
            center_dice = dice_coef(
                torch.from_numpy(center_pred).unsqueeze(0),
                torch.from_numpy(center_gt).unsqueeze(0),
            )
            first_manual = int(center_dice <= self.th)

        # ---- Mode B: center by model1, others by model2 with reference ----
        else:
            _, acc = self._infer_slice(
                idx=center,
                ct_dir=ct_dir,
                mk_dir=mk_dir,
                model=self.m1,
                need_ref=False,
                ref_idx=None,
                final_masks=final_masks,
                initial_masks=initial_masks,
                accepted_status=accepted_status,
                pid=pid,
                slices=slices,
            )
            accepted_status[center] = acc
            if not acc:
                manual += 1
                first_manual += 1

            # Process right side first (center+1 to n-1)
            for r in range(center + 1, n):
                ref = r - 1
                _, acc_r = self._infer_slice(
                    idx=r,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m2,
                    need_ref=True,
                    ref_idx=ref,
                    final_masks=final_masks,
                    initial_masks=initial_masks,
                    accepted_status=accepted_status,
                    pid=pid,
                    slices=slices,
                )
                accepted_status[r] = acc_r
                if not acc_r:
                    manual += 1

            # Process left side (center-1 to 0)
            for l in range(center - 1, -1, -1):
                ref = l + 1
                _, acc_l = self._infer_slice(
                    idx=l,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m2,
                    need_ref=True,
                    ref_idx=ref,
                    final_masks=final_masks,
                    initial_masks=initial_masks,
                    accepted_status=accepted_status,
                    pid=pid,
                    slices=slices,
                )
                accepted_status[l] = acc_l
                if not acc_l:
                    manual += 1

        # averages -- all model types now store masks at ORIGINAL resolution
        avg_initial_dice = float(
            np.mean(
                [
                    dice_coef(
                        torch.from_numpy(mk).unsqueeze(0),
                        torch.from_numpy(np.array(Image.open(os.path.join(mk_dir, s)).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0),
                    )
                    for mk, s in zip(initial_masks, slices)
                ]
            )
        )
        avg_final_dice = float(
            np.mean(
                [
                    dice_coef(
                        torch.from_numpy(mk).unsqueeze(0),
                        torch.from_numpy(np.array(Image.open(os.path.join(mk_dir, s)).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0),
                    )
                    for mk, s in zip(final_masks, slices)
                ]
            )
        )

        return {
            "n_slices": n,
            "manual_corrections": manual,
            "avg_initial_dice": avg_initial_dice,
            "avg_final_dice": avg_final_dice,
            "first_manual": first_manual,
        }

    # ---------- helper: run inference + doctor decision -----
    def _infer_slice(
        self,
        idx: int,
        ct_dir: str,
        mk_dir: str,
        model,
        need_ref: bool,
        ref_idx: int,
        final_masks: List[np.ndarray],
        initial_masks: List[np.ndarray],
        accepted_status: List[bool],
        pid: str,
        slices: List[str],
    ) -> Tuple[float, bool]:
        sl_name = slices[idx]
        ct_path = os.path.join(ct_dir, sl_name)
        mk_path = os.path.join(mk_dir, sl_name)

        # Get GT mask at ORIGINAL resolution (for all model types)
        # Traditional model still needs the 224x224 image for inference input,
        # but metrics are always computed at original resolution.
        gt_np = np.array(Image.open(mk_path).convert("L"), dtype=np.float32) / 255.0
        gt_t = torch.from_numpy(gt_np).unsqueeze(0).to(self.dev)  # (1, H_orig, W_orig)

        if self.model_type == "traditional":
            # Traditional model inference (runs at self.size, e.g. 224x224)
            img_np = _prep_img(ct_path, self.size)
            img_t = _to_tensor(img_np).to(self.dev)

            orig_H, orig_W = gt_np.shape  # original resolution from GT

            if need_ref:
                ref_name = slices[ref_idx]
                ref_ct = os.path.join(ct_dir, ref_name)

                ref_img_np = _prep_img(ref_ct, self.size)

                # Reference mask: prediction if ref slice was accepted, else GT.
                if accepted_status[ref_idx]:
                    ref_mk_np = initial_masks[ref_idx]
                else:
                    ref_mk_np = final_masks[ref_idx]

                # ref_mk_np is stored at original resolution; resize to self.size for model input
                if ref_mk_np.shape != (self.size, self.size):
                    ref_mk_np_224 = zoom(
                        ref_mk_np,
                        (self.size / ref_mk_np.shape[0], self.size / ref_mk_np.shape[1]),
                        order=0,
                    )
                else:
                    ref_mk_np_224 = ref_mk_np

                ref_img_t = _to_tensor(ref_img_np).to(self.dev)
                ref_mk_t = _to_tensor(ref_mk_np_224).to(self.dev)

                with torch.no_grad():
                    out = model(img_t, ref_img_t, ref_mk_t)
                    logit = out["final"] if isinstance(out, dict) and "final" in out else out
            else:
                with torch.no_grad():
                    logit = model(img_t)

            pred = torch.argmax(F.softmax(logit, dim=1), dim=1).float()  # (1, self.size, self.size)
            pred_np_224 = pred.squeeze(0).detach().cpu().numpy()

            # Upsample prediction from self.size to original resolution for metric computation & storage
            if (orig_H, orig_W) != (self.size, self.size):
                pred_np = zoom(
                    pred_np_224,
                    (orig_H / self.size, orig_W / self.size),
                    order=0,
                ).astype(np.float32)
                pred_np = (pred_np > 0.5).astype(np.float32)
            else:
                pred_np = pred_np_224.astype(np.float32)

            # Compute dice at ORIGINAL resolution
            pred_t_orig = torch.from_numpy(pred_np).unsqueeze(0).to(self.dev)
            dice = dice_coef(pred_t_orig, gt_t)

        elif self.model_type == "medsam":
            # MedSAM inference
            img_1024_tensor, H, W = medsam_prep_img(ct_path)
            img_1024_tensor = img_1024_tensor.to(self.dev)
            
            # Get box prompt
            if need_ref:
                # Reference mask for box_prompt: prediction if ref slice was accepted, else GT.
                if accepted_status[ref_idx]:
                    ref_mask_for_box = initial_masks[ref_idx]
                else:
                    ref_mask_for_box = final_masks[ref_idx]
                ref_mask_uint8 = (ref_mask_for_box * 255).astype(np.uint8)

                box_prompt = medsam_extract_box_from_mask(ref_mask_uint8)
                
                # MedSAM_with_TRACE (model2): direct forward call
                ref_name = slices[ref_idx]
                ref_ct_path = os.path.join(ct_dir, ref_name)
                
                # Prepare reference image (1024x1024)
                ref_img_1024_tensor, _, _ = medsam_prep_img(ref_ct_path)
                ref_img_1024_tensor = ref_img_1024_tensor.to(self.dev)
                
                # Reference mask (1024×1024): prediction if ref slice was accepted, else GT.
                if accepted_status[ref_idx]:
                    ref_mask_np = initial_masks[ref_idx]
                else:
                    ref_mask_np = final_masks[ref_idx]
                
                # Resize ref_mask to 1024x1024
                ref_mask_resized = cv2.resize(ref_mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                ref_gt_tensor = torch.from_numpy(ref_mask_resized).float().unsqueeze(0).unsqueeze(0).to(self.dev)  # (1, 1, 1024, 1024)
                
                # Scale box to 1024x1024
                # box_prompt is [[x, y, x+w, y+h]] in original coordinates
                box_1024 = box_prompt / np.array([W, H, W, H]) * 1024  # (1, 4)
                # MedSAM_with_TRACE expects numpy array format (1, 4) like test_ref_finetune.py
                # Ensure box_np is (1, 4) format, not (4,)
                if len(box_1024.shape) == 2 and box_1024.shape[0] == 1:
                    box_np = box_1024  # (1, 4)
                elif len(box_1024.shape) == 1:
                    box_np = box_1024.reshape(1, 4)  # (4,) -> (1, 4)
                else:
                    # If somehow it's already flattened, reshape it
                    box_np = box_1024.flatten()[:4].reshape(1, 4)  # Ensure (1, 4)
                
                # Forward pass with MedSAM_with_TRACE (like test_ref_finetune.py)
                # boxes_np is (1, 4), MedSAM_with_TRACE.forward will convert to (1, 1, 4)
                with torch.no_grad():
                    outputs = model(img_1024_tensor, box_np, ref_img_1024_tensor, ref_gt_tensor)
                    if isinstance(outputs, dict):
                        logits = outputs["final"]
                    else:
                        logits = outputs
                    
                    # Convert logits to mask
                    pred_mask = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (1, 1024, 1024)
                    pred_np = (pred_mask[0] > 0.5).astype(np.float32)
                    
                    # Resize back to original size
                    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                # Mode A: use center slice GT for all slices
                # Mode B center slice: use its own GT
                if self.ai_mode == "A":
                    # Mode A: all slices use center slice GT
                    # IMPORTANT: First resize center slice mask to current slice size, then extract box
                    # This ensures box coordinates match the current slice dimensions
                    center_idx = len(slices) // 2
                    center_mk_path = os.path.join(mk_dir, slices[center_idx])
                    center_mask_uint8 = np.array(Image.open(center_mk_path).convert("L"))
                    H_center, W_center = center_mask_uint8.shape
                    
                    # Resize center slice mask to current slice dimensions
                    center_mask_resized = cv2.resize(center_mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                    box_prompt = medsam_extract_box_from_mask(center_mask_resized)  # (1, 4) [[x, y, x+w, y+h]]
                    # Scale box to 1024x1024 using current slice dimensions (not center slice dimensions!)
                    # box_prompt is (1, 4), scale factors are (4,), numpy broadcasting will work
                    box_1024 = box_prompt / np.array([W, H, W, H]) * 1024  # (1, 4)
                else:
                    # Mode B center slice: use its own GT
                    current_mask_uint8 = np.array(Image.open(mk_path).convert("L"))
                    box_prompt = medsam_extract_box_from_mask(current_mask_uint8)
                    # Scale box to 1024x1024 using current slice dimensions
                    box_1024 = box_prompt / np.array([W, H, W, H]) * 1024
                
                # Standard MedSAM (model1): use MedSAM_Wrapper forward
                # box_1024 is in format [[x, y, x+w, y+h]] at 1024x1024 scale
                # Convert to (4,) numpy array format like test_vanilla_finetune.py
                box_np = box_1024[0] if len(box_1024.shape) == 2 else box_1024  # (4,)
                with torch.no_grad():
                    ori_res_masks = model(img_1024_tensor, box_np)  # (1, 1, 1024, 1024)
                    pred_mask = torch.sigmoid(ori_res_masks)  # (1, 1, 1024, 1024)
                    pred_mask_np = pred_mask.squeeze().cpu().numpy()  # (1024, 1024)
                    pred_np = (pred_mask_np > 0.5).astype(np.float32)
                    
                    # Resize back to original size
                    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Calculate dice
            pred_t = torch.from_numpy(pred_np).unsqueeze(0).to(self.dev)
            dice = dice_coef(pred_t, gt_t)

        else:  # medsam2
            # MedSAM2 inference
            img_rgb, H, W = medsam2_prep_img(ct_path)
            
            # Get box prompt
            if need_ref:
                # Reference mask for box_prompt: prediction if ref slice was accepted, else GT.
                if accepted_status[ref_idx]:
                    ref_mask_for_box = initial_masks[ref_idx]
                    if ref_mask_for_box.max() <= 1.0:
                        ref_mask_uint8 = (ref_mask_for_box * 255).astype(np.uint8)
                    else:
                        ref_mask_uint8 = ref_mask_for_box.astype(np.uint8)
                else:
                    ref_mask_for_box = final_masks[ref_idx]
                    ref_mask_uint8 = (ref_mask_for_box * 255).astype(np.uint8)
                
                box_prompt = medsam2_extract_box_from_mask(ref_mask_uint8, image_size=512)
            else:
                # Mode A: use center slice GT for all slices
                # Mode B center slice: use its own GT
                if self.ai_mode == "A":
                    # Mode A: all slices use center slice GT
                    center_idx = len(slices) // 2
                    center_mk_path = os.path.join(mk_dir, slices[center_idx])
                    center_mask_uint8 = np.array(Image.open(center_mk_path).convert("L"))
                    box_prompt = medsam2_extract_box_from_mask(center_mask_uint8, image_size=512)
                else:
                    # Mode B center slice: use its own GT
                    current_mask_uint8 = np.array(Image.open(mk_path).convert("L"))
                    box_prompt = medsam2_extract_box_from_mask(current_mask_uint8, image_size=512)
            
            # Check if model is MedSAM2_with_TRACE (model2) or standard MedSAM2 (model1)
            # In Mode B, when need_ref=True, model is model2 (MedSAM2_with_TRACE) which needs direct forward call
            # In Mode A or when need_ref=False, model is model1 (standard MedSAM2) which uses predictor
            if need_ref:
                # Model2 (MedSAM2_with_TRACE): direct forward call
                # MedSAM2_with_TRACE: direct forward call
                ref_name = slices[ref_idx]
                ref_ct_path = os.path.join(ct_dir, ref_name)
                
                # Prepare reference image
                ref_img_rgb, _, _ = medsam2_prep_img(ref_ct_path)
                
                # Reference mask: prediction if ref slice was accepted, else GT.
                if accepted_status[ref_idx]:
                    ref_mask_np = initial_masks[ref_idx]
                else:
                    ref_mask_np = final_masks[ref_idx]
                
                # Resize images and masks to 512x512 for model input
                img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                ref_img_resized = cv2.resize(ref_img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                # Normalize images: [0,255] -> [0,1] -> ImageNet mean/std (consistent with training dataset)
                _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, 512, 512)
                img_tensor = (img_tensor - _mean) / _std
                ref_img_tensor = torch.from_numpy(ref_img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, 512, 512)
                ref_img_tensor = (ref_img_tensor - _mean) / _std
                
                # Resize and prepare reference mask
                ref_mask_resized = cv2.resize(ref_mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
                ref_gt_tensor = torch.from_numpy(ref_mask_resized).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
                
                # Prepare box (scale to 512x512)
                # box_prompt is [x_min, y_min, x_max, y_max] in original coordinates
                box_scaled = box_prompt / np.array([W, H, W, H]) * 512
                # MedSAM2_with_TRACE expects (B, 4) format, not (4,)
                # Ensure box_np is (1, 4) numpy array for batch_size=1
                if isinstance(box_scaled, np.ndarray):
                    box_flat = box_scaled.flatten()[:4]  # (4,)
                else:
                    box_flat = np.array(box_scaled).flatten()[:4]  # (4,)
                box_np = box_flat.reshape(1, 4)  # (1, 4) - batch_size=1
                
                # Move to device
                img_tensor = img_tensor.to(self.dev)
                ref_img_tensor = ref_img_tensor.to(self.dev)
                ref_gt_tensor = ref_gt_tensor.to(self.dev)
                
                # Forward pass (boxes as numpy array (1, 4), like the TRACE eval path)
                with torch.no_grad():
                    output = model(img_tensor, box_np, ref_img_tensor, ref_gt_tensor, image_size=512)
                    if isinstance(output, dict):
                        logits = output["final"]
                    else:
                        logits = output
                    
                    # Convert logits to mask
                    pred_mask = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (1, 512, 512)
                    pred_np = (pred_mask[0] > 0.5).astype(np.float32)
                    
                    # Resize back to original size
                    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                # Standard MedSAM2 (model1): direct call without predictor
                pred_np = medsam2_inference_direct(model, img_rgb, box_prompt, H, W, self.dev)
                pred_np = pred_np.astype(np.float32)  # Already binary [0,1]
            
            # Calculate dice
            pred_t = torch.from_numpy(pred_np).unsqueeze(0).to(self.dev)
            dice = dice_coef(pred_t, gt_t)

        initial_masks[idx] = pred_np

        # "accepted" is still computed, but final is always GT
        accepted = (dice > self.th)
        final_np = gt_np

        final_masks[idx] = final_np
        self._save_mask(final_np, pid, sl_name)

        return dice, accepted

    # ---------------- helper: save final mask ---------------
    def _save_mask(self, mask_np: np.ndarray, pid: str, sl_name: str):
        dst_dir = os.path.join(self.save_root, "predictions", pid)
        os.makedirs(dst_dir, exist_ok=True)
        out = os.path.join(dst_dir, sl_name)
        Image.fromarray((mask_np * 255).astype(np.uint8)).save(out)


# ======================= main ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kits", choices=["kits", "lits", "pancreas", "colon"], help="dataset name")
    parser.add_argument("--save_path", type=str, default="./simulation_output/",
                        help="path to save simulation results")
    parser.add_argument("--img_size", type=int, default=224, help="input size (for traditional models)")
    parser.add_argument("--n_skip", type=int, default=3, help="num skip connections")
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16", help="vit backbone")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--vit_patches_size", type=int, default=16, help="vit patch size")
    parser.add_argument("--model_name", type=str, default="TransUNet", 
                        help="method name: TransUNet, MedFormer, AttentionUNet, UNetPlusPlus, SwinUnet, FAT_Net, H2Former, MedSAM, MedSAM2")
    parser.add_argument("--medsam2_cfg", type=str, default="sam2.1_hiera_t512.yaml",
                        help="MedSAM2 config file name (e.g., sam2.1_hiera_t512.yaml, will be searched in sam2/configs/)")

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        help="Space-separated thresholds, e.g. --thresholds 0.7 0.75 0.8 0.85 0.9 0.95",
    )
    parser.add_argument(
        "--exclude-patients",
        type=str,
        default="",
        help='Comma-separated patient IDs to exclude, e.g., "patient_001,patient_002". Default: "". Use "" to disable.',
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_config = {
        "kits": {"root_path": "./2D_data/kits/test", "num_classes": 2},
        "pancreas": {"root_path": "./2D_data/pancreas/test", "num_classes": 2},
        "lits": {"root_path": "./2D_data/lits/test", "num_classes": 2},
        "colon": {"root_path": "./2D_data/colon/test", "num_classes": 2},
    }

    if args.dataset not in dataset_config:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(dataset_config.keys())}")

    root_path = dataset_config[args.dataset]["root_path"]
    num_classes = dataset_config[args.dataset]["num_classes"]

    exclude_patients = (
        [p.strip() for p in args.exclude_patients.split(",") if p.strip()]
        if args.exclude_patients
        else []
    )

    # Determine model type
    if args.model_name in ["MedSAM"]:
        model_type = "medsam"
        img_size = 1024  # MedSAM uses 1024
    elif args.model_name in ["MedSAM2"]:
        model_type = "medsam2"
        img_size = 512  # MedSAM2 uses 512
    else:
        model_type = "traditional"
        img_size = args.img_size

    # -------- build models --------
    device = "cuda"
    
    if args.model_name == "TransUNet":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = args.n_skip
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.vit_name.find("R50") != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

        model1 = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        print("Created the original TransUNet as model1!")

        config_small = CONFIGS_ViT_seg["R18-ViT-S_16"]
        config_small.n_classes = num_classes
        config_small.n_skip = args.n_skip
        config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model2 = TransUNet_ours(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        print("Created the TransUNet_ours as model2!")

        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "transunet_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))

        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "transunet_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "MedFormer":
        model1 = MedFormer(in_chan=1, num_classes=num_classes).cuda()
        model2 = MedFormer_ours(in_chan=1, num_classes=num_classes).cuda()
        print("Created the MedFormer as model1!")
        print("Created the MedFormer_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "medformer_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))

        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "medformer_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "AttentionUNet":
        model1 = AttentionUNet(in_ch=1, num_classes=num_classes).cuda()
        model2 = AttentionUNet_ours(in_ch=1, num_classes=num_classes).cuda()
        print("Created the AttentionUNet as model1!")
        print("Created the AttentionUNet_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "attention_unet_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))

        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "attention_unet_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "UNetPlusPlus":
        model1 = UNetPlusPlus(in_ch=1, num_classes=num_classes).cuda()
        model2 = UNetPlusPlus_ours(in_ch=1, num_classes=num_classes).cuda()
        print("Created the UNetPlusPlus as model1!")
        print("Created the UNetPlusPlus_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "unetpp_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))
        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "unetpp_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "SwinUnet":
        model1 = SwinUnet(SwinUnet_config(), img_size=args.img_size, num_classes=num_classes).cuda()
        model2 = SwinUnet_ours(SwinUnet_config(), img_size=args.img_size, num_classes=num_classes).cuda()
        print("Created the SwinUnet as model1!")
        print("Created the SwinUnet_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "swin_unet_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))
        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "swin_unet_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "FAT_Net":
        model1 = FAT_Net(n_channels=1, n_classes=num_classes).cuda()
        model2 = FATNet_ours(in_chan=1, num_classes=num_classes).cuda()
        print("Created the FAT_Net as model1!")
        print("Created the FATNet_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "FAT_Net_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))
        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "FAT_Net_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "H2Former":
        model1 = res34_swin_MS(image_size=args.img_size, num_class=num_classes).cuda()
        model2 = H2Former_ours(img_size=args.img_size, num_classes=num_classes).cuda()
        print("Created the res34_swin_MS as model1!")
        print("Created the H2Former_ours as model2!")
        ckpt_path1 = os.path.join(
            "./checkpoints/",
            "H2Former_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model1.load_state_dict(torch.load(_resolve_ckpt(ckpt_path1)))
        print("Loaded model1 ckpt:", _resolve_ckpt(ckpt_path1))
        ckpt_path2 = os.path.join(
            "./checkpoints/",
            "H2Former_ours_neighbor_" + args.dataset + "224",
            "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
        )
        model2.load_state_dict(torch.load(_resolve_ckpt(ckpt_path2)))
        print("Loaded model2 ckpt:", _resolve_ckpt(ckpt_path2))
        
    elif args.model_name == "MedSAM":
        # Lazy import — only the `medsam` conda env has segment_anything etc.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../foundation_models/medsam'))
        from segment_anything import sam_model_registry
        import vit_seg_configs as medsam_configs
        from My_utils import MedSAM_with_TRACE, TRACE

        # MedSAM model loading
        medsam_root = os.path.join(os.path.dirname(__file__), '../foundation_models/medsam')
        medsam_base_ckpt = os.path.join(medsam_root, "medsam_vit_b.pth")
        
        # Load base SAM model
        sam_model = sam_model_registry["vit_b"](checkpoint=medsam_base_ckpt)



        # --- build two independent SAM models ---
        sam_model1 = sam_model_registry["vit_b"](checkpoint=medsam_base_ckpt).to(device)
        sam_model2 = sam_model_registry["vit_b"](checkpoint=medsam_base_ckpt).to(device)



        
        # Model1: standard MedSAM (neighbor version, no TRACE)
        # Wrap with MedSAM_Wrapper (like test_vanilla_finetune.py)
        model1 = MedSAM_Wrapper(
            image_encoder=sam_model1.image_encoder,
            mask_decoder=sam_model1.mask_decoder,
            prompt_encoder=sam_model1.prompt_encoder
        )
        ckpt_path1_pattern = os.path.join(medsam_root, "work_dir", f"finetune_neighbor-{args.dataset}-*", "medsam_model_best.pth")
        import glob
        ckpt_path1_list = glob.glob(ckpt_path1_pattern)
        if not ckpt_path1_list:
            raise FileNotFoundError(f"Cannot find model1 checkpoint matching pattern: {ckpt_path1_pattern}")
        ckpt_path1 = sorted(ckpt_path1_list)[-1]  # Get the latest one
        model1.load_state_dict(torch.load(ckpt_path1, map_location=device)["model"], strict=True)
        print("Loaded MedSAM model1 ckpt:", ckpt_path1)
        
        # Model2: MedSAM_with_TRACE (neighbor)
        config_small = medsam_configs.get_r18_s16_config()
        config_small.n_classes = 2
        config_small.n_skip = 3
        config_small.patches.grid = (int(1024 / 16), int(1024 / 16))
        refinement_mod = TRACE(config_small, img_size=1024, num_classes=2, pretrained=True)
        model2 = MedSAM_with_TRACE(
            image_encoder=sam_model2.image_encoder,
            mask_decoder=sam_model2.mask_decoder,
            prompt_encoder=sam_model2.prompt_encoder,
            refinement=refinement_mod,
        )
        ckpt_path2_pattern = os.path.join(medsam_root, "work_dir", f"with_TRACE_neighbor-{args.dataset}-*", "medsam_model_best.pth")
        ckpt_path2_list = glob.glob(ckpt_path2_pattern)
        if not ckpt_path2_list:
            raise FileNotFoundError(f"Cannot find model2 checkpoint matching pattern: {ckpt_path2_pattern}")
        ckpt_path2 = sorted(ckpt_path2_list)[-1]  # Get the latest one
        model2.load_state_dict(torch.load(ckpt_path2, map_location=device)["model"], strict=True)
        print("Loaded MedSAM model2 ckpt:", ckpt_path2)
        
    elif args.model_name == "MedSAM2":
        # Lazy import — only the `medsam2` conda env has sam2/hydra/omegaconf installed.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../foundation_models/medsam2'))
        from sam2.build_sam import build_sam2
        from sam2.utils.transforms import SAM2Transforms
        from training.model.medsam2_with_trace import MedSAM2_with_TRACE
        from training.model.trace import TRACE as medsam2_TRACE

        # MedSAM2 model loading
        medsam2_root = os.path.join(os.path.dirname(__file__), '../foundation_models/medsam2')
        import glob
        # Hydra expects config path relative to sam2 package, e.g., "configs/sam2.1_hiera_t512.yaml"
        # If user provides just filename, prepend "configs/"
        if "/" not in args.medsam2_cfg:
            cfg_path = f"configs/{args.medsam2_cfg}"
        else:
            cfg_path = args.medsam2_cfg
        
        # Model1: standard MedSAM2 (neighbor, no TRACE)
        ckpt_path1_pattern = os.path.join(medsam2_root, "work_dir", f"MedSAM2-2D-baseline-{args.dataset}-neighbor", "*", "best.pth")
        ckpt_path1_list = glob.glob(ckpt_path1_pattern)
        if not ckpt_path1_list:
            raise FileNotFoundError(f"Cannot find model1 checkpoint matching pattern: {ckpt_path1_pattern}")
        ckpt_path1 = sorted(ckpt_path1_list)[-1]  # Get the latest one
        
        model1 = build_sam2(
            config_file=cfg_path,
            ckpt_path=None,  # Don't load from config, load from checkpoint below
            device=device,
            mode="eval"
        )
        # Load checkpoint
        checkpoint1 = torch.load(ckpt_path1, map_location=device)
        if "model" in checkpoint1:
            model1.load_state_dict(checkpoint1["model"], strict=False)
        else:
            model1.load_state_dict(checkpoint1, strict=False)
        print("Loaded MedSAM2 model1 ckpt:", ckpt_path1)
        
        # Model2: MedSAM2_with_TRACE (neighbor)
        sam2_base = build_sam2(
            config_file=cfg_path,
            ckpt_path=None,
            device=device,
            mode="eval"
        )
        # Load base weights
        checkpoint1 = torch.load(ckpt_path1, map_location=device)
        if "model" in checkpoint1:
            sam2_base.load_state_dict(checkpoint1["model"], strict=False)
        else:
            sam2_base.load_state_dict(checkpoint1, strict=False)
        
        # Create refinement config
        class RefinementConfig:
            def __init__(self):
                self.classifier = None
        
        refinement_config = RefinementConfig()
        refinement_module = medsam2_TRACE(
            refinement_config,
            img_size=512,
            pretrained=True
        )
        
        model2 = MedSAM2_with_TRACE(
            sam2_model=sam2_base,
            refinement=refinement_module,
            refine_iters=3,
            detach_between_iters=True
        )
        
        ckpt_path2_pattern = os.path.join(medsam2_root, "work_dir", f"MedSAM2-2D-with_TRACE-{args.dataset}-neighbor", "*", "best.pth")
        ckpt_path2_list = glob.glob(ckpt_path2_pattern)
        if not ckpt_path2_list:
            raise FileNotFoundError(f"Cannot find model2 checkpoint matching pattern: {ckpt_path2_pattern}")
        ckpt_path2 = sorted(ckpt_path2_list)[-1]  # Get the latest one
        
        checkpoint2 = torch.load(ckpt_path2, map_location=device)
        if "model" in checkpoint2:
            model2.load_state_dict(checkpoint2["model"], strict=False)
        else:
            model2.load_state_dict(checkpoint2, strict=False)
        print("Loaded MedSAM2 model2 ckpt:", ckpt_path2)
        
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    # -------- sweep thresholds for both modes --------
    thresholds = [float(x) for x in args.thresholds]
    ai_modes = ["A", "B"]

    out_dir = os.path.join(args.save_path, args.model_name, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    results = {m: {"reject_rates": [], "total_slices": [], "total_rejects": []} for m in ai_modes}

    for th in thresholds:
        for m in ai_modes:
            th_dir = os.path.join(out_dir, f"mode{m}", f"{th:.2f}")
            os.makedirs(th_dir, exist_ok=True)

            tester = DoctorSimTester(
                data_root=root_path,
                model1=model1,
                model2=model2,
                threshold=th,
                save_root=th_dir,
                device="cuda",
                img_size=img_size,
                exclude_patients=exclude_patients,
                ai_mode=m,
                model_type=model_type,
            )

            summary = tester.run()
            reject_rate = summary["total_manual_corrections"] / max(1, summary["total_slices"])

            results[m]["reject_rates"].append(float(reject_rate))
            results[m]["total_slices"].append(int(summary["total_slices"]))
            results[m]["total_rejects"].append(int(summary["total_manual_corrections"]))

    # -------- save curve data --------
    curve_json = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "thresholds": thresholds,
        "exclude_patients": exclude_patients,
        "modeA": results["A"],
        "modeB": results["B"],
    }
    with open(os.path.join(out_dir, "reject_rate_vs_threshold.json"), "w") as f:
        json.dump(curve_json, f, indent=2)

    # -------- plot (two lines) --------
    plt.figure()
    plt.plot(thresholds, results["A"]["reject_rates"], marker="o", label="AI Mode A")
    plt.plot(thresholds, results["B"]["reject_rates"], marker="o", label="AI Mode B")

    plt.xlabel("Threshold")
    plt.ylabel("Reject Rate")
    plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds])
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()

    plt.title(f"{args.model_name} on {args.dataset}")

    out_png = os.path.join(out_dir, "reject_rate_vs_threshold.png")
    out_pdf = os.path.join(out_dir, "reject_rate_vs_threshold.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(
        f"\nSaved reject-rate curve to:\n  {out_png}\n  {out_pdf}\n"
        f"and data:\n  {os.path.join(out_dir, 'reject_rate_vs_threshold.json')}\n"
    )
