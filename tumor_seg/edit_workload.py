# Edit Workload Comparison v2:
# - Compare edit workload between two AI modes
# - Mode A: per-slice prediction with model1(img) only
# - Mode B: ref-conditioned prediction with model2(img, ref_img, ref_gt_mask)['final']
# - Reference mask for Mode B: always GT (neighbor reference, option 1)
# - Every slice requires doctor's edit (no accept/reject logic)
# - Metrics: edit ratio (modified pixels / GT pixels)
# - Support traditional models, MedSAM, and MedSAM2

import os, json
import sys
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, binary_erosion, distance_transform_edt
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

# MedSAM imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../foundation_models/medsam'))
from segment_anything import sam_model_registry
import vit_seg_configs as medsam_configs
from My_utils import MedSAM_with_TRACE, TRACE
import torch.nn as nn

# MedSAM2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../foundation_models/medsam2'))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.model.medsam2_with_trace import MedSAM2_with_TRACE
from training.model.trace import TRACE as medsam2_TRACE


# ====================== Metrics =============================

def compute_edit_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute edit ratio: (FP + FN) / GT_area
    pred, gt: (H, W), binary masks (0 or 1)
    """
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    
    # FP: predicted as 1 but GT is 0
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    # FN: predicted as 0 but GT is 1
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    
    gt_area = np.sum(gt_bin)
    if gt_area == 0:
        # If GT is empty, use total pixels as denominator
        return (fp + fn) / (pred.shape[0] * pred.shape[1])
    
    return (fp + fn) / gt_area


def _binary(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr.squeeze()
    return (arr > 0.5)


def _boundary(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.sum() == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    eroded = binary_erosion(mask_bool, border_value=0)
    return mask_bool & ~eroded


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = _binary(pred); g = _binary(gt)
    tp = float((p & g).sum())
    return (2.0 * tp + eps) / (float(p.sum()) + float(g.sum()) + eps)


def sdsc_2px(pred: np.ndarray, gt: np.ndarray, tau_px: float = 2.0) -> float:
    p = _binary(pred); g = _binary(gt)
    pb = _boundary(p); gb = _boundary(g)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_to_gb = distance_transform_edt(~gb)
    dt_to_pb = distance_transform_edt(~pb)
    pred_close = int((dt_to_gb[pb] <= tau_px).sum())
    gt_close = int((dt_to_pb[gb] <= tau_px).sum())
    return (pred_close + gt_close) / float(pb.sum() + gb.sum())


def fn_rate_metric(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = _binary(pred); g = _binary(gt)
    tp = float((p & g).sum())
    fn = float((~p & g).sum())
    return (fn + eps) / (fn + tp + eps)


def fp_rate_metric(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = _binary(pred); g = _binary(gt)
    if p.sum() == 0:
        return 0.0
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    return (fp + eps) / (fp + tp + eps)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _binary(pred); g = _binary(gt)
    pb = _boundary(p); gb = _boundary(g)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan")
    dt_to_gb = distance_transform_edt(~gb)
    dt_to_pb = distance_transform_edt(~pb)
    d_p_to_g = dt_to_gb[pb]
    d_g_to_p = dt_to_pb[gb]
    return max(float(np.percentile(d_p_to_g, 95)), float(np.percentile(d_g_to_p, 95)))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {
        'edit_ratio': compute_edit_ratio(pred, gt),
        'dice': dice_np(pred, gt),
        'sdsc': sdsc_2px(pred, gt),
        'fn_rate': fn_rate_metric(pred, gt),
        'fp_rate': fp_rate_metric(pred, gt),
        'hd95': hd95(pred, gt),
    }


# ====================== utils =============================

def _prep_img(path: str, out: int = 224) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        print(f"Warning: {path} is all zeros.")
        arr = arr * 0.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=3)  # bicubic


def _prep_msk(path: str, out: int = 224) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=0)  # nearest


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H,W) -> (1,1,H,W)"""
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


# ====================== MedSAM utils =============================

def medsam_prep_img(path: str) -> Tuple[torch.Tensor, int, int]:
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


# MedSAM wrapper class (for model1), same as simulation_v3.py
class MedSAM_Wrapper(nn.Module):
    """Wrapper for standard MedSAM model (model1)"""
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding, features = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 1:
                box_torch = box_torch.unsqueeze(0).unsqueeze(0)
            elif len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    """MedSAM inference function"""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


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
def medsam2_inference(image_predictor, img_rgb, box):
    """
    MedSAM2 inference function
    Args:
        image_predictor: SAM2ImagePredictor instance
        img_rgb: RGB image (H, W, 3), uint8, pixel values in [0, 255]
        box: bounding box [x_min, y_min, x_max, y_max]
    Returns:
        medsam_seg: binary mask (H, W)
    """
    image_predictor.set_image(img_rgb)
    
    box_array = np.array(box).reshape(1, 4) if len(box.shape) == 1 else box
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_array,
        multimask_output=False,
        return_logits=False,
        normalize_coords=True,
    )
    
    medsam_seg = (masks[0] > 0.0).astype(np.uint8)
    return medsam_seg


# ==================== core tester =========================

class EditWorkloadTester:
    """
    Compare edit workload between Mode A and Mode B.
    
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
        self.save_root = save_root
        self.dev = device
        self.size = img_size
        self.exclude_patients = exclude_patients or []
        self.ai_mode = ai_mode.upper()
        self.model_type = model_type.lower()
        assert self.ai_mode in ["A", "B"], f"ai_mode must be 'A' or 'B', got {ai_mode}"
        assert self.model_type in ["traditional", "medsam", "medsam2"], f"model_type must be 'traditional', 'medsam', or 'medsam2', got {model_type}"
        
        # For MedSAM2, create image predictor only for model1 (standard MedSAM2)
        # Model2 (MedSAM2_with_TRACE) has different forward signature, so we use it directly
        if self.model_type == "medsam2":
            self.medsam2_predictor1 = SAM2ImagePredictor(self.m1)

    # ---------------------------- public --------------------
    def run(self) -> Dict:
        """Traverse all volumes and compute metrics."""
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
            print(f"[Mode {self.ai_mode}] Excluded patients {exclude}: {before} -> {after} patients")

        all_metrics_list = []  # List of all metric dicts from all slices

        for pid in tqdm(patient_dirs, desc=f"Patients (Mode {self.ai_mode})"):
            slice_metrics = self._process_volume(pid)
            all_metrics_list.extend(slice_metrics)

        # Aggregate metrics across all slices
        _META_KEYS = {"patient_id", "slice_name"}
        if len(all_metrics_list) == 0:
            summary = {
                "ai_mode": self.ai_mode,
                "excluded_patients": exclude,
                "total_slices": 0,
                "avg_metrics": {},
                "per_slice": [],
            }
        else:
            numeric_keys = [k for k in all_metrics_list[0].keys() if k not in _META_KEYS]
            avg_metrics = {
                name: float(np.nanmean([m[name] for m in all_metrics_list]))
                for name in numeric_keys
            }
            summary = {
                "ai_mode": self.ai_mode,
                "excluded_patients": exclude,
                "total_slices": len(all_metrics_list),
                "avg_metrics": avg_metrics,
                "per_slice": all_metrics_list,
            }

        print(
            f"\n>>> DONE (Mode {self.ai_mode}). "
            f"total_slices = {summary['total_slices']}, "
            f"avg_edit_ratio = {summary['avg_metrics'].get('edit_ratio', 0):.4f}, "
            f"avg_dice = {summary['avg_metrics'].get('dice', 0):.4f} <<<\n"
        )

        return summary

    # ------------------------ per-volume --------------------
    def _process_volume(self, pid: str) -> List[Dict[str, float]]:
        """Process one volume and return list of metrics for each slice."""
        ct_dir = os.path.join(self.root, "CT", pid)
        mk_dir = os.path.join(self.root, "Mask", pid)

        # sort by integer stem, filenames like 000.png
        slices = sorted(
            [f for f in os.listdir(ct_dir) if f.lower().endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        n = len(slices)
        if n == 0:
            return []

        center = n // 2  # if even, take the later one
        slice_metrics_list = []

        def _tag(m, idx):
            m["patient_id"] = pid
            m["slice_name"] = slices[idx]
            return m

        # ---- Mode A: every slice uses model1(img) ----
        if self.ai_mode == "A":
            for idx in range(n):
                metrics = self._infer_slice(
                    idx=idx,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m1,
                    need_ref=False,
                    ref_idx=None,
                    slices=slices,
                )
                if metrics is not None:
                    slice_metrics_list.append(_tag(metrics, idx))

        # ---- Mode B: center by model1, others by model2 with GT reference ----
        else:
            # Center slice
            metrics_center = self._infer_slice(
                idx=center,
                ct_dir=ct_dir,
                mk_dir=mk_dir,
                model=self.m1,
                need_ref=False,
                ref_idx=None,
                slices=slices,
            )
            if metrics_center is not None:
                slice_metrics_list.append(_tag(metrics_center, center))

            # Process right side first (center+1 to n-1)
            for r in range(center + 1, n):
                metrics = self._infer_slice(
                    idx=r,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m2,
                    need_ref=True,
                    ref_idx=r - 1,  # neighbor reference
                    slices=slices,
                )
                if metrics is not None:
                    slice_metrics_list.append(_tag(metrics, r))

            # Process left side (center-1 to 0)
            for l in range(center - 1, -1, -1):
                metrics = self._infer_slice(
                    idx=l,
                    ct_dir=ct_dir,
                    mk_dir=mk_dir,
                    model=self.m2,
                    need_ref=True,
                    ref_idx=l + 1,  # neighbor reference
                    slices=slices,
                )
                if metrics is not None:
                    slice_metrics_list.append(_tag(metrics, l))

        return slice_metrics_list

    # ---------- helper: run inference + compute metrics -----
    def _infer_slice(
        self,
        idx: int,
        ct_dir: str,
        mk_dir: str,
        model,
        need_ref: bool,
        ref_idx: int,
        slices: List[str],
    ) -> Dict[str, float]:
        """Run inference on one slice and compute metrics."""
        sl_name = slices[idx]
        ct_path = os.path.join(ct_dir, sl_name)
        mk_path = os.path.join(mk_dir, sl_name)

        # Get GT mask at original resolution (all model types)
        gt_np = np.array(Image.open(mk_path).convert("L"), dtype=np.float32) / 255.0

        # Skip slices with extremely small GT (≤ 3 pixels) to avoid degenerate edit ratios
        gt_bin = (gt_np > 0.5).astype(np.float32)
        if np.sum(gt_bin) <= 3:
            return None

        if self.model_type == "traditional":
            # Traditional model inference
            img_np = _prep_img(ct_path, self.size)
            img_t = _to_tensor(img_np).to(self.dev)

            if need_ref:
                ref_name = slices[ref_idx]
                ref_ct = os.path.join(ct_dir, ref_name)
                ref_mk = os.path.join(mk_dir, ref_name)

                ref_img_np = _prep_img(ref_ct, self.size)
                ref_mk_np = _prep_msk(ref_mk, self.size)  # always GT reference

                ref_img_t = _to_tensor(ref_img_np).to(self.dev)
                ref_mk_t = _to_tensor(ref_mk_np).to(self.dev)

                with torch.no_grad():
                    out = model(img_t, ref_img_t, ref_mk_t)
                    logit = out["final"] if isinstance(out, dict) and "final" in out else out
            else:
                with torch.no_grad():
                    logit = model(img_t)

            pred = torch.argmax(F.softmax(logit, dim=1), dim=1).float()  # (1,H,W)
            pred_np = pred.squeeze(0).detach().cpu().numpy()  # (H,W) at model resolution
            orig_h, orig_w = gt_np.shape
            if pred_np.shape[0] != orig_h or pred_np.shape[1] != orig_w:
                pred_np = zoom(pred_np, (orig_h / pred_np.shape[0], orig_w / pred_np.shape[1]), order=0)
                pred_np = (pred_np > 0.5).astype(np.float32)

        elif self.model_type == "medsam":
            # MedSAM inference
            img_1024_tensor, H, W = medsam_prep_img(ct_path)
            img_1024_tensor = img_1024_tensor.to(self.dev)
            
            # Get box prompt (same logic as simulation_v3.py)
            if need_ref:
                # Mode B: use reference slice's GT for box_prompt
                ref_name = slices[ref_idx]
                ref_mk_path = os.path.join(mk_dir, ref_name)
                ref_mask_uint8 = np.array(Image.open(ref_mk_path).convert("L"))
                box_prompt = medsam_extract_box_from_mask(ref_mask_uint8)
            else:
                # Mode A: use center slice GT for all slices; Mode B center: use its own GT
                if self.ai_mode == "A":
                    # Mode A: resize center slice mask to current slice size, then extract box (like simulation_v3)
                    center_idx = len(slices) // 2
                    center_mk_path = os.path.join(mk_dir, slices[center_idx])
                    center_mask_uint8 = np.array(Image.open(center_mk_path).convert("L"))
                    center_mask_resized = cv2.resize(center_mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                    box_prompt = medsam_extract_box_from_mask(center_mask_resized)
                else:
                    current_mask_uint8 = np.array(Image.open(mk_path).convert("L"))
                    box_prompt = medsam_extract_box_from_mask(current_mask_uint8)

            # Scale box to 1024x1024
            box_1024 = box_prompt / np.array([W, H, W, H]) * 1024  # (1, 4)

            if need_ref:
                # MedSAM_with_TRACE (model2): direct forward call, box_np (1, 4) like simulation.py
                ref_name = slices[ref_idx]
                ref_ct_path = os.path.join(ct_dir, ref_name)
                ref_mk_path = os.path.join(mk_dir, ref_name)

                ref_img_1024_tensor, _, _ = medsam_prep_img(ref_ct_path)
                ref_img_1024_tensor = ref_img_1024_tensor.to(self.dev)

                ref_mask_np = np.array(Image.open(ref_mk_path).convert("L"), dtype=np.float32) / 255.0
                ref_mask_resized = cv2.resize(ref_mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                ref_gt_tensor = torch.from_numpy(ref_mask_resized).float().unsqueeze(0).unsqueeze(0).to(self.dev)

                if len(box_1024.shape) == 2 and box_1024.shape[0] == 1:
                    box_np = box_1024
                else:
                    box_np = box_1024.flatten()[:4].reshape(1, 4)

                with torch.no_grad():
                    outputs = model(img_1024_tensor, box_np, ref_img_1024_tensor, ref_gt_tensor)
                    if isinstance(outputs, dict):
                        logits = outputs["final"]
                    else:
                        logits = outputs
                    pred_mask = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    pred_np = (pred_mask[0] > 0.5).astype(np.float32)
                    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                # Standard MedSAM (model1): use MedSAM_Wrapper forward like simulation.py
                box_np = box_1024[0] if len(box_1024.shape) == 2 else box_1024  # (4,)
                with torch.no_grad():
                    ori_res_masks = model(img_1024_tensor, box_np)
                    pred_mask = torch.sigmoid(ori_res_masks)
                    pred_mask_np = pred_mask.squeeze().cpu().numpy()
                    pred_np = (pred_mask_np > 0.5).astype(np.float32)
                    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)

        else:  # medsam2
            # MedSAM2 inference
            img_rgb, H, W = medsam2_prep_img(ct_path)
            
            # Get box prompt
            if need_ref:
                # Mode B: use reference slice's GT for box_prompt
                ref_name = slices[ref_idx]
                ref_mk_path = os.path.join(mk_dir, ref_name)
                ref_mask_uint8 = np.array(Image.open(ref_mk_path).convert("L"))
                box_prompt = medsam2_extract_box_from_mask(ref_mask_uint8, image_size=512)
            else:
                # Mode A: use center slice GT for all slices
                # Mode B center slice: use its own GT
                if self.ai_mode == "A":
                    center_idx = len(slices) // 2
                    center_mk_path = os.path.join(mk_dir, slices[center_idx])
                    center_mask_uint8 = np.array(Image.open(center_mk_path).convert("L"))
                    box_prompt = medsam2_extract_box_from_mask(center_mask_uint8, image_size=512)
                else:
                    current_mask_uint8 = np.array(Image.open(mk_path).convert("L"))
                    box_prompt = medsam2_extract_box_from_mask(current_mask_uint8, image_size=512)
            
            if need_ref:
                # Model2 (MedSAM2_with_TRACE): direct forward call
                ref_name = slices[ref_idx]
                ref_ct_path = os.path.join(ct_dir, ref_name)
                ref_mk_path = os.path.join(mk_dir, ref_name)
                
                # Prepare reference image
                ref_img_rgb, _, _ = medsam2_prep_img(ref_ct_path)
                
                # Prepare reference mask - always GT
                ref_mask_np = np.array(Image.open(ref_mk_path).convert("L"), dtype=np.float32) / 255.0
                
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
                box_scaled = box_prompt / np.array([W, H, W, H]) * 512
                box_tensor = torch.from_numpy(box_scaled).float().unsqueeze(0)  # (1, 4)
                
                # Move to device
                img_tensor = img_tensor.to(self.dev)
                ref_img_tensor = ref_img_tensor.to(self.dev)
                ref_gt_tensor = ref_gt_tensor.to(self.dev)
                box_tensor = box_tensor.to(self.dev)
                
                # Forward pass
                with torch.no_grad():
                    output = model(img_tensor, box_tensor, ref_img_tensor, ref_gt_tensor, image_size=512)
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
                # Standard MedSAM2: use predictor
                predictor = self.medsam2_predictor1
                pred_np = medsam2_inference(predictor, img_rgb, box_prompt)
                pred_np = pred_np.astype(np.float32)  # Already binary [0,1]

        # Compute metrics (use original GT size)
        metrics = compute_all_metrics(pred_np, gt_np)

        return metrics


# ======================= main ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./simulation_output/edit_workload/",
                        help="path to save results")
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
        "--exclude-patients",
        type=str,
        default="",
        help='Comma-separated patient IDs to exclude, e.g., "SMet42,SMet17". Default: "". Use "" to disable.',
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help='Datasets to run. Default: all 4 datasets. E.g., --datasets kits lits',
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

    # Helper function to load models for a specific dataset
    def load_models_for_dataset(dataset_name, model_name, img_size, n_skip, vit_name, vit_patches_size, medsam2_cfg=None):
        """Load model1 and model2 for a specific dataset."""
        num_classes = dataset_config[dataset_name]["num_classes"]
        device = "cuda"
        
        if model_name == "TransUNet":
            config_vit = CONFIGS_ViT_seg[vit_name]
            config_vit.n_classes = num_classes
            config_vit.n_skip = n_skip
            config_vit.patches.size = (vit_patches_size, vit_patches_size)
            if vit_name.find("R50") != -1:
                config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

            model1 = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()

            config_small = CONFIGS_ViT_seg["R18-ViT-S_16"]
            config_small.n_classes = num_classes
            config_small.n_skip = n_skip
            config_small.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
            model2 = TransUNet_ours(config_vit, config_small, img_size=img_size, num_classes=config_vit.n_classes).cuda()

            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "transunet_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "transunet_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "MedFormer":
            model1 = MedFormer(in_chan=1, num_classes=num_classes).cuda()
            model2 = MedFormer_ours(in_chan=1, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "medformer_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "medformer_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "AttentionUNet":
            model1 = AttentionUNet(in_ch=1, num_classes=num_classes).cuda()
            model2 = AttentionUNet_ours(in_ch=1, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "attention_unet_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "attention_unet_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "UNetPlusPlus":
            model1 = UNetPlusPlus(in_ch=1, num_classes=num_classes).cuda()
            model2 = UNetPlusPlus_ours(in_ch=1, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "unetpp_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "unetpp_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "SwinUnet":
            model1 = SwinUnet(SwinUnet_config(), img_size=img_size, num_classes=num_classes).cuda()
            model2 = SwinUnet_ours(SwinUnet_config(), img_size=img_size, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "swin_unet_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "swin_unet_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "FAT_Net":
            model1 = FAT_Net(n_channels=1, n_classes=num_classes).cuda()
            model2 = FATNet_ours(in_chan=1, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "FAT_Net_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "FAT_Net_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "H2Former":
            model1 = res34_swin_MS(image_size=img_size, num_class=num_classes).cuda()
            model2 = H2Former_ours(img_size=img_size, num_classes=num_classes).cuda()
            ckpt_path1 = os.path.join(
                "./checkpoints/",
                "H2Former_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model1.load_state_dict(torch.load(ckpt_path1))
            print(f"Loaded model1 ckpt: {ckpt_path1}")
            ckpt_path2 = os.path.join(
                "./checkpoints/",
                "H2Former_ours_neighbor_" + dataset_name + "224",
                "TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224/epoch_149.pth",
            )
            model2.load_state_dict(torch.load(ckpt_path2))
            print(f"Loaded model2 ckpt: {ckpt_path2}")
        elif model_name == "MedSAM":
            # MedSAM model loading (must create two independent SAM models to avoid overwriting when loading two ckpts)
            medsam_root = os.path.join(os.path.dirname(__file__), '../foundation_models/medsam')
            medsam_base_ckpt = os.path.join(medsam_root, "medsam_vit_b.pth")
            import glob

            sam_model1 = sam_model_registry["vit_b"](checkpoint=medsam_base_ckpt).to(device)
            sam_model2 = sam_model_registry["vit_b"](checkpoint=medsam_base_ckpt).to(device)

            # Model1: standard MedSAM (neighbor version, no TRACE) - use MedSAM_Wrapper like simulation.py
            model1 = MedSAM_Wrapper(
                image_encoder=sam_model1.image_encoder,
                mask_decoder=sam_model1.mask_decoder,
                prompt_encoder=sam_model1.prompt_encoder,
            )
            ckpt_path1_pattern = os.path.join(medsam_root, "work_dir", f"finetune_neighbor-{dataset_name}-*", "medsam_model_best.pth")
            ckpt_path1_list = glob.glob(ckpt_path1_pattern)
            if not ckpt_path1_list:
                raise FileNotFoundError(f"Cannot find model1 checkpoint matching pattern: {ckpt_path1_pattern}")
            ckpt_path1 = sorted(ckpt_path1_list)[-1]
            model1.load_state_dict(torch.load(ckpt_path1, map_location=device)["model"], strict=True)
            model1 = model1.to(device).eval()
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
            ckpt_path2_pattern = os.path.join(medsam_root, "work_dir", f"with_TRACE_neighbor-{dataset_name}-*", "medsam_model_best.pth")
            ckpt_path2_list = glob.glob(ckpt_path2_pattern)
            if not ckpt_path2_list:
                raise FileNotFoundError(f"Cannot find model2 checkpoint matching pattern: {ckpt_path2_pattern}")
            ckpt_path2 = sorted(ckpt_path2_list)[-1]
            model2.load_state_dict(torch.load(ckpt_path2, map_location=device)["model"], strict=True)
            model2 = model2.to(device).eval()
            print("Loaded MedSAM model2 ckpt:", ckpt_path2)
        elif model_name == "MedSAM2":
            # MedSAM2 model loading
            medsam2_root = os.path.join(os.path.dirname(__file__), '../foundation_models/medsam2')
            import glob
            # Hydra expects config path relative to sam2 package, e.g., "configs/sam2.1_hiera_t512.yaml"
            # If user provides just filename, prepend "configs/"
            if "/" not in medsam2_cfg:
                cfg_path = f"configs/{medsam2_cfg}"
            else:
                cfg_path = medsam2_cfg
            
            # Model1: standard MedSAM2 (neighbor, no TRACE)
            ckpt_path1_pattern = os.path.join(medsam2_root, "work_dir", f"MedSAM2-2D-baseline-{dataset_name}-neighbor", "*", "best.pth")
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
            
            ckpt_path2_pattern = os.path.join(medsam2_root, "work_dir", f"MedSAM2-2D-with_TRACE-{dataset_name}-neighbor", "*", "best.pth")
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
            raise ValueError(f"Unknown model_name: {model_name}")
        
        return model1, model2

    # -------- Run for all datasets and both modes --------
    all_datasets = ["kits", "pancreas", "lits", "colon"]
    datasets = args.datasets if args.datasets else all_datasets
    ai_modes = ["A", "B"]
    
    save_base = os.path.join(args.save_path, args.model_name)
    os.makedirs(save_base, exist_ok=True)
    
    # Store results: {dataset: {mode: {metric: value}}}
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}\n")
        
        if dataset not in dataset_config:
            print(f"Warning: dataset {dataset} not found, skipping...")
            continue
            
        root_path = dataset_config[dataset]["root_path"]
        all_results[dataset] = {}
        
        # Load models for this dataset
        print(f"Loading models for dataset: {dataset}")
        model1, model2 = load_models_for_dataset(
            dataset, args.model_name, img_size, args.n_skip, args.vit_name, args.vit_patches_size, args.medsam2_cfg
        )
        
        per_slice_all = []
        for mode in ai_modes:
            print(f"\n--- Mode {mode} ---\n")

            th_dir = os.path.join(save_base, dataset, f"mode{mode}")
            os.makedirs(th_dir, exist_ok=True)

            tester = EditWorkloadTester(
                data_root=root_path,
                model1=model1,
                model2=model2,
                save_root=th_dir,
                device="cuda",
                img_size=img_size,
                exclude_patients=exclude_patients,
                ai_mode=mode,
                model_type=model_type,
            )

            summary = tester.run()
            all_results[dataset][mode] = summary["avg_metrics"]
            for row in summary["per_slice"]:
                row_copy = dict(row)
                row_copy["mode"] = mode
                per_slice_all.append(row_copy)

        # Save per-slice CSV for this dataset
        import csv
        csv_path = os.path.join(save_base, f"{dataset}_per_slice.csv")
        if per_slice_all:
            fieldnames = ["patient_id", "slice_name", "mode",
                          "edit_ratio", "dice", "sdsc", "fn_rate", "fp_rate", "hd95"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(per_slice_all)
            print(f"Saved per-slice data: {csv_path}")

    # -------- Save results and plot --------
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    import fcntl
    json_path = os.path.join(save_base, "edit_workload_results.json")
    lock_path = json_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing_results = {}
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("results", {})
            existing_results.update(_sanitize(all_results))
            results_json = {
                "model_name": args.model_name,
                "exclude_patients": exclude_patients,
                "results": existing_results,
            }
            with open(json_path, "w") as f:
                json.dump(results_json, f, indent=2)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    print(f"\nResults saved to: {json_path}\n")
    
    # Plot bar charts
    metric_name = "edit_ratio"  # Currently only one metric, but easy to extend
    
    # Plot for each dataset
    for dataset in datasets:
        if dataset not in all_results:
            continue
            
        mode_a_value = all_results[dataset]["A"].get(metric_name, 0)
        mode_b_value = all_results[dataset]["B"].get(metric_name, 0)
        
        plt.figure(figsize=(6, 5))
        plt.bar(["Mode A", "Mode B"], [mode_a_value, mode_b_value], width=0.6, color=['#3498db', '#e74c3c'])
        plt.ylabel("Edit Ratio", fontsize=12)
        plt.title(f"{args.model_name} on {dataset}", fontsize=14)
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
        
        out_png = os.path.join(save_base, f"{dataset}_edit_workload.png")
        out_pdf = os.path.join(save_base, f"{dataset}_edit_workload.pdf")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        print(f"Saved plot for {dataset}: {out_png}")
    
    # Plot average across all datasets
    avg_mode_a = np.mean([all_results[d]["A"].get(metric_name, 0) for d in datasets if d in all_results])
    avg_mode_b = np.mean([all_results[d]["B"].get(metric_name, 0) for d in datasets if d in all_results])
    
    plt.figure(figsize=(6, 5))
    plt.bar(["Mode A", "Mode B"], [avg_mode_a, avg_mode_b], width=0.6, color=['#3498db', '#e74c3c'])
    plt.ylabel("Edit Ratio", fontsize=12)
    plt.title(f"{args.model_name} Average", fontsize=14)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    
    out_png = os.path.join(save_base, "average_edit_workload.png")
    out_pdf = os.path.join(save_base, "average_edit_workload.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved average plot: {out_png}")
    
    print("\n" + "="*60)
    print("All done!")
    print("="*60)
