# -*- coding: utf-8 -*-
"""
Evaluate a fine-tuned MedSAM2 video-mode + refinement module checkpoint in the
same two-half middle-slice box-prompt protocol as test_medsam2_2d_video.py, but
additionally runs the TRACE module on each propagated slice.

Inputs per volume:
  imgs[D, H, W], middle_slice GT (→ box prompt + refinement reference)
Outputs per volume:
  refined_pred[D, H, W] binary masks, compared to per-slice GT for Dice/IoU.

Writes: results/results_<dataset>_video_refine.json (does not overwrite the
existing baseline / tumorseg JSONs).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2_video_predictor
from training.model.trace import TRACE
from training.utils.train_utils import register_omegaconf_resolvers

# Register the custom OmegaConf resolvers used by the training yaml (times/
# divide/etc.) so that build_sam2_video_predictor's Hydra compose can
# evaluate interpolations inside the training config.
try:
    register_omegaconf_resolvers()
except Exception:
    # Already registered on re-import
    pass


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ----------------------------------------------------------------------------
# Generic helpers (copied / adapted from test_medsam2_2d_video.py)
# ----------------------------------------------------------------------------
def compute_metrics(pred, target, smooth=1e-6):
    pred = (pred.flatten() > 0.5).astype(np.float32)
    target = (target.flatten() > 0.5).astype(np.float32)
    inter = (pred * target).sum()
    total = pred.sum() + target.sum()
    union = total - inter
    iou = (inter + smooth) / (union + smooth)
    dice = (2.0 * inter + smooth) / (total + smooth)
    return iou, dice


def extract_box_from_mask(mask, image_size=512):
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = (mask > 0.5).astype(np.uint8)
    H, W = mask_binary.shape
    mask_resized = cv2.resize(mask_binary, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_resized > 0)
    if len(ys) == 0:
        return np.array([0, 0, W, H])
    x0, x1 = int(xs.min() * W / image_size), int(xs.max() * W / image_size)
    y0, y1 = int(ys.min() * H / image_size), int(ys.max() * H / image_size)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return np.array([x0, y0, x1, y1])


def organize_patient_slices(root_dir, case_num):
    ct_dir = os.path.join(root_dir, "CT", case_num)
    mask_dir = os.path.join(root_dir, "Mask", case_num)
    if not os.path.isdir(ct_dir):
        return []
    out = []
    for fn in os.listdir(ct_dir):
        ip = os.path.join(ct_dir, fn)
        mp = os.path.join(mask_dir, fn)
        if os.path.isfile(ip) and os.path.exists(mp):
            out.append((ip, mp, fn))
    out.sort(key=lambda x: x[2])
    return out


def create_temp_video_dir(slice_paths, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for local_idx, (ip, _, _) in enumerate(slice_paths):
        img = io.imread(ip)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = (img - img.min()) / np.clip(img.max() - img.min(), 1e-8, None) * 255.0
            img = img.astype(np.uint8)
        io.imsave(os.path.join(temp_dir, f"{local_idx:05d}.jpg"), img)
    return temp_dir


def load_image_normalized_512(image_path):
    """Load PNG, resize to 512x512, normalize with ImageNet stats (same as training)."""
    img = Image.open(image_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (512, 512, 3)
    arr = arr.transpose(2, 0, 1)                       # (3, 512, 512)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr  # (3, 512, 512) float32


def mask_to_512_binary(mask):
    """Resize a HxW mask to 512x512 with NN, return float32 {0,1}."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask_bin = (mask > 127).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)
    m = cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.float32)


# ----------------------------------------------------------------------------
# The dedicated refine-ready predictor
# ----------------------------------------------------------------------------
class MedSAM2VideoRefinePredictor:
    """Wraps the SAM2 video predictor + TRACE, loading a combined ckpt."""

    def __init__(self, config_file: str, ckpt_path: str, device: str = "cuda:0",
                 refine_iters: int = 3):
        self.device = torch.device(device)
        self.refine_iters = refine_iters

        # Build SAM2 video predictor WITHOUT auto-loading (we load the merged ckpt below).
        self.video_predictor = build_sam2_video_predictor(
            config_file=config_file, ckpt_path=None, device=self.device, mode="eval"
        )
        # Build TRACE with its own ImageNet init; will overwrite via ckpt.
        self.refinement = TRACE(
            config=SimpleNamespace(classifier=None),
            img_size=512,
            pretrained=False,
        ).to(self.device)

        # Load training ckpt, split keys by prefix.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        sam2_state = {k: v for k, v in state.items() if not k.startswith("refinement.")}
        refine_state = {k[len("refinement."):]: v for k, v in state.items()
                        if k.startswith("refinement.")}
        missing, unexpected = self.video_predictor.load_state_dict(sam2_state, strict=False)
        print(f"[ckpt] SAM2 video predictor: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        if missing:
            print("  (first 5 missing keys):", missing[:5])
        if unexpected:
            print("  (first 5 unexpected keys):", unexpected[:5])
        self.refinement.load_state_dict(refine_state, strict=True)

        self.video_predictor.eval()
        self.refinement.eval()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _refine_slice(self, cur_img_512, cur_pred_512, ref_img_512, ref_pred_512, ref_gt_512):
        """All inputs are torch tensors (B=1) at 512x512; returns refined binary (512,512)."""
        target_2ch = torch.cat([cur_img_512[:, 0:1], cur_pred_512], dim=1)       # (1,2,512,512)
        ref_3ch = torch.cat([ref_img_512[:, 0:1], ref_pred_512, ref_gt_512], dim=1)  # (1,3,512,512)

        logits = self.refinement(target_2ch, ref_3ch)
        for _ in range(1, self.refine_iters):
            nxt = torch.sigmoid(logits)
            target_2ch = torch.cat([cur_img_512[:, 0:1], nxt], dim=1)
            logits = self.refinement(target_2ch, ref_3ch)
        return (logits > 0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_patient(self, case_num, slice_paths, middle_idx, middle_mask_path,
                        temp_base_dir):
        """
        Returns dict {original_slice_idx: refined_mask_at_origHxW (uint8, binary)}.
        """
        # Middle slice & box prompt
        middle_mask = cv2.imread(middle_mask_path, cv2.IMREAD_GRAYSCALE)
        middle_image_path = middle_mask_path.replace("Mask", "CT")
        middle_img_np = io.imread(middle_image_path)
        H = middle_img_np.shape[0]
        W = middle_img_np.shape[1]
        box_prompt = extract_box_from_mask(middle_mask, image_size=512)

        # Two-half propagation (same structure as test_medsam2_2d_video.py)
        part1_slices = slice_paths[:middle_idx + 1]
        part2_slices = slice_paths[middle_idx:]

        # raw_logits[original_idx] = video_res_masks[0, 0] at video_H x video_W (== orig HxW)
        raw_logits = {}

        def propagate(slices, start_local_idx, reverse, mapping):
            tmp_dir = os.path.join(temp_base_dir, f"{case_num}_tmp_{'rev' if reverse else 'fwd'}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            create_temp_video_dir(slices, tmp_dir)
            state = self.video_predictor.init_state(video_path=tmp_dir, async_loading_frames=False)
            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=start_local_idx,
                obj_id=0,
                box=box_prompt.astype(np.float32),
                normalize_coords=True,
            )
            for local_idx, _, video_res_masks in self.video_predictor.propagate_in_video(
                state, start_frame_idx=start_local_idx, reverse=reverse
            ):
                raw_logits[mapping(local_idx)] = video_res_masks[0, 0].cpu().numpy()
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if len(part1_slices) > 0:
            propagate(part1_slices, start_local_idx=len(part1_slices) - 1,
                      reverse=True, mapping=lambda li: li)
        if len(part2_slices) > 0:
            propagate(part2_slices, start_local_idx=0,
                      reverse=False, mapping=lambda li: middle_idx + li)

        # Build refinement reference (middle slice)
        ref_img_512 = torch.from_numpy(load_image_normalized_512(middle_image_path)).unsqueeze(0).to(self.device)
        mid_gt_512 = torch.from_numpy(mask_to_512_binary(middle_mask)).unsqueeze(0).unsqueeze(0).to(self.device)
        # middle pred: take the logits we already collected and resize to 512
        mid_logits = raw_logits.get(middle_idx)
        if mid_logits is None:
            raise RuntimeError(f"No middle-slice logits collected for {case_num}")
        mid_pred_512 = torch.from_numpy(mid_logits).unsqueeze(0).unsqueeze(0).to(self.device)
        mid_pred_512 = F.interpolate(mid_pred_512, size=(512, 512), mode="bilinear", align_corners=False)
        mid_pred_512 = torch.sigmoid(mid_pred_512)

        # Refine each slice
        refined_at_orig = {}
        for original_idx in sorted(raw_logits.keys()):
            image_path = slice_paths[original_idx][0]
            cur_img_512 = torch.from_numpy(load_image_normalized_512(image_path)).unsqueeze(0).to(self.device)
            cur_logits = raw_logits[original_idx]
            cur_pred_512 = torch.from_numpy(cur_logits).unsqueeze(0).unsqueeze(0).to(self.device)
            cur_pred_512 = F.interpolate(cur_pred_512, size=(512, 512), mode="bilinear", align_corners=False)
            cur_pred_512 = torch.sigmoid(cur_pred_512)

            refined_512 = self._refine_slice(
                cur_img_512, cur_pred_512,
                ref_img_512, mid_pred_512, mid_gt_512
            )  # (512, 512) uint8 binary
            refined_orig = cv2.resize(
                refined_512.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            )
            refined_at_orig[original_idx] = refined_orig
        return refined_at_orig


# ----------------------------------------------------------------------------
# Dataset evaluation loop
# ----------------------------------------------------------------------------
@torch.inference_mode()
def eval_dataset(predictor: MedSAM2VideoRefinePredictor, root_dir: str,
                 annotation_path: str):
    ref_map = {}
    if os.path.exists(annotation_path):
        ref_map = json.load(open(annotation_path))

    ct_dir = os.path.join(root_dir, "CT")
    case_nums = sorted(d for d in os.listdir(ct_dir)
                       if os.path.isdir(os.path.join(ct_dir, d)))

    total_iou = 0.0
    total_dice = 0.0
    n = 0
    temp_root = tempfile.mkdtemp(prefix="medsam2_video_refine_")

    try:
        for case in tqdm(case_nums, desc="Processing patients"):
            slice_paths = organize_patient_slices(root_dir, case)
            if not slice_paths:
                continue
            info = ref_map.get(case)
            middle_mask_path = None
            if isinstance(info, dict):
                middle_mask_path = info.get("mask_path")
            if not middle_mask_path or not os.path.exists(middle_mask_path):
                print(f"[WARN] no middle-slice mask for {case}, skip")
                continue
            middle_filename = os.path.basename(middle_mask_path)
            middle_idx = None
            for idx, (_, _, fn) in enumerate(slice_paths):
                if fn == middle_filename:
                    middle_idx = idx
                    break
            if middle_idx is None:
                print(f"[WARN] middle slice {middle_filename} not found in {case}")
                continue

            try:
                refined = predictor.process_patient(
                    case, slice_paths, middle_idx, middle_mask_path, temp_root
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[ERR] processing {case}: {e}")
                continue

            processed_middle = False
            for original_idx, (_, mask_path, _) in enumerate(slice_paths):
                if original_idx not in refined:
                    continue
                if original_idx == middle_idx:
                    if processed_middle:
                        continue
                    processed_middle = True
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    continue
                pred = refined[original_idx]
                iou, dice = compute_metrics(pred, (gt > 127).astype(np.uint8))
                total_iou += iou; total_dice += dice; n += 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if n == 0:
        return 0.0, 0.0, 0
    return total_iou / n, total_dice / n, n


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["kits", "pancreas", "lits", "colon", "local"])
    ap.add_argument("--checkpoint", required=True, help="Path to training checkpoint.pt")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml",
                    help="Inference-side Hydra config (top-level `model:` structure). "
                         "Not the training yaml.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--refine-iters", type=int, default=3)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    ckpt_abs = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(root, args.checkpoint)
    print(f"[INFO] config={args.cfg}  ckpt={ckpt_abs}")

    predictor = MedSAM2VideoRefinePredictor(
        config_file=args.cfg, ckpt_path=ckpt_abs,
        device=args.device, refine_iters=args.refine_iters,
    )

    dataset_dir = os.path.join("./2D_data", args.data, "test")
    annotation_path = os.path.join(dataset_dir, "annotation_dict_middle.json")
    print(f"[INFO] evaluating on {dataset_dir}")

    iou, dice, n = eval_dataset(predictor, dataset_dir, annotation_path)
    print(f"\n=== {args.data} (video + refine) ===")
    print(f"  n slices: {n}")
    print(f"  avg IoU : {iou:.4f}")
    print(f"  avg Dice: {dice:.4f}")

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "dataset": args.data,
        "mode": "video_refine",
        "checkpoint": ckpt_abs,
        "avg_iou": float(iou),
        "avg_dice": float(dice),
        "n_slices": int(n),
    }
    out_path = os.path.join(results_dir, f"results_{args.data}_video_refine.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[OK] saved → {out_path}")


if __name__ == "__main__":
    main()
