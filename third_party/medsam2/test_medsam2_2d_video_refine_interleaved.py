# -*- coding: utf-8 -*-
"""
Interleaved inference variant for SAM2TrainWithRefineInterleaved checkpoints.

For each slice produced by the video predictor's propagation, immediately run
TRACE and inject the refined mask back into the inference state's
memory bank (re-encoding `maskmem_features` and `maskmem_pos_enc` via the
predictor's `_run_memory_encoder`). Subsequent frames' memory_attention then
queries refined-mask memories — matching the training-time interleaved
behavior of `SAM2TrainWithRefineInterleaved`.

Two halves (split at middle) are still used, mirroring the training split-into
forward/reverse-from-middle protocol. Reference for refinement is always the
middle slice's *refined* prediction (and its GT mask), which is cached after
the first iteration of each half.

Output: results/results_<dataset>_video_refineE_interleaved.json (does not
clobber the post-hoc results_<ds>_video_refineE.json).
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

# Custom OmegaConf resolvers (matched to training train.py).
try:
    register_omegaconf_resolvers()
except Exception:
    pass


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ----------------------------------------------------------------------------
# Helpers (mirrored from test_medsam2_2d_video_refine.py)
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
        mask_bin = (mask > 127).astype(np.uint8)
    else:
        mask_bin = (mask > 0.5).astype(np.uint8)
    H, W = mask_bin.shape
    mask_resized = cv2.resize(mask_bin, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
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
    img = Image.open(image_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def mask_to_512_binary(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask_bin = (mask > 127).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)
    m = cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.float32)


# ----------------------------------------------------------------------------
# Interleaved predictor
# ----------------------------------------------------------------------------
class MedSAM2VideoRefineInterleavedPredictor:
    def __init__(self, config_file: str, ckpt_path: str, device: str = "cuda:0",
                 refine_iters: int = 3):
        self.device = torch.device(device)
        self.refine_iters = int(refine_iters)
        self.video_predictor = build_sam2_video_predictor(
            config_file=config_file, ckpt_path=None, device=self.device, mode="eval"
        )
        self.refinement = TRACE(
            config=SimpleNamespace(classifier=None),
            img_size=512,
            pretrained=False,
        ).to(self.device)

        # Load combined ckpt and split.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        sam2_state = {k: v for k, v in state.items() if not k.startswith("refinement.")}
        refine_state = {k[len("refinement."):]: v for k, v in state.items()
                        if k.startswith("refinement.")}
        missing, unexpected = self.video_predictor.load_state_dict(sam2_state, strict=False)
        print(f"[ckpt] SAM2 video predictor: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        self.refinement.load_state_dict(refine_state, strict=True)

        self.video_predictor.eval()
        self.refinement.eval()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _refine_at_512(self, cur_img_512, cur_pred_512, ref_img_512, ref_pred_512, ref_gt_512):
        target_2ch = torch.cat([cur_img_512[:, 0:1], cur_pred_512], dim=1)
        ref_3ch = torch.cat([ref_img_512[:, 0:1], ref_pred_512, ref_gt_512], dim=1)
        logits = self.refinement(target_2ch, ref_3ch)
        for _ in range(1, self.refine_iters):
            nxt = torch.sigmoid(logits)
            target_2ch = torch.cat([cur_img_512[:, 0:1], nxt], dim=1)
            logits = self.refinement(target_2ch, ref_3ch)
        return logits  # (1, 1, 512, 512) raw logits

    @torch.inference_mode()
    def _override_memory(self, inference_state, frame_idx, storage_key,
                         refined_logits_512, is_mask_from_pts):
        """Re-encode memory for `frame_idx` using refined logits and overwrite
        the inference state's stored memory features/pos_enc for that frame."""
        batch_size = self.video_predictor._get_obj_num(inference_state)
        prev = inference_state["output_dict"][storage_key][frame_idx]
        object_score_logits = prev["object_score_logits"]
        new_features, new_pos_enc = self.video_predictor._run_memory_encoder(
            inference_state=inference_state,
            frame_idx=frame_idx,
            batch_size=batch_size,
            high_res_masks=refined_logits_512,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        prev["maskmem_features"] = new_features
        prev["maskmem_pos_enc"] = new_pos_enc

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_patient(self, case_num, slice_paths, middle_idx, middle_mask_path,
                        temp_base_dir):
        """Returns dict {original_slice_idx: refined_binary_mask_at_origHxW}."""
        middle_mask = cv2.imread(middle_mask_path, cv2.IMREAD_GRAYSCALE)
        middle_image_path = middle_mask_path.replace("Mask", "CT")
        middle_img_np = io.imread(middle_image_path)
        H, W = middle_img_np.shape[:2]
        box_prompt = extract_box_from_mask(middle_mask, image_size=512)

        # Reference inputs (constant across the patient).
        ref_img_512 = torch.from_numpy(load_image_normalized_512(middle_image_path)).unsqueeze(0).to(self.device)
        ref_gt_512 = torch.from_numpy(mask_to_512_binary(middle_mask)).unsqueeze(0).unsqueeze(0).to(self.device)

        part1_slices = slice_paths[:middle_idx + 1]
        part2_slices = slice_paths[middle_idx:]
        refined_at_orig = {}

        def run_half(slices, start_local_idx, reverse, mapping):
            tmp_dir = os.path.join(temp_base_dir, f"{case_num}_il_{'rev' if reverse else 'fwd'}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            create_temp_video_dir(slices, tmp_dir)
            state = self.video_predictor.init_state(video_path=tmp_dir, async_loading_frames=False)

            # Box prompt at middle (= start_local_idx within this half).
            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=start_local_idx,
                obj_id=0,
                box=box_prompt.astype(np.float32),
                normalize_coords=True,
            )

            ref_pred_512_cached = None  # frame-0 (= middle) refined pred, set after first iteration

            # propagate_in_video yields one frame at a time, including the cond frame (middle)
            # at the start of the iteration. We refine after each yield and overwrite memory
            # in inference_state so the *next* frame's memory_attention sees refined memories.
            for local_idx, _, video_res_masks in self.video_predictor.propagate_in_video(
                state, start_frame_idx=start_local_idx, reverse=reverse
            ):
                # video_res_masks is (num_objs, 1, video_H, video_W). Resize to 512x512 for refinement.
                cur_512_logits = video_res_masks.unsqueeze(0) if video_res_masks.ndim == 3 else video_res_masks
                cur_512_logits = cur_512_logits.to(self.device).float()
                if cur_512_logits.shape[-1] != 512 or cur_512_logits.shape[-2] != 512:
                    cur_512_logits = F.interpolate(cur_512_logits, size=(512, 512),
                                                    mode="bilinear", align_corners=False)
                cur_pred_512 = torch.sigmoid(cur_512_logits)

                # Image at 512 for this slice.
                img_path = slices[local_idx][0]
                cur_img_512 = torch.from_numpy(load_image_normalized_512(img_path)).unsqueeze(0).to(self.device)

                # Reference for refinement.
                if ref_pred_512_cached is None:
                    # First yield = middle slice; reference is itself.
                    ref_pred_512_for_this = cur_pred_512
                else:
                    ref_pred_512_for_this = ref_pred_512_cached

                refined_logits = self._refine_at_512(
                    cur_img_512, cur_pred_512, ref_img_512, ref_pred_512_for_this, ref_gt_512
                )

                if ref_pred_512_cached is None:
                    ref_pred_512_cached = torch.sigmoid(refined_logits).detach()

                # Decide which storage bucket the frame is in.
                if local_idx in state["output_dict"]["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    is_mask_from_pts = True   # middle was box-prompted
                else:
                    storage_key = "non_cond_frame_outputs"
                    is_mask_from_pts = False

                # Re-encode this frame's memory using refined logits.
                self._override_memory(
                    state, local_idx, storage_key, refined_logits.detach(),
                    is_mask_from_pts=is_mask_from_pts,
                )

                # Output: refined binary at original H×W.
                refined_512_bin = (refined_logits > 0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                refined_orig = cv2.resize(refined_512_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                refined_at_orig[mapping(local_idx)] = refined_orig

            shutil.rmtree(tmp_dir, ignore_errors=True)

        if len(part1_slices) > 0:
            # Reverse-from-middle: middle is at position len-1 in the cropped slices list.
            run_half(part1_slices, len(part1_slices) - 1, reverse=True,
                     mapping=lambda li: li)
        if len(part2_slices) > 0:
            # Forward-from-middle: middle is at position 0 in the cropped slices list.
            run_half(part2_slices, 0, reverse=False,
                     mapping=lambda li: middle_idx + li)

        return refined_at_orig


# ----------------------------------------------------------------------------
# Eval driver
# ----------------------------------------------------------------------------
@torch.inference_mode()
def eval_dataset(predictor: MedSAM2VideoRefineInterleavedPredictor,
                 root_dir: str, annotation_path: str):
    ref_map = {}
    if os.path.exists(annotation_path):
        ref_map = json.load(open(annotation_path))

    ct_dir = os.path.join(root_dir, "CT")
    case_nums = sorted(d for d in os.listdir(ct_dir)
                       if os.path.isdir(os.path.join(ct_dir, d)))

    total_iou = 0.0
    total_dice = 0.0
    n = 0
    temp_root = tempfile.mkdtemp(prefix="medsam2_video_refine_il_")

    try:
        for case in tqdm(case_nums, desc="Patients (interleaved)"):
            slice_paths = organize_patient_slices(root_dir, case)
            if not slice_paths:
                continue
            info = ref_map.get(case)
            middle_mask_path = info.get("mask_path") if isinstance(info, dict) else None
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["kits", "pancreas", "lits", "colon", "local"])
    ap.add_argument("--checkpoint", required=True, help="Path to training checkpoint.pt")
    ap.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml",
                    help="Inference-side Hydra config (top-level `model:`)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--refine-iters", type=int, default=3)
    ap.add_argument("--out-suffix", default="refineE_interleaved",
                    help="Output file suffix: results_<ds>_video_<suffix>.json")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    ckpt_abs = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(root, args.checkpoint)
    print(f"[INFO] config={args.cfg}  ckpt={ckpt_abs}  refine_iters={args.refine_iters}")

    predictor = MedSAM2VideoRefineInterleavedPredictor(
        config_file=args.cfg, ckpt_path=ckpt_abs,
        device=args.device, refine_iters=args.refine_iters,
    )

    dataset_dir = os.path.join("./2D_data", args.data, "test")
    annotation_path = os.path.join(dataset_dir, "annotation_dict_middle.json")
    print(f"[INFO] evaluating on {dataset_dir}")

    iou, dice, n = eval_dataset(predictor, dataset_dir, annotation_path)
    print(f"\n=== {args.data} (video + refine, INTERLEAVED) ===")
    print(f"  n slices: {n}")
    print(f"  avg IoU : {iou:.4f}")
    print(f"  avg Dice: {dice:.4f}")

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "dataset": args.data,
        "mode": f"video_{args.out_suffix}",
        "checkpoint": ckpt_abs,
        "avg_iou": float(iou),
        "avg_dice": float(dice),
        "n_slices": int(n),
    }
    out_path = os.path.join(results_dir, f"results_{args.data}_video_{args.out_suffix}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[OK] saved → {out_path}")


if __name__ == "__main__":
    main()
