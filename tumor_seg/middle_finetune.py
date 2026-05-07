"""Middle-slice test-time fine-tuning baseline.

For each case:
  1. Start from the pretrained TransUNet checkpoint of the corresponding dataset.
  2. Fine-tune a fresh deepcopy of the model for `ft_steps` gradient updates on the
     single (CT, Mask) pair of that case's middle slice (defined by
     `annotation_dict_middle.json`).
  3. Predict every slice of the case with the fine-tuned model and record Dice / IoU.
  4. Discard the temporary model; move on to the next case.

This mirrors the loss, data pipeline, and checkpoint layout of `test_oneshot.py`,
so the two scripts are directly comparable.
"""
import os
import json
import argparse
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
from torch.nn.modules.loss import CrossEntropyLoss

from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .utils import DiceLoss


def compute_metrics(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return iou.item(), dice.item()


def _prep_img(path: str, out: int = 224) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        print(f"Warning: {path} are all zeros.")
        arr = arr * 0.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=3)


def _prep_msk(path: str, out: int = 224) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape
    return zoom(arr, (out / h, out / w), order=0)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


def run_middle_finetune(
    data_root: str,
    model_class,
    model_ckpt: str,
    device: str = "cuda",
    lr: float = 1e-4,
    ft_steps: int = 20,
    max_patients: int = None,
    img_size: int = 224,
):
    ct_root = os.path.join(data_root, "CT")
    mk_root = os.path.join(data_root, "Mask")
    ann_path = os.path.join(data_root, "annotation_dict_middle.json")
    with open(ann_path, "r") as f:
        middle_dict = json.load(f)

    model_base = model_class().to(device)
    model_base.load_state_dict(torch.load(model_ckpt))
    model_base.eval()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(2)

    case_ids = sorted(middle_dict.keys())
    if max_patients is not None:
        case_ids = case_ids[:max_patients]

    dice_all, iou_all = [], []
    per_case_summary = []

    for pid in tqdm(case_ids, desc="cases"):
        ct_dir = os.path.join(ct_root, pid)
        mk_dir = os.path.join(mk_root, pid)
        if not os.path.isdir(ct_dir) or not os.path.isdir(mk_dir):
            tqdm.write(f"[skip] {pid}: CT or Mask directory missing")
            continue

        info = middle_dict[pid]
        mid_ct = info.get("ct_path") or os.path.join(ct_dir, info["middle_filename"])
        mid_mk = info.get("mask_path") or os.path.join(mk_dir, info["middle_filename"])
        if not (os.path.isfile(mid_ct) and os.path.isfile(mid_mk)):
            tqdm.write(f"[skip] {pid}: middle slice files missing")
            continue

        # Fresh per-case model cloned from the pretrained checkpoint.
        model = deepcopy(model_base)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        img_ref = _to_tensor(_prep_img(mid_ct, img_size)).to(device)
        gt_ref = _to_tensor(_prep_msk(mid_mk, img_size)).to(device)
        gt_ref_long = gt_ref.squeeze(1).long()

        model.train()
        for _ in range(ft_steps):
            pred = model(img_ref)
            loss_ce = ce_loss(pred, gt_ref_long)
            loss_dice = dice_loss(pred, gt_ref_long, softmax=True)
            loss = 0.5 * (loss_ce + loss_dice)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Inference on every slice of the case with the fine-tuned model.
        # Forward at img_size (model's training resolution), then resize the
        # prediction and label back to the original PNG resolution before
        # computing Dice/IoU — same convention as test.py.
        model.eval()
        slices = sorted(os.listdir(ct_dir), key=lambda x: int(os.path.splitext(x)[0]))
        case_dice, case_iou = [], []
        with torch.no_grad():
            for fname in slices:
                img_path = os.path.join(ct_dir, fname)
                mk_path = os.path.join(mk_dir, fname.replace("CT", "Mask"))
                if not os.path.isfile(mk_path):
                    continue
                img = _to_tensor(_prep_img(img_path, img_size)).to(device)
                gt = _to_tensor(_prep_msk(mk_path, img_size)).to(device)
                out = torch.argmax(torch.softmax(model(img), dim=1), dim=1)

                orig_w, orig_h = Image.open(img_path).size  # PIL size: (W, H)
                cur_h, cur_w = out.shape[-2], out.shape[-1]
                if orig_h != cur_h or orig_w != cur_w:
                    out_np = out.squeeze().cpu().numpy().astype(np.float32)
                    gt_np = gt.squeeze().cpu().numpy().astype(np.float32)
                    out_np = zoom(out_np, (orig_h / cur_h, orig_w / cur_w), order=0)
                    gt_np = zoom(gt_np, (orig_h / cur_h, orig_w / cur_w), order=0)
                    out_t = torch.from_numpy((out_np > 0.5).astype(np.float32))
                    gt_t = torch.from_numpy((gt_np > 0.5).astype(np.float32))
                    iou, dice = compute_metrics(out_t, gt_t)
                else:
                    iou, dice = compute_metrics(out, gt)
                case_dice.append(dice)
                case_iou.append(iou)

        if case_dice:
            avg_c_dice = float(np.mean(case_dice))
            avg_c_iou = float(np.mean(case_iou))
            per_case_summary.append((pid, len(case_dice), avg_c_dice, avg_c_iou))
            dice_all.extend(case_dice)
            iou_all.extend(case_iou)
            tqdm.write(
                f"  {pid:24s} slices={len(case_dice):4d}  "
                f"Dice={avg_c_dice:.4f}  IoU={avg_c_iou:.4f}"
            )

        del model, optimizer
        torch.cuda.empty_cache()

    print("\n=== Per-case summary (middle-slice fine-tune) ===")
    for pid, n, d, i in per_case_summary:
        print(f"  {pid:24s} slices={n:4d}  Dice={d:.4f}  IoU={i:.4f}")

    print(
        f"\n[middle-FT] ft_steps={ft_steps}, lr={lr}, "
        f"cases={len(per_case_summary)}, slices={len(dice_all)}"
    )
    if dice_all:
        print(f"Average Dice: {np.mean(dice_all):.4f}")
        print(f"Average IoU:  {np.mean(iou_all):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="colon",
        choices=["colon", "kits", "lits", "pancreas", "local"],
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--n_skip", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--ft_steps", type=int, default=20,
        help="gradient steps on the middle slice per case",
    )
    args = parser.parse_args()

    dataset_config = {
        "colon": "./2D_data/colon/test",
        "kits": "./2D_data/kits/test",
        "lits": "./2D_data/lits/test",
        "pancreas": "./2D_data/pancreas/test",
        "local": "./2D_data/local/test",
    }

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 2
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.patch_size, args.patch_size)
    if "R50" in args.vit_name:
        config_vit.patches.grid = (
            args.img_size // args.patch_size,
            args.img_size // args.patch_size,
        )

    def build_model():
        return ViT_seg(config_vit, img_size=args.img_size, num_classes=2)

    model_ckpt = os.path.join(
        "./checkpoints/",
        f"TU_{args.dataset}224/"
        f"TU_pretrain_{args.vit_name}_skip3_epo150_bs24_224/epoch_149.pth",
    )
    assert os.path.isfile(model_ckpt), f"ckpt not found: {model_ckpt}"

    run_middle_finetune(
        data_root=dataset_config[args.dataset],
        model_class=build_model,
        model_ckpt=model_ckpt,
        device="cuda",
        lr=args.lr,
        ft_steps=args.ft_steps,
        max_patients=args.max_patients,
        img_size=args.img_size,
    )
