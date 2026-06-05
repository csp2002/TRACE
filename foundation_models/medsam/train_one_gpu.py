# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
finetune MedSAM + TRACE refinement (with reference mask)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import json
from PIL import Image
from My_utils import *
import vit_seg_configs as configs

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


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



        

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
parser.add_argument(
    "--task_name", type=str, default=None,
    help="Override the auto-generated task_name. If omitted, derived from --use_trace + --ref "
         "as 'with_TRACE_<ref>' or 'finetune_<ref>'.",
)
parser.add_argument(
    "--use_trace", action="store_true", default=False,
    help="Train MedSAM with the TRACE refinement add-on (model2). Without this flag, train baseline MedSAM (model1).",
)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="medsam_vit_b.pth",
    help="Architecture seed; loaded via segment_anything's sam_model_registry. "
         "Default is the MedSAM-finetuned ckpt itself (shape-compatible with SAM ViT-B). "
         "The MedSAM weights are reloaded with strict=False right after via --pretrain_model_path, "
         "so this just avoids the upstream Facebook-AI download prompt that fires when the basename "
         "is 'sam_vit_b_01ec64.pth' and the file is missing.",
)
parser.add_argument("--pretrain_model_path", type=str, default="medsam_vit_b.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
# train
parser.add_argument("--num_epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=8)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--freeze", action="store_true", default=False, help="freeze MedSAM")
parser.add_argument('--ref', type=str, default='neighbor', choices=['largest','neighbor','middle'], help='choose the reference slice')
args = parser.parse_args()

# Auto-construct task_name to match the layout simulation.py expects:
#   baseline → finetune_<ref>-<dataset>-<timestamp>/medsam_model_best.pth
#   +TRACE   → with_TRACE_<ref>-<dataset>-<timestamp>/medsam_model_best.pth
if args.task_name is None:
    args.task_name = f"with_TRACE_{args.ref}" if args.use_trace else f"finetune_{args.ref}"

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.data,
            "model_type": args.model_type,
        },
    )
# (optional wandb monitoring is gated by --use_wandb above)
args.task_name = args.task_name + "-" + args.data
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model



def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    if args.use_trace:
        config_small = configs.get_r18_s16_config()
        config_small.n_classes = 2
        config_small.n_skip = 3
        config_small.patches.grid = (int(1024 / 16), int(1024 / 16))
        medsam_model = MedSAM_with_TRACE(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            refinement=TRACE(config_small),
        )
        print('Using MedSAM + TRACE')
    else:
        medsam_model = MedSAM_Wrapper(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        )
        print('Using baseline MedSAM')

    
    medsam_model.load_state_dict(
        torch.load(args.pretrain_model_path, map_location=device),
        strict=False,
    )
    print("load pretrain model from: ", args.pretrain_model_path)
    if args.freeze:
        for param in medsam_model.parameters():
            param.requires_grad = False
        for param in medsam_model.refinement.parameters():
            param.requires_grad = True
        print("Have frozen MedSAM!")
    medsam_model = medsam_model.to(device)
    medsam_model.train()

    print(
    "Number of total parameters: {:.2f}M".format(
        sum(p.numel() for p in medsam_model.parameters()) / 1e6
    )
    )
    print(
        "Number of trainable parameters: {:.2f}M".format(
            sum(p.numel() for p in medsam_model.parameters() if p.requires_grad) / 1e6
        )
    )
    #

    trainable_params = [p for p in medsam_model.parameters() if p.requires_grad]
    #print number of trainable parameters
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10

    # 2D_data/ lives at the repo root; anchor paths to this file so the script
    # works regardless of where the user runs it from (e.g., README example
    # does `cd foundation_models/medsam` first, which broke the old relative
    # './2D_data/...' paths).
    _data_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '2D_data'
    )
    dataset_config = {
        'kits':     {'root_path': os.path.join(_data_root, 'kits'),     'num_classes': 2},
        'pancreas': {'root_path': os.path.join(_data_root, 'pancreas'), 'num_classes': 2},
        'lits':     {'root_path': os.path.join(_data_root, 'lits'),     'num_classes': 2},
        'colon':    {'root_path': os.path.join(_data_root, 'colon'),    'num_classes': 2},
    }
    root_path = dataset_config[args.data]['root_path']
    if args.ref == 'neighbor':
        train_dataset = Dataset_neighbor(
            base_dir=root_path,
            mode='train'
        )
        print('Using neighbor-slice reference dataset')
    elif args.ref == 'middle':
        train_dataset = Dataset_middle(
            base_dir=root_path,
            mode='train'
        )
        print('Using middle-slice reference dataset')
    else:
        raise ValueError(
            f"Unsupported --ref value: {args.ref!r}; expected 'middle' or 'neighbor'"
        )

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, ref_img, ref_gt, img_name)  in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D, ref_img, ref_gt = image.to(device), gt2D.to(device), ref_img.to(device), ref_gt.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                
                if args.use_trace:
                    # medsam_pred, ori_mask, ref_mask = medsam_model(image, boxes_np, ref_img, ref_gt)
                    outputs = medsam_model(image, boxes_np, ref_img, ref_gt)


                    loss_ce = 0
                    loss_seg = 0
                    for i, logits_i in enumerate(outputs['iters']):
                        loss_seg += seg_loss(logits_i, gt2D)
                        loss_ce += ce_loss(logits_i, gt2D.float())
                    loss = loss_seg + loss_ce
                else:
                    # Baseline MedSAM (no TRACE refinement): box-prompted forward returns the mask.
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
