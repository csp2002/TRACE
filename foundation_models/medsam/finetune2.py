# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
csp: finetune the original medsam without reference mask (use middle slice GT to extract bounding box)
"""

# %% setup environment
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


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

class Dataset(Dataset):   
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
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

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(image_path)
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0


        x, y = image.shape
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2]==4:
            img_3c = image[:,:,:3]
        else:
            img_3c = image
        # if x != self.output_size[0] or y != self.output_size[1]:
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        mask_1024 = transform.resize(
                mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        
        # case_num, slice_num = mask_path.split('/')[-2:]
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # folder_name = os.path.dirname(mask_path)
        # if not os.path.exists(ref_mask_path):
        #     ref_mask_path = dict[folder_name]
        
        # ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        # ref_mask = ref_mask / 255.0
        # ref_x, ref_y = ref_mask.shape
        # # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        # ref_mask_1024 = transform.resize(
        #         ref_mask,
        #         (1024,1024),
        #         order=0,
        #         preserve_range=True,
        #         mode="constant",
        #         anti_aliasing=False,
        #     )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(mask_1024 > 0)
        
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
            
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
       

        
        # add perturbation to bounding box coordinates
        H, W = mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            img_name,
        )
        

class Dataset_vol2flow(Dataset):   #use pseudo label as reference mask, the input only includes original image and reference/pseudo mask
    def __init__(self, base_dir, mode):
        
        # self.split = split
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
        self.dataset = base_dir.split('/')[-1]
        self.mode = mode
        self.bbox_shift = 20
        self.image_paths, self.mask_paths = self._get_image_mask_paths()
    
    def _get_image_mask_paths(self):
        image_paths = []
        mask_paths = []
        ct_dir = os.path.join(self.data_dir, self.mode, "CT")
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

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img_name = os.path.basename(image_path)
        # print('image_path:', image_path)
        # print('mask_path:', mask_path)
        # raise Exception
        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('image:', image.shape, image.max(), image.min())
        # print('mask:', mask.shape, mask.max(), mask.min())
        # # print('image0:', image.shape, image.max(), image.min())
        # normalize image and mask
        # image = (image - image.min()) / (image.max() - image.min())
        mask = mask / 255.0


        x, y = image.shape
        if len(image.shape) == 2:
            img_3c = np.repeat(image[:, :, None], 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2]==4:
            img_3c = image[:,:,:3]
        else:
            img_3c = image
        # if x != self.output_size[0] or y != self.output_size[1]:
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        mask_1024 = transform.resize(
                mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        
        case_num, slice_num = mask_path.split('/')[-2:]
        ref_mask_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse',self.dataset+'_result',self.dataset, self.mode, 'Mask',case_num,slice_num)
        
        dict_path = os.path.join('./Vol2Flow_me/codes/models',self.dataset,'Vol2Flow_depth:256_M:5_mse','annotation_dict_'+self.mode+'.json')
        # ref_mask_path = os.path.join('./Sli2Vol/Sli2Vol_result', self.dataset, case_num, slice_num)
        # dict_path = os.path.join('./Sli2Vol/result', self.dataset+'_train', 'annotation_dict_'+self.mode+'.json')
        with open(dict_path, 'r') as f:
            dict = json.load(f)
        folder_name = os.path.dirname(mask_path)
        if not os.path.exists(ref_mask_path):
            ref_mask_path = dict[folder_name]
        
        ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
        ref_mask = ref_mask / 255.0
        ref_x, ref_y = ref_mask.shape
        # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
        ref_mask_1024 = transform.resize(
                ref_mask,
                (1024,1024),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
        # print('ref_mask_1024:', ref_mask_path, ref_mask_1024.max(), ref_mask_1024.min())
        # y_indices, x_indices = np.where(ref_mask_1024 > 0)
        
        y_indices, x_indices = np.where(ref_mask_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            ref_mask_path = dict[folder_name]
            ref_mask = np.array(Image.open(ref_mask_path).convert("L"), dtype=np.float32)
        # ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min())
            ref_mask = ref_mask / 255.0
            ref_x, ref_y = ref_mask.shape
            # ref_image = zoom(ref_image, (224 / ref_x, 224 / ref_y), order=3)
            ref_mask_1024 = transform.resize(
                    ref_mask,
                    (1024,1024),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
            y_indices, x_indices = np.where(ref_mask_1024 > 0)
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

       
            
        # folder_name = os.path.dirname(mask_path)
        # dict_path = os.path.join(self.data_dir, 'largest_slice.json')
        # with open(dict_path, 'r') as f:
        #     dict = json.load(f)
        # ref_mask_path = dict[folder_name]
        # ref_image_path = ref_mask_path.replace('Mask', 'CT')
        # print('ref_image_path:', ref_image_path)
        # print('ref_mask_path:', ref_mask_path)
        # ref_image = np.array(Image.open(ref_image_path).convert("L"), dtype=np.float32)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min())
        
       

        
        # add perturbation to bounding box coordinates
        H, W = ref_mask_1024.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
            #倒数第二个文件夹名作为case_name
       
        #打印sample的所有key
        # print('sample:', sample.keys())
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(mask_1024[None, :, :]).float(),
            torch.tensor(bboxes).float(),
            img_name,
        )
        
# %% sanity test of dataset class
# tr_dataset = NpyDataset("data/npy/CT_Abd")
# tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
# for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
#     print(image.shape, gt.shape, bboxes.shape)
#     # show the example
#     _, axs = plt.subplots(1, 2, figsize=(25, 25))
#     idx = random.randint(0, 7)
#     axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
#     show_mask(gt[idx].cpu().numpy(), axs[0])
#     show_box(bboxes[idx].numpy(), axs[0])
#     axs[0].axis("off")
#     # set title
#     axs[0].set_title(names_temp[idx])
#     idx = random.randint(0, 7)
#     axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
#     show_mask(gt[idx].cpu().numpy(), axs[1])
#     show_box(bboxes[idx].numpy(), axs[1])
#     axs[1].axis("off")
#     # set title
#     axs[1].set_title(names_temp[idx])
#     # plt.show()
#     plt.subplots_adjust(wspace=0.01, hspace=0)
#     plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
#     plt.close()
#     break

# %% set up parser
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-i",
#     "--tr_npy_path",
#     type=str,
#     default="data/npy/CT_Abd",
#     help="path to training npy files; two subfolders: gts and imgs",
# )
parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon","local"]
    )
parser.add_argument("--task_name", type=str, default="finetune_middle")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
# parser.add_argument(
#     "--checkpoint", type=str, default="medsam_vit_b.pth"
# )
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("--pretrain_model_path", type=str, default="medsam_vit_b.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
# train
parser.add_argument("--num_epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=8)   #csp change it from 0 to 8
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
parser.add_argument("--use_pseudo_label", action="store_true", default=False, help="use pseudo labels from vol2flow")
parser.add_argument('--ref', type=str, default='middle', choices=['largest', 'random', 'neighbor','middle'], help='choose the reference slice')
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

args.task_name = args.task_name + "-" + args.data
# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding, features = self.image_encoder(image)  # (B, 256, 64, 64)
        f0, f1, f2 = features  # bs,64,64,768
        # print('f0:', f0.shape, 'f1:', f1.shape, 'f2:', f2.shape)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

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


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    # print(type(sam_model))
    # sam_model = sam_model_registry[args.model_type]
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    )
    if args.load_pretrain:
    #从checkpoint里加载参数，不严格要求匹配
        medsam_model.load_state_dict(
            torch.load(args.pretrain_model_path, map_location=device),
            strict=True,
        )
        print("load pretrain model from: ", args.pretrain_model_path)
    medsam_model = medsam_model.to(device)
    medsam_model.train()

    # print(
    #     "Number of total parameters: ",
    #     sum(p.numel() for p in medsam_model.parameters()),
    # )  # 93735472
    # print(
    #     "Number of trainable parameters: ",
    #     sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    # )  # 93729252
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


    trainable_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    # + list(medsam_model.refinement.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in trainable_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10

    dataset_config = {
        'local': {
            'root_path': './2D_data/local',
            'num_classes': 2,
        },
        'kits': {
            'root_path': './2D_data/kits',
            'num_classes': 2,
        },
        'pancreas': {
            'root_path': './2D_data/pancreas',
            'num_classes': 2,
        },
        'lits': {
            'root_path': './2D_data/lits',
            'num_classes': 2,
        },
        'colon': {
            'root_path': './2D_data/colon',
            'num_classes': 2,
        },
    }
    root_path = dataset_config[args.data]['root_path']
    if args.use_pseudo_label:
        train_dataset = Dataset_vol2flow(
            base_dir=root_path,
            mode='train'
        )
        print('Using pseudo labels from vol2flow!')
    elif args.ref == 'middle':
        train_dataset = Dataset_v6_2(
                base_dir=root_path,
                mode='train'
            )
        print('Using box prompt extracted from middle slice GT!')
    elif args.ref == 'neighbor':
        train_dataset = Dataset_v9(
            base_dir=root_path,
            mode='train'
        )
        print('Using box prompt extracted from neighbor slice GT!')
    
    

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
        for step, (image, gt2D, boxes, _, _, _)  in enumerate(tqdm(train_dataloader)):
            # print('image:', image.shape, image.max(), image.min())
            # print('gt2D:', gt2D.shape, gt2D.max(), gt2D.min())
            # print('boxes:', boxes.shape, boxes.max(), boxes.min())
            # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
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
                medsam_pred = medsam_model(image, boxes_np)
                # print('medsam_pred:', medsam_pred.shape, medsam_pred.max(), medsam_pred.min())
                # raise Exception
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
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
