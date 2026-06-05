import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .datasets.tumor_dataset import *
from .utils import test_single_volume
from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import TransUNet_ours
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .networks.medformer import MedFormer,MedFormer_ours
from .networks.attention_unet import AttentionUNet,AttentionUNet_ours
from .networks.unetpp import UNetPlusPlus,UNetPlusPlus_ours
from .networks.swin_unet import SwinUnet,SwinUnet_ours
from .networks.swin_unet import SwinUnet_config
from .networks.FAT_Net import FAT_Net,FATNet_ours
from .networks.H2Former import res34_swin_MS,H2Former_ours
from PIL import Image




def compute_metrics(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    total = (pred + target).sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (total + smooth)
    
    return iou.item(), dice.item()

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.root_path, mode='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    total_iou = 0.0
    total_dice = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), desc="testing", ncols=80):
        image, label, ref_image, ref_mask, _, image_path = (
            sampled_batch["image"].unsqueeze(0).cuda(),
            sampled_batch["label"].cuda(),
            sampled_batch['ref_image'].unsqueeze(1).cuda(),
            sampled_batch['ref_mask'].unsqueeze(1).cuda(),
            sampled_batch['case_name'],
            sampled_batch['image_path'],
        )
        if 'ours' in args.exp_name:
            outputs = model(image, ref_image, ref_mask)
        else:
            outputs = model(image)
        if isinstance(outputs, dict):
            outputs = outputs['final']
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)


        # Resize prediction and label back to original resolution before computing metrics
        orig_img = Image.open(image_path[0]).convert("L")
        orig_h, orig_w = orig_img.size[1], orig_img.size[0]  # PIL: (W, H)
        cur_h, cur_w = out.shape[-2], out.shape[-1]
        if orig_h != cur_h or orig_w != cur_w:
            from scipy.ndimage import zoom as scipy_zoom
            out_np = out.squeeze().cpu().numpy().astype(np.float32)
            label_np = label.squeeze().cpu().numpy().astype(np.float32)
            out_np = scipy_zoom(out_np, (orig_h / cur_h, orig_w / cur_w), order=0)
            label_np = scipy_zoom(label_np, (orig_h / cur_h, orig_w / cur_w), order=0)
            out_t = torch.from_numpy((out_np > 0.5).astype(np.float32))
            label_t = torch.from_numpy((label_np > 0.5).astype(np.float32))
            iou, dice = compute_metrics(out_t, label_t)
        else:
            iou, dice = compute_metrics(out, label)
        total_iou += iou
        total_dice += dice

        #save the image slice by slice, do not use test_single_volume
        
        if test_save_path is not None:
            save_folder = os.path.join(test_save_path, image_path[0].split('/')[-2])
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, image_path[0].split('/')[-1])
            prediction = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(prediction).save(save_path)
            
            



    avg_iou = total_iou / len(db_test)
    avg_dice = total_dice / len(db_test)
    
    logging.info('mean_dice=%f mean_iou=%f' % (avg_dice, avg_iou))
    print(f"Mean Dice: {avg_dice:.4f} | Mean IoU: {avg_iou:.4f}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='', help='root dir for validation volume data')  # for acdc volume_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_save', action="store_true", help='whether to save results during inference')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--test_save_dir', type=str, default='./predictions', help='directory to save predicted masks (PNG)')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--exp_name', type=str,
                        default='TU', help='experiment_name')
    parser.add_argument('--train_ref', type=str, default='middle', choices=['neighbor', 'middle'], help='choose the reference slice')
    parser.add_argument('--test_ref', type=str, default='middle', choices=['neighbor', 'middle'], help='choose the reference slice')
    parser.add_argument('--train_dataset', type=str, default=None, help='training dataset name for OOD testing (overrides --dataset in checkpoint path)')
    parser.add_argument('--has_confidence', action="store_true", help='whether to use confidence map in refinement module')
    parser.add_argument('--deep_supervision', action='store_true', help='whether to use deep supervision')
    args = parser.parse_args()
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_config = {
        'kits': {
            'root_path': './2D_data/kits/test',
            'num_classes': 2,
        },
        'pancreas': {
            'root_path': './2D_data/pancreas/test',
            'num_classes': 2,
        },
        'lits': {
            'root_path': './2D_data/lits/test',
            'num_classes': 2,
        },
        'colon': {
            'root_path': './2D_data/colon/test',
            'num_classes': 2,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    if args.test_ref == 'middle':
        args.Dataset = Dataset_middle
    elif args.test_ref == 'neighbor':
        args.Dataset = Dataset_neighbor
    else:
        raise ValueError("--test_ref must be 'middle' or 'neighbor', got %r" % args.test_ref)
    args.is_pretrain = True
    # name the same snapshot defined in train script!
    # For OOD testing: --train_dataset overrides --dataset in checkpoint path
    ckpt_dataset = args.train_dataset if args.train_dataset else dataset_name
    # Mirror train.py: baseline has no `_<ref>_` token; `_ours` encodes train_ref.
    if '_ours' in args.exp_name:
        args.exp = args.exp_name + '_' + args.train_ref + '_' + ckpt_dataset + str(args.img_size)
    else:
        args.exp = args.exp_name + '_' + ckpt_dataset + str(args.img_size)

    snapshot_path = "./checkpoints/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    if args.exp_name == 'transunet':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.exp_name == 'transunet_ours':
        config_small = CONFIGS_ViT_seg['R18-ViT-S_16']
        config_small.n_classes = args.num_classes
        config_small.n_skip = args.n_skip
        config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = TransUNet_ours(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.exp_name == 'medformer':
        net = MedFormer(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'medformer_ours':
        net = MedFormer_ours(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'attention_unet':
        net = AttentionUNet(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'attention_unet_ours':
        net = AttentionUNet_ours(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'unetpp':
        net = UNetPlusPlus(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'unetpp_ours':
        net = UNetPlusPlus_ours(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'swin_unet':
        net = SwinUnet(SwinUnet_config(), img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'swin_unet_ours':
        net = SwinUnet_ours(SwinUnet_config(), img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'FAT_Net':
        net = FAT_Net(n_channels=1, n_classes=args.num_classes).cuda()
    elif args.exp_name == 'FAT_Net_ours':
        net = FATNet_ours(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'H2Former':
        net = res34_swin_MS(image_size=args.img_size, num_class=args.num_classes).cuda()
    elif args.exp_name == 'H2Former_ours':
        net = H2Former_ours(img_size=args.img_size, num_classes=args.num_classes).cuda()
    else:
        raise Exception('Invalid experiment name, cannot find appropriate model')
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))

    
    snapshot_name = snapshot_path.split('/')[-1]
    # For OOD: append test dataset to log/save paths to avoid overwriting
    ood_suffix = '_testOn_' + dataset_name if args.train_dataset else ''
    log_folder = './test_log/test_log_' + args.exp + ood_suffix
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    if args.is_save:
        # args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp + ood_suffix, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


