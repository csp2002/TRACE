import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
from .networks._download import ensure_weight
from .trainer import trainer_tumor



_WEIGHT_URLS = {
    "swin_tiny": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "r50_vit_b16": "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz",
}
_WEIGHT_PATHS = {
    "swin_tiny": "./tumor_seg/networks/swin_tiny_patch4_window7_224.pth",
    "resnet34": "./tumor_seg/networks/resnet34.pth",
    "r50_vit_b16": "./tumor_seg/networks/R50+ViT-B_16.npz",
}

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--exp_name', type=str,
                    default='TU', help='experiment_name')
parser.add_argument('--freeze', 
                    action='store_true', help='load and freeze the original fully trained TransUnet')
parser.add_argument('--auxiliary_loss', action='store_true', help='whether to use auxiliary loss')
parser.add_argument('--has_confidence', action='store_true', help='whether to use confidence map in refinement module')
parser.add_argument('--deep_supervision', action='store_true', help='whether to use deep supervision')
parser.add_argument('--ref', type=str, default='middle', choices=['neighbor', 'middle'], help='choose the reference slice')
args = parser.parse_args()


if __name__ == "__main__":
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
    dataset_name = args.dataset
    dataset_config = {
        'kits': {
            'root_path': './2D_data/kits/train',
            'num_classes': 2,
        },
        'pancreas': {
            'root_path': './2D_data/pancreas/train',
            'num_classes': 2,
        },
        'lits': {
            'root_path': './2D_data/lits/train',
            'num_classes': 2,
        },
        'colon': {
            'root_path': './2D_data/colon/train',
            'num_classes': 2,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    
    args.is_pretrain = True
    # baseline (no `_ours`) trains without a reference slice → its checkpoint
    # path has no `_<ref>_` token. `_ours` variants do encode the ref protocol.
    if '_ours' in args.exp_name:
        args.exp = args.exp_name + '_' + args.ref + '_' + dataset_name + str(args.img_size)
    else:
        args.exp = args.exp_name + '_' + dataset_name + str(args.img_size)
    snapshot_path = "./checkpoints/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    if args.exp_name == 'transunet':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    elif args.exp_name == 'transunet_ours':
        config_small = CONFIGS_ViT_seg['R18-ViT-S_16']
        config_small.n_classes = args.num_classes
        config_small.n_skip = args.n_skip
        config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = TransUNet_ours(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        print('Using TransUNet + TRACE')
    elif args.exp_name == 'medformer':
        print('Using medformer baseline model')
        net = MedFormer(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'medformer_ours':
        print('Using medformer + ours model')
        net = MedFormer_ours(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'attention_unet':
        print('Using attention unet baseline model')
        net = AttentionUNet(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'attention_unet_ours':
        print('Using attention unet + ours model')
        net = AttentionUNet_ours(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'unetpp':
        print('Using unet++ baseline model')
        net = UNetPlusPlus(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'unetpp_ours':
        print('Using unet++ + ours model')
        net = UNetPlusPlus_ours(in_ch=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'swin_unet':
        print('Using swin unet baseline model')
        net = SwinUnet(SwinUnet_config(), img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(ensure_weight(_WEIGHT_URLS["swin_tiny"], _WEIGHT_PATHS["swin_tiny"]))
    elif args.exp_name == 'swin_unet_ours':
        print('Using swin unet + ours model')
        net = SwinUnet_ours(SwinUnet_config(), img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(ensure_weight(_WEIGHT_URLS["swin_tiny"], _WEIGHT_PATHS["swin_tiny"]))
    elif args.exp_name == 'FAT_Net':
        print('Using FAT_Net baseline model')
        net = FAT_Net(n_channels=1, n_classes=args.num_classes).cuda()
    elif args.exp_name == 'FAT_Net_ours':
        print('Using FAT_Net + ours model')
        net = FATNet_ours(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'H2Former':
        print('Using H2Former baseline model')
        net = res34_swin_MS(image_size=args.img_size, num_class=args.num_classes).cuda()
        model_dict = net.state_dict()
        pre_dict = torch.load(ensure_weight(_WEIGHT_URLS["resnet34"], _WEIGHT_PATHS["resnet34"]))
        matched_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(matched_dict)
        net.load_state_dict(model_dict)
    elif args.exp_name == 'H2Former_ours':
        print('Using H2Former + ours model')
        net = H2Former_ours(img_size=args.img_size, num_classes=args.num_classes).cuda()
        model_dict = net.state_dict()
        pre_dict = torch.load(ensure_weight(_WEIGHT_URLS["resnet34"], _WEIGHT_PATHS["resnet34"]))
        matched_dict = {k: v for k, v in pre_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(matched_dict)
        net.load_state_dict(model_dict)
    else:
        raise Exception('Invalid experiment name, cannot find appropriate model')

    if args.exp_name == 'transunet' or args.exp_name == 'transunet_ours' :  #for transunet
        config_vit.pretrained_path = ensure_weight(
            _WEIGHT_URLS["r50_vit_b16"], _WEIGHT_PATHS["r50_vit_b16"]
        )
        net.load_from(weights=np.load(config_vit.pretrained_path))
        
    num_params = sum(p.numel() for name, p in net.named_parameters()if 'refinement_module' in name )
    print('Number of parameters in refinement_module:', num_params/1e6, 'M')
    if args.freeze:
        ckpt_folder = snapshot_path.split('/')[-1]
        args.exp_name + dataset_name + str(args.img_size)
        ckpt_path = os.path.join('../model', 'TU'+ '_' + dataset_name + str(args.img_size), ckpt_folder, 'epoch_149.pth')
        print('loading pretrained model from {}'.format(ckpt_path))
        # Load the checkpoint into the net (non-strict)
        net.load_state_dict(torch.load(ckpt_path), strict=False)
        print(' Successfully loaded pretrained model from {}'.format(ckpt_path))
        # Freeze parameters
        for name, param in net.named_parameters():
            if 'fusion' in name or 'refinement' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    trainer = {dataset_name: trainer_tumor,}
    trainer[dataset_name](args, net, snapshot_path)