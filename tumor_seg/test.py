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
from .datasets.dataset_synapse import *
from .utils import test_single_volume
from .networks.vit_seg_modeling import VisionTransformer as ViT_seg
from .networks.vit_seg_modeling import My_VisionTransformer as My_ViT_seg
from .networks.vit_seg_modeling import My_VisionTransformer_v6554 as My_ViT_seg_v6554
from .networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .networks.medformer import MedFormer,MedFormer_ours
from .networks.attention_unet import AttentionUNet,AttentionUNet_ours
from .networks.unetpp import UNetPlusPlus,UNetPlusPlus_ours
from .networks.swin_unet import SwinUnet,SwinUnet_ours
from .networks.swin_unet import SwinUnet_config
from .networks.FAT_Net import FAT_Net,FATNet_ours
from .networks.MISSFormer import MISSFormer
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
    if not args.n_square:
        db_test = args.Dataset(base_dir=args.root_path, mode='test')
    else:
        db_test = args.Dataset(base_dir=args.root_path, mode='test', k=0)
        print('Using n_square dataset (n(n-1) pairs)')
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("Number of testing samples: ", len(testloader))
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    # metric_list = 0.0
    total_iou = 0.0
    total_dice = 0.0
    th_70 = 0
    th_75 = 0
    th_80 = 0
    th_85 = 0
    th_90 = 0

    from collections import defaultdict
    import matplotlib.pyplot as plt

    distance_dice_dict = defaultdict(list)

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        if args.n_square:
            image, label, ref_image, ref_mask, distance, image_path = sampled_batch["image"].unsqueeze(0).cuda(), sampled_batch["label"].cuda(), sampled_batch['ref_image'].unsqueeze(1).cuda(), sampled_batch['ref_mask'].unsqueeze(1).cuda(), sampled_batch['distance'].cuda(), sampled_batch['image_path']  #img:1,1,224,224 label:1,224,224
        else:
            image, label, ref_image, ref_mask, _, image_path = sampled_batch["image"].unsqueeze(0).cuda(), sampled_batch["label"].cuda(), sampled_batch['ref_image'].unsqueeze(1).cuda(), sampled_batch['ref_mask'].unsqueeze(1).cuda(), sampled_batch['case_name'], sampled_batch['image_path']
        
        # #print shape max min
        # print('image:', image.shape, image.max(), image.min()) # (1,1,224,224)
        # print('label:', label.shape, label.max(), label.min())   #(1,224,224)
        # print('ref_image:', ref_image.shape, ref_image.max(), ref_image.min()) #(1,1,224,224)
        # print('ref_mask:', ref_mask.shape, ref_mask.max(), ref_mask.min())  # (1,1,224,224)
        # print('distance:', distance.shape, type(distance), distance.max(), distance.min())
        
        if args.exp_name == 'TU' or args.exp_name == 'medformer' or args.exp_name == 'attention_unet' or args.exp_name == 'unetpp' or args.exp_name == 'swin_unet' or args.exp_name == 'FAT_Net' or args.exp_name == 'H2Former':
            outputs = model(image) # bs,class_num,224,224  #used in the original TransUNet
        # elif not args.has_confidence:
        #     outputs = model(image, ref_image, ref_mask)
        elif args.exp_name == 'v2' or args.exp_name == 'v5' or args.exp_name == 'v6' or args.exp_name == 'v6.1' or args.exp_name == 'v6.2' or args.exp_name == 'v6.3' or 'v6.5' in args.exp_name or 'ours' in args.exp_name :
            outputs = model(image, ref_image, ref_mask)

        elif args.exp_name == 'sli2vol_v3o1' or args.exp_name == 'vol2flow_v3o1':
            outputs = model(image, ref_mask)
        elif args.exp_name == 'sli2vol_v3o2' or args.exp_name == 'vol2flow_v3o2':
            outputs, foreground_mask, modulation_map, ref_mask = model(image, ref_mask)
        elif args.exp_name == 'sli2vol_v4' or args.exp_name == 'vol2flow_v4':
                outputs,  ref_mask, flow = model(image, ref_mask)
        else:
            outputs, refined_mask = model(image, ref_image, ref_mask)  #used when adding my two modules
        # print('outputs:', outputs.shape, outputs.max(), outputs.min())   (1,2,224,224)
        # raise Exception
        if isinstance(outputs, dict):
            outputs = outputs['final']
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        # out = out.cpu().detach().numpy()

        # out= ref_mask.squeeze(1)  # (1,224,224)  #used when using only reference mask as output

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
        if iou<0.2:
            # print(f"iou: {iou}, dice: {dice}")
            print(f"image_path: {image_path}")
        total_iou += iou
        total_dice += dice
        if dice <= 0.7:
            th_70 += 1
        if dice <= 0.75:
            th_75 += 1
        if dice <= 0.8:
            th_80 += 1
        if dice <= 0.85:
            th_85 += 1
        if dice <= 0.9:
            th_90 += 1

        # 记录 (distance, dice)
        if args.n_square:
            dist = int(distance[0])  # 假设 distance 是 batch size = 1 的 tensor
            distance_dice_dict[dist].append(dice)

        #save the image slice by slice, do not use test_single_volume
        # print('image_path:', image_path)
        # print('test_save_path:', test_save_path)
        
        # print('folder:', folder)
        # raise Exception
        if test_save_path is not None:
            save_folder = os.path.join(test_save_path, image_path[0].split('/')[-2])
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, image_path[0].split('/')[-1])
            prediction = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(prediction).save(save_path)
            # print('save_path:', save_path)
            
            



        # metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
        #                               test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # metric_list += np.array(metric_i)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    avg_iou = total_iou / len(db_test)
    avg_dice = total_dice / len(db_test)
    
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_iou : %f' % (avg_dice, avg_iou))
    logging.info('Testing performance in best val model: th_70 : %d th_75 : %d th_80 : %d th_85 : %d th_90 : %d' % (th_70, th_75, th_80, th_85, th_90))
    print("Testing Finished!")

    # if args.n_square:
    #     distance_list = sorted(distance_dice_dict.keys())
    #     data = [distance_dice_dict[d] for d in distance_list]  # 每个 distance 的 dice 列表
    #     labels = [str(d) for d in distance_list]

    #     plt.figure(figsize=(12, 6))
    #     plt.boxplot(data, labels=labels, showfliers=True)  # 显示离群值
    #     plt.ylim(0, 1)
    #     plt.xlabel("Distance to Reference Slice")
    #     plt.ylabel("Dice Score")
    #     plt.title("Dice Distribution per Distance")
    #     plt.grid(True)
    #     plt.tight_layout()

    #     save_folder = os.path.join('./visualization', 'n(n-1)', args.exp_name)
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #     plt.savefig(os.path.join(save_folder, f'{args.dataset}_boxplot.png'))  # 保存图像
    #     plt.close()


        
    #     distance_list = sorted(distance_dice_dict.keys())
    #     mean_dice_list = [np.mean(distance_dice_dict[d]) for d in distance_list]

    #     # 可视化
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(distance_list, mean_dice_list, marker='o')
    #     plt.ylim(0, 1)
    #     plt.xlabel("Distance to Reference Slice")
    #     plt.ylabel("Dice Score")
    #     plt.title("Dice vs. Reference Distance")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     save_folder = os.path.join('./visualization', 'n(n-1)',args.exp_name)
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #     plt.savefig(os.path.join(save_folder, f'{args.dataset}.png'))  # 保存图像
    #     plt.close()  # 关闭当前图像








def inference_multiple_ref(args, model, test_save_path=None):   #used when multiple reference slices
    if not args.n_square:
        db_test = args.Dataset(base_dir=args.root_path, mode='test')
    else:
        db_test = args.Dataset(base_dir=args.root_path, mode='test', k=0)
        print('Using n_square dataset (n(n-1) pairs)')
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    # metric_list = 0.0
    total_iou = 0.0
    total_dice = 0.0
    

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        
        image, label, ref_image1, ref_mask1, ref_image2, ref_mask2, distance1, distance2, case_name, image_path = sampled_batch["image"].unsqueeze(0).cuda(), sampled_batch["label"].cuda(), sampled_batch['ref_image1'].unsqueeze(1).cuda(), sampled_batch['ref_mask1'].unsqueeze(1).cuda(), sampled_batch['ref_image2'].unsqueeze(1).cuda(), sampled_batch['ref_mask2'].unsqueeze(1).cuda(),sampled_batch['distance1'].cuda(), sampled_batch['distance2'].cuda(), sampled_batch['case_name'], sampled_batch['image_path']  #img:1,1,224,224 label:1,224,224

        
        #print shape max min
        # print('image:', image.shape, image.max(), image.min())
        # print('label:', label.shape, label.max(), label.min())
        # print('distance:', distance.shape, type(distance), distance.max(), distance.min())
        if args.strategy == 'closest':
            # 选择距离最近的参考图像和掩码
            if distance1 <= distance2:
                outputs = model(image, ref_image1, ref_mask1)
            else:
                outputs = model(image, ref_image2, ref_mask2)
        elif args.strategy == 'average':
            outputs1 = model(image, ref_image1, ref_mask1)
            outputs2 = model(image, ref_image2, ref_mask2)
            outputs = (outputs1 + outputs2) / 2
        elif args.strategy == 'weighted':
            outputs1 = model(image, ref_image1, ref_mask1)
            outputs2 = model(image, ref_image2, ref_mask2)
            outputs = distance2 / (distance1 + distance2) * outputs1 + distance1 / (distance1 + distance2) * outputs2
        
        # print('outputs:', outputs.shape, outputs.max(), outputs.min())
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        # out = out.cpu().detach().numpy()

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
        # print(f"iou: {iou}, dice: {dice}")
        total_iou += iou
        total_dice += dice

        #save the image slice by slice, do not use test_single_volume
        # print('image_path:', image_path)
        # print('test_save_path:', test_save_path)
        
        # print('folder:', folder)
        # raise Exception
        if test_save_path is not None:
            save_folder = os.path.join(test_save_path, image_path[0].split('/')[-2])
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, image_path[0].split('/')[-1])
            prediction = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(prediction).save(save_path)
            # print('save_path:', save_path)
            
            



      
    avg_iou = total_iou / len(db_test)
    avg_dice = total_dice / len(db_test)
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_iou : %f' % (avg_dice, avg_iou))
    print("Testing Finished!")

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='', help='root dir for validation volume data')  # for acdc volume_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    # parser.add_argument('--list_dir', type=str,
    #                     default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_save', action="store_true", help='whether to save results during inference')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--test_save_dir', type=str, default='../my_predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--exp_name', type=str,
                        default='TU', help='experiment_name')
    parser.add_argument('--train_ref', type=str, default='middle', help='choose the reference slice')
    parser.add_argument('--test_ref', type=str, default='middle', help='choose the reference slice')
    parser.add_argument('--train_dataset', type=str, default=None, help='training dataset name for OOD testing (overrides --dataset in checkpoint path)')
    parser.add_argument('--has_confidence', action="store_true", help='whether to use confidence map in refinement module')
    parser.add_argument('--n_square', action="store_true", help='whether to average the metrics for n(n-1) pairs during inference')
    parser.add_argument('--deep_supervision', action='store_true', help='whether to use deep supervision')
    parser.add_argument('--strategy',type=str, default=None, choices=['average', 'weighted', 'closest','middle'], help='strategy when multiple reference')
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
    # dataset_config = {
    #     'Synapse': {
    #         'Dataset': Synapse_dataset,
    #         'volume_path': '../data/Synapse/test_vol_h5',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #         'z_spacing': 1,
    #     },
    # }
    dataset_config = {
        'local': {
            'root_path': './2D_data/local/test',
            'num_classes': 2,
        },
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
    # args.volume_path = dataset_config[dataset_name]['volume_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    if args.exp_name == 'sli2vol':
        args.Dataset = Dataset_sli2vol
    elif args.exp_name == 'vol2flow':
        args.Dataset = Dataset_vol2flow
    elif args.test_ref == 'largest':
        args.Dataset = Dataset_v5
        print('Using v5 dataset (largest as reference)')
    elif args.test_ref == 'middle':
        args.Dataset = Dataset_v5_2
        print('Using v5_2 dataset (middle as reference)')
    elif args.test_ref == 'neighbor':
        args.Dataset = Dataset_v6_5
        print('Using v6.5 dataset (neighbor as reference)')
    else:
        args.Dataset = Synapse_dataset
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    # args.z_spacing = dataset_config[dataset_name]['z_spacing']
    
    if args.n_square:
        if args.strategy is not None:
            args.Dataset = Dataset_v6_4
            print('Using dataset_v6_4')
        else:
            args.Dataset = Dataset_v6_2
    else:
        
        if args.strategy is not None:
            args.Dataset = Dataset_v6_3
            print('Using dataset_v6_3')

    print(args)
    args.is_pretrain = True
    # name the same snapshot defined in train script!
    # For OOD testing: --train_dataset overrides --dataset in checkpoint path
    ckpt_dataset = args.train_dataset if args.train_dataset else dataset_name
    if args.exp_name == 'TU':
        args.exp = args.exp_name + '_' + ckpt_dataset + str(args.img_size)
    else:
        args.exp = args.exp_name + '_' + args.train_ref + '_' + ckpt_dataset + str(args.img_size)
    # args.exp = args.exp_name + '_' + dataset_name + str(args.img_size)

    snapshot_path = "./checkpoints/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
    #     snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    # snapshot_path = snapshot_path + '_deep_supervision' if args.deep_supervision else snapshot_path
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    if args.exp_name == 'TU':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        print('Using original TransUNet!')
    # elif not args.has_confidence:
    #     net = My_ViT_seg2(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'sli2vol' or args.exp_name == 'vol2flow':
    #     net = My_ViT_seg_v3(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'sli2vol_v2' or args.exp_name == 'vol2flow_v2':
    #     net = My_ViT_seg_v4(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'sli2vol_v3o1' or args.exp_name == 'vol2flow_v3o1':
    #     net = My_ViT_seg_v5(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'sli2vol_v3o2' or args.exp_name == 'vol2flow_v3o2':
    #     net = My_ViT_seg_v6(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'sli2vol_v4' or args.exp_name == 'vol2flow_v4':
    #     net = My_ViT_seg_v7(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # elif args.exp_name == 'v5':
    #     net = My_ViT_seg_v8(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    #     print('Using My_ViT_seg_v8')
    # elif args.exp_name == 'v6':
    #     config_small = CONFIGS_ViT_seg['R18-ViT-S_16']
    #     config_small.n_classes = args.num_classes
    #     config_small.n_skip = args.n_skip
    #     config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    #     net = My_ViT_seg_v9(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    #     print('Using My_ViT_seg_v9')
    # elif args.exp_name == 'v6.1' or args.exp_name == 'v6.2' or args.exp_name == 'v6.3' or args.exp_name == 'v6.5':
    #     config_small = CONFIGS_ViT_seg['R18-ViT-S_16']
    #     config_small.n_classes = args.num_classes
    #     config_small.n_skip = args.n_skip
    #     config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    #     net = My_ViT_seg_v10(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    #     print('Using My_ViT_seg_v10')
    elif 'v6.5.' in args.exp_name:
        exp_num = args.exp_name.split('.')[-1]
        config_small = CONFIGS_ViT_seg['R18-ViT-S_16']
        config_small.n_classes = args.num_classes
        config_small.n_skip = args.n_skip
        config_small.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net_name = 'My_ViT_seg_v65' + exp_num
        net = eval(net_name)(config_vit, config_small, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        print('Using ' + net_name)
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
        # net.load_from("./tumor_seg/networks/swin_tiny_patch4_window7_224.pth")
    elif args.exp_name == 'swin_unet_ours':
        print('Using swin unet + ours model')
        net = SwinUnet_ours(SwinUnet_config(), img_size=args.img_size, num_classes=args.num_classes).cuda()
        # net.load_from("./tumor_seg/networks/swin_tiny_patch4_window7_224.pth")
    elif args.exp_name == 'FAT_Net':
        print('Using FAT_Net baseline model')
        net = FAT_Net(n_channels=1, n_classes=args.num_classes).cuda()
    elif args.exp_name == 'FAT_Net_ours':
        print('Using FAT_Net + ours model')
        net = FATNet_ours(in_chan=1, num_classes=args.num_classes).cuda()
    elif args.exp_name == 'MISSFormer':
        print('Using MISSFormer baseline model')
        net = MISSFormer(num_classes=args.num_classes).cuda()
    elif args.exp_name == 'H2Former':
        print('Using H2Former baseline model')
        net = res34_swin_MS(image_size=args.img_size, num_class=args.num_classes).cuda()
    elif args.exp_name == 'H2Former_ours':
        print('Using H2Former + ours model')
        net = H2Former_ours(img_size=args.img_size, num_classes=args.num_classes).cuda()
    else:
        raise Exception('Invalid experiment name, cannot find appropriate model')
    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # snapshot =snapshot.replace('149','39')
    net.load_state_dict(torch.load(snapshot))
    print('Have loaded snapshot in:',snapshot)
    #计算模型参数量
    #打印模型参数名字和参数数量
    num_params = sum(p.numel() for name, p in net.named_parameters()if 'refinement_module' in name )
    print('Number of parameters in refinement_module:', num_params/1e6, 'M')
    # for name, param in net.named_parameters():
    #     print(name, param.numel())
    
    num_params = sum(p.numel() for name, p in net.named_parameters()if 'refinement_module.decoder' in name )
    print('Number of parameters in refinement_module.decoder:', num_params/1e6, 'M')
    # raise Exception
    # num_params = sum(p.numel() for p in net.parameters() if 'fusion' not in p.name and 'refinement' not in p.name)
    # print('Number of parameters:', num_params)
    # raise Exception

    
    snapshot_name = snapshot_path.split('/')[-1]
    # For OOD: append test dataset to log/save paths to avoid overwriting
    ood_suffix = '_testOn_' + dataset_name if args.train_dataset else ''
    log_folder = './test_log/test_log_' + args.exp + ood_suffix
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    if args.train_dataset:
        logging.info('OOD testing: train_dataset=%s, test_dataset=%s' % (args.train_dataset, dataset_name))
    if args.is_save:
        # args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp + ood_suffix, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    if args.strategy is not None:
        print('Using multiple reference slices!')
        inference_multiple_ref(args, net, test_save_path)
    else:
        inference(args, net, test_save_path)


