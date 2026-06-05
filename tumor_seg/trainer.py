import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import DiceLoss
from torchvision import transforms

def trainer_tumor(args, model, snapshot_path):
    from .datasets.tumor_dataset import Dataset_baseline, RandomGenerator, RandomGenerator_ref
    from .datasets.tumor_dataset import Dataset_middle, Dataset_neighbor
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if args.ref == 'middle':
        DatasetCls = Dataset_middle
        GenCls = RandomGenerator_ref
        print('Using middle-slice reference dataset')
    elif args.ref == 'neighbor':
        DatasetCls = Dataset_neighbor
        GenCls = RandomGenerator_ref
        print('Using neighbor-slice reference dataset')
    else:
        DatasetCls = Dataset_baseline
        GenCls = RandomGenerator
        print('Using baseline (no-reference) dataset')
    # Train uses random augmentation; val mirrors the test pipeline (resize only,
    # no flips/rotations) so the best-checkpoint selection signal is deterministic.
    train_transform = transforms.Compose([GenCls(output_size=[args.img_size, args.img_size], augment=True)])
    val_transform = transforms.Compose([GenCls(output_size=[args.img_size, args.img_size], augment=False)])
    db_train = DatasetCls(base_dir=args.root_path, transform=train_transform, mode='train')
    # In this repo's 2D_data layout the split is encoded in the directory path
    # (.../<dataset>/{train,val,test}/...), and the Dataset classes build paths
    # straight from base_dir without inserting `mode`. So to read the held-out
    # val split we must point base_dir at the sibling val/ directory rather than
    # rely on mode='val' (which would otherwise re-read the train slices).
    val_base_dir = os.path.join(os.path.dirname(os.path.normpath(args.root_path)), 'val')
    try:
        db_val = DatasetCls(base_dir=val_base_dir, transform=val_transform, mode='val')
    except Exception as e:
        db_val = None
        logging.warning("val dataset unavailable at %s (%s); skipping val-based best-ckpt selection.", val_base_dir, e)
    print("The length of train set is: {}".format(len(db_train)))
    if db_val is not None:
        print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = (DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
                 if db_val is not None and len(db_val) > 0 else None)
    # raise Exception
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    best_val_loss = float('inf')

    def _forward_loss(sampled_batch):
        """Shared forward + (CE + Dice) loss path used by both training and validation."""
        image_batch = sampled_batch['image'].cuda()
        label_batch = sampled_batch['label'].cuda()
        if 'ours' in args.exp_name:
            ref_image_batch = sampled_batch['ref_image'].unsqueeze(1).cuda()
            ref_mask_batch = sampled_batch['ref_mask'].unsqueeze(1).cuda()
            outputs = model(image_batch, ref_image_batch, ref_mask_batch)
        else:
            outputs = model(image_batch)
        if args.deep_supervision:
            ce = 0
            dl = 0
            for logits_i in outputs['iters']:
                ce = ce + ce_loss(logits_i, label_batch[:].long())
                dl = dl + dice_loss(logits_i, label_batch, softmax=True)
        else:
            # `_ours` models return a dict {'final': ..., 'iters': [...]}; the rest return a tensor.
            logits_for_loss = outputs['final'] if isinstance(outputs, dict) else outputs
            ce = ce_loss(logits_for_loss, label_batch[:].long())
            dl = dice_loss(logits_for_loss, label_batch, softmax=True)
        return 0.5 * ce + 0.5 * dl, ce

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            loss, loss_ce = _forward_loss(sampled_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        # ---- end-of-epoch validation: track best-val checkpoint ----
        if valloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in valloader:
                    v_loss, _ = _forward_loss(val_batch)
                    val_losses.append(v_loss.item())
            val_mean = float(np.mean(val_losses)) if val_losses else float('inf')
            writer.add_scalar('val/total_loss', val_mean, epoch_num)
            logging.info('[epoch %d] val loss = %f' % (epoch_num, val_mean))
            if val_mean < best_val_loss:
                best_val_loss = val_mean
                best_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                logging.info('[epoch %d] new best val loss = %f -> saved %s' % (epoch_num, val_mean, best_path))

            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min())
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        # save_interval = 10  # int(max_epoch/6)
        # if  (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"