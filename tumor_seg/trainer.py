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
        db_train = Dataset_middle(base_dir=args.root_path,
                                transform=transforms.Compose(
                                    [RandomGenerator_ref(output_size=[args.img_size, args.img_size])]), mode='train')
        print('Using middle-slice reference dataset')
    elif args.ref == 'neighbor':
        db_train = Dataset_neighbor(base_dir=args.root_path,
                                transform=transforms.Compose(
                                    [RandomGenerator_ref(output_size=[args.img_size, args.img_size])]), mode='train')
        print('Using neighbor-slice reference dataset')
    else:
        db_train = Dataset_baseline(base_dir=args.root_path,
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size])]), mode='train')
        print('Using baseline (no-reference) dataset')
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
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
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, ref_image_batch, ref_mask_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['ref_image'], sampled_batch['ref_mask']
            image_batch, label_batch, ref_image_batch, ref_mask_batch = image_batch.cuda(), label_batch.cuda(), ref_image_batch.unsqueeze(1).cuda(), ref_mask_batch.unsqueeze(1).cuda()
            # print('image_batch:', image_batch.shape, image_batch.max(), image_batch.min())
            # print('label_batch:', label_batch.shape, label_batch.max(), label_batch.min())
            # print('ref_image_batch:', ref_image_batch.shape, ref_image_batch.max(), ref_image_batch.min())
            # print('ref_mask_batch:', ref_mask_batch.shape, ref_mask_batch.max(), ref_mask_batch.min())
            #img: bs,1,224,224  label: bs,224,224, ref_img: bs,1,224,224, ref_mask: bs,1,224,224
            #print shape,max,min
            # print('image:', image_batch.shape, image_batch.max(), image_batch.min())
            # print('label:', label_batch.shape, label_batch.max(), label_batch.min())
            # raise Exception
            
            if 'ours' in args.exp_name:
                outputs = model(image_batch, ref_image_batch, ref_mask_batch)
            else:
                outputs = model(image_batch)
            # print('outputs:', outputs.shape, outputs.max(), outputs.min())
            # raise Exception
            
            if args.deep_supervision:
                loss_ce = 0
                loss_dice = 0
                for i, logits_i in enumerate(outputs['iters']):
                    loss_ce += ce_loss(logits_i, label_batch[:].long())
                    loss_dice += dice_loss(logits_i, label_batch, softmax=True)
                # loss_ce /= len(outputs['iters'])
                # loss_dice /= len(outputs['iters'])
            else:
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
            
            loss = 0.5 * loss_ce + 0.5 * loss_dice 
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