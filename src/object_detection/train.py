# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0103, R0914, W1203, E0401, E0611, W0601

""" Train"""

import os
import time
import math
import random
import logging
import argparse
from test import test
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.models import Model, load_darknet_weights
from utils.torch_utils import intersect_dicts, select_device
from utils.dataset import LoadImagesAndLabels
from utils.utils import init_seeds, plot_results, adjust_learning_rate
from utils.loss import compute_loss
from utils.prune_utils import parse_module_index, gather_bn_weights, bn_l1_regularization, distillation_loss


logger = logging.getLogger(__name__)

mixed_precision = True
try:
    from apex import amp
except ImportWarning:
    print("Not install apex!")
    mixed_precision = False

# Directories of the save weights
weights_dir = 'weights' + os.sep
last = weights_dir + 'last.pt'
best = weights_dir + 'best.pt'
results_file = 'results.txt'


def parse():
    """Parser for command-line options, arguments and sub-commands"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='500200 batches at bs 16, 117263 images = 273 epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='effective bs = batch_size * ccumulate = 16*4 = 64')
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--teacher_cfg', type=str, default='', help='teacher model cfg file for knowledge distillation')
    parser.add_argument('--data', type=str, default='data/coco2017.yaml', help='*.data file path')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='the file of hyp path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights')
    parser.add_argument('--teacher_weights', type=str, default='', help='teacher model weights')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture, defaultpw, uCE, uBCE')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--sparsity_training', '-st', dest='st', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--penalty_factor', '-pf', type=float, default=0.0001, help='scale sparse rate')
    parser.add_argument('--prune', type=int, default=0, help='0:nomal prune 1:other prune ')
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")
    args = parser.parse_args()

    return args


def main():
    """ Train and test

    :param opt: args
    :param writer: tensorboard
    :return:
    """

    global opt
    opt = parse()

    arc = opt.arc
    cfg = opt.cfg
    teacher_cfg = opt.teacher_cfg
    img_size = opt.img_size
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights
    teacher_weights = opt.teacher_weights
    multi_scale = opt.multi_scale
    sparsity_training = opt.st

    opt.weights = last if opt.resume else opt.weights

    # Initial logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)

    # Train
    logger.info(opt)
    if opt.local_rank in [-1, 0]:
        logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        writer = SummaryWriter()

    # Hyperparameters
    with open(opt.hyp) as f_hyp:
        hyp = yaml.safe_load(f_hyp)
    # data dict
    with open(opt.data) as f_data:
        data = yaml.safe_load(f_data)

    # Distributed training initialize
    device = select_device(opt.device)
    if opt.local_rank != -1:
        dist.init_process_group(init_method="env://", backend='nccl')
        torch.cuda.set_device(opt.local_rank)
        device = torch.device(f"cuda:{opt.local_rank}")
        # world_size = torch.distributed.get_world_size()

    init_seeds()
    cuda = device.type != 'cpu'
    torch.backends.cudnn.benchmark = True

    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5) + 1
        img_size_max = round(img_size / 32 * 1.5) - 1
        img_size = img_size_max * 32  # initiate with maximum multi_scale size
        logger.info(f'Using multi-scale  {img_size_min * 32} - {img_size}')

    train_path = data['train']
    num_classes = int(data['num_classes'])  # number of classes

    # Load dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,
                                  rect=opt.rect)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if opt.local_rank != -1 else None
    num_worker = os.cpu_count() // torch.cuda.device_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([num_worker, batch_size, 8]),
                                             shuffle=not (opt.rect or train_sampler),
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Load model
    model = Model(cfg, img_size, arc=arc).to(device)

    # Load teacher model
    if teacher_cfg:
        teacher_model = Model(teacher_cfg, img_size, arc).to(device)

    # optimizer parameter groups
    param_group0, param_group1 = [], []
    for key, value in model.named_parameters():
        if 'Conv2d.weight' in key:
            param_group1.append(value)
        else:
            param_group0.append(value)
    if opt.adam:
        optimizer = optim.Adam(param_group0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(param_group0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # add param_group1 with weight_decay
    optimizer.add_param_group({'params': param_group1, 'weight_decay': hyp['weight_decay']})
    logger.info(f'Optimizer groups: {len(param_group1)} conv.weight, {len(param_group0)} other')
    del param_group0, param_group1

    start_epoch = 0
    best_fitness = 0.
    if weights.endswith('.pt'):
        checkpoint = torch.load(weights, map_location=device)
        state_dict = intersect_dicts(checkpoint['model'], model.state_dict())
        model.load_state_dict(state_dict, strict=False)
        print('loaded weights from', weights, '\n')

        # load optimizer
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_fitness = checkpoint['best_fitness']
        # load results
        if checkpoint.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(checkpoint['training_results'])
        # resume
        if opt.resume:
            start_epoch = checkpoint['epoch'] + 1
        del checkpoint

    elif len(weights) > 0:
        # weights are 'yolov4.weights', 'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)
        logger.info(f'loaded weights from {weights}\n')

    # Load teacher weights
    if teacher_cfg:
        if teacher_weights.endswith('.pt'):
            teacher_model.load_state_dict(torch.load(teacher_weights, map_location=device)['model'])
        elif teacher_weights.endswith('.weights'):
            load_darknet_weights(teacher_model, teacher_weights)
        else:
            raise Exception('pls provide proper teacher weights for knowledge distillation')
        if not mixed_precision:
            teacher_model.eval()
        logger.info('<......................using knowledge distillation....................>')
        logger.info(f'teacher model: {teacher_weights}\n')

    # Sparsity training
    if opt.prune == 0:
        _, _, prune_index = parse_module_index(model.module_dicts)
        if sparsity_training:
            logger.info('normal sparse training')

    if mixed_precision:
        if teacher_cfg:
            [model, teacher_model], optimizer = amp.initialize([model, teacher_model], optimizer,
                                                               opt_level='O1', verbosity=1)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=1)

    # SyncBatchNorm and distributed training
    if cuda and opt.local_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank])
        model.module_list = model.module.module_list
        model.yolo_layers = model.module.yolo_layers

    for index in prune_index:
        bn_weights = gather_bn_weights(model.module_list, [index])
        if opt.local_rank == 0:
            writer.add_histogram('before_train_per_layer_bn_weights/hist', bn_weights.numpy(), index, bins='doane')

    # Start training
    model.num_classes = num_classes
    model.arc = opt.arc
    model.hyp = hyp
    num_batch_size = len(dataloader)
    # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    results = (0, 0, 0, 0, 0, 0, 0)
    start_train_time = time.time()
    logger.info('Image sizes %d \n Starting training for %d epochs...', img_size, epochs)

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mean_losses = torch.zeros(4).to(device)
        mean_soft_target = torch.zeros(1).to(device)
        pbar = enumerate(dataloader)
        logger.info(('\n %10s %10s %10s %10s %10s %10s %10s %10s'), 'Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total',
                    'targets', 'img_size')
        if opt.local_rank in [-1, 0]:
            pbar = tqdm(pbar, total=num_batch_size)
        optimizer.zero_grad()

        for i, (imgs, targets, _, _) in pbar:  # batch -------------------------------------------------------------
            num_integrated_batches = i + num_batch_size * epoch

            # Adjust the learning rate
            learning_rate = adjust_learning_rate(optimizer, num_integrated_batches, num_batch_size, hyp, epoch, epochs)
            if i == 0 and opt.local_rank in [-1, 0]:
                logger.info(f'learning rate: {learning_rate}')
            imgs = imgs.to(device) / 255.0
            targets = targets.to(device)

            # Multi-Scale training
            if multi_scale:
                if num_integrated_batches / accumulate % 10 == 0:
                    img_size = random.randrange(img_size_min, img_size_max + 1) * 32
                scale_factor = img_size / max(imgs.shape[2:])
                if scale_factor != 1:
                    new_shape = [math.ceil(x * scale_factor / 32.) * 32 for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)

            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)

            # knowledge distillation
            soft_target = 0
            if teacher_cfg:
                if mixed_precision:
                    with torch.no_grad():
                        output_teacher = teacher_model(imgs)
                else:
                    _, output_teacher = teacher_model(imgs)
                soft_target = distillation_loss(pred, output_teacher, model.num_classes, imgs.size(0))
                loss += soft_target

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Sparse the BN layer that needs pruning
            if sparsity_training:
                # bn_l1_regularization(model.module_list, opt.penalty_factor, cba_index, epoch, epochs)
                bn_l1_regularization(model.module_list, opt.penalty_factor, prune_index, epoch, epochs)

            # Accumulate gradient for x batches before optimizing
            if num_integrated_batches % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            if opt.local_rank in [-1, 0]:
                mean_losses = (mean_losses * i + loss_items) / (i + 1)
                mean_soft_target = (mean_soft_target * i + soft_target) / (i + 1)
                memory = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                description = ('%10s' * 2 + '%10.3g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), '%.3gG' % memory, *mean_losses, mean_soft_target, img_size)
                pbar.set_description(description)

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        # scheduler.step()

        if opt.local_rank in [-1, 0]:
            final_epoch = epoch + 1 == epochs
            # Calculate mAP
            if not (opt.notest or opt.nosave) or final_epoch:
                with torch.no_grad():
                    results, _ = test(cfg, data,
                                      batch_size=batch_size,
                                      img_size=opt.img_size,
                                      model=model,
                                      conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                      save_json=final_epoch and epoch > 0)

            # Write epoch results
            with open(results_file, 'a') as file:
                # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                file.write(description + '%10.3g' * 7 % results + '\n')

            # Write Tensorboard results
            if writer:
                outputs = list(mean_losses) + list(results)
                titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                          'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
                for output, title in zip(outputs, titles):
                    writer.add_scalar(title, output, epoch)
                bn_weights = gather_bn_weights(model.module_list, prune_index)
                writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')

            # Update best mAP
            fitness = results[2]
            if fitness > best_fitness:
                best_fitness = fitness

            # Save training results
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save and opt.local_rank == 0:
                with open(results_file, 'r') as file:
                    # Create checkpoint
                    checkpoint = {'epoch': epoch,
                                  'best_fitness': best_fitness,
                                  'training_results': file.read(),
                                  'model': model.module.state_dict() if isinstance(
                                   model, nn.parallel.DistributedDataParallel) else model.state_dict(),
                                  'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last checkpoint
                torch.save(checkpoint, last)

                # Save best checkpoint
                if best_fitness == fitness:
                    torch.save(checkpoint, best)

                # Delete checkpoint
                del checkpoint

            # end epoch -----------------------------------------------------------------------------------------------
    # end training

    if opt.local_rank in [-1, 0]:
        if len(opt.name):
            os.rename('results.txt', 'results_%s.txt' % opt.name)
        plot_results()  # save as results.png
        print(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - start_train_time) / 3600:.3f} hours.\n')
    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':

    main()
