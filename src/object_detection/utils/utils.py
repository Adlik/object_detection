# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""utils"""

import os
import math
import glob
import random
from pathlib import Path
import matplotlib.pyplot as plt
from cv2 import cv2
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loss import bbox_iou


def init_seeds(seed=0):
    """init seeds"""
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    """Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html"""

    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def coco80_to_coco91_class():
    """converts 80-index (val2014) to 91-index (paper)
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    :return: x
    """

    coco91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
              34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
              61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return coco91


def xyxy2xywh(box):
    """ Convert n x 4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    where x1y1 = top-left, x2y2 = bottom-right

    :param box: boxes=[x1, y1, x2, y2], boxes.shape=[n, 4]
    :return: boxes=[x, y, w, h]
    """

    box_xywh = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
    box_xywh[:, 0] = (box[:, 0] + box[:, 2]) / 2   # x center
    box_xywh[:, 1] = (box[:, 1] + box[:, 3]) / 2   # y center
    box_xywh[:, 2] = box[:, 2] - box[:, 0]         # width
    box_xywh[:, 3] = box[:, 3] - box[:, 1]         # height
    return box_xywh


def xywh2xyxy(box):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right

    :param box: boxes=[x, y, w, h], boxes.shape=[n, 4]
    :return: boxes=[x1, y1, x2, y2]
    """

    box_xyxy = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
    box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2   # top left x
    box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2   # top left y
    box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2   # bottom right x
    box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2   # bottom right y
    return box_xyxy


def color_list():
    """
    https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    :return: first 10 plt colors as (r,g,b)
    """

    def hex2rgb(hexs):
        return tuple(int(hexs[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def plot_images_mosaic(images, targets, paths=None, fname='train_images.jpg', names=None, max_size=608):
    """Plot image grid with labels

    :param images: images
    :param targets: targets
    :param paths: paths
    :param fname: fname
    :param names: names
    :param max_size: max size
    :return: mosaic
    """

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    thickness_line = 3  # line thickness
    thickness_font = max(thickness_line - 1, 1)  # font thickness
    max_subplots = 16
    batch_size, _, height, width = images.shape  # batch size, _, height, width
    batch_size = min(batch_size, max_subplots)  # limit plot images
    num_subplots = np.ceil(batch_size ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(height, width)
    if scale_factor < 1:
        height = math.ceil(scale_factor * height)
        width = math.ceil(scale_factor * width)

    colors = color_list()  # list of colors
    mosaic = np.full((int(num_subplots * height), int(num_subplots * width), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(width * (i // num_subplots))
        block_y = int(height * (i % num_subplots))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (width, height))

        mosaic[block_y:block_y + height, block_x:block_x + width, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= width  # scale to pixels
                    boxes[[1, 3]] *= height
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=thickness_line)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=thickness_line / 3, thickness=thickness_font)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, thickness_line / 3, [220, 220, 220],
                        thickness=thickness_font, lineType=cv2.LINE_AA)
        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + width, block_y + height), (255, 255, 255), thickness=3)

    if fname:
        ratio = min(1280. / max(height, width) / num_subplots, 1.0)  # ratio to limit image size
        resize_w = int(num_subplots * width * ratio)
        resize_h = int(num_subplots * height * ratio)
        mosaic = cv2.resize(mosaic, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image img

    :param box: [x1, y1, x2, y2]
    :param img: image
    :param color: True or Float
    :param label: label
    :param line_thickness: line_thickness
    :return: image with bounding box
    """

    line_thick = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, pt1, pt2, color, thickness=line_thick, lineType=cv2.LINE_AA)
    if label:
        font_thick = max(line_thick - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thick / 3, thickness=font_thick)[0]
        pt2 = pt1[0] + t_size[0], pt1[1] - t_size[1] - 3
        cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (pt1[0], pt1[1] - 2), 0, line_thick / 3, [225, 255, 255],
                    hickness=font_thick, lineType=cv2.LINE_AA)


def plot_results(start=0, stop=0, path='data/results'):
    """Plot training results files 'results*.txt'

    :param start: start
    :param stop: stop
    :param path: the path to save the results
    :return: results.png
    """

    fig, axes = plt.subplots(2, 5, figsize=(14, 7))
    axes = axes.ravel()
    desc = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
            'val GIoU', 'val Objectness', 'val Classification', 'mAP', 'F1']

    # result_file = path + os.sep + 'results.txt'
    # for f in sorted(glob.glob(result_file)):
    for result_file in sorted(glob.glob('results*.txt')):
        results = np.loadtxt(result_file, usecols=[2, 3, 4, 7, 8, 11, 12, 13, 9, 10], ndmin=2).T
        num_rows = results.shape[1]
        x_axis = range(start, min(stop, num_rows) if stop else num_rows)
        for i in range(10):
            y_axis = results[i, x_axis]
            if i in [0, 1, 2, 5, 6, 7]:
                y_axis[y_axis == 0] = np.nan  # dont show zero loss values
            axes[i].plot(x_axis, y_axis, marker='.', label=result_file.replace('.txt', ''))
            axes[i].set_title(desc[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                axes[i].get_shared_y_axes().join(axes[i], axes[i - 5])

    fig.tight_layout()
    axes[1].legend()
    fig.savefig(os.path.join(path, 'results.png'), dpi=200)


def adjust_learning_rate(optimizer, iteration, epoch_size, hyp, epoch, epochs):
    """adjust learning rate, warmup and lr decay

    :param optimizer: optimizer
    :param gamma: gamma
    :param iteration: iteration
    :param epoch_size: epoch_size
    :param hyp: hyperparameters
    :param epoch: epoch
    :param epochs: the number of epochs
    :return: lr
    """

    step_index = 0
    if epoch < 6:
        # The first 6 epochs carried out warm up
        learning_rate = 1e-6 + (hyp['lr0'] - 1e-6) * iteration / (epoch_size * 2)
    else:
        if epoch > epochs * 0.5:
            # At 50% of the epochs, the learning rate decays in Gamma
            step_index = 1
        if epoch > epochs * 0.7:
            # At 70% of the epochs, the learning rate decays in Gamma^2
            step_index = 2
        learning_rate = hyp['lr0'] * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)

    :param prediction: prediction
    :param conf_thres: confidence threshold
    :param nms_thres: nms threshold
    :return: (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for cls in pred[:, -1].unique():
            det_cls = pred[pred[:, -1] == cls]  # select class c
            num_det_cls = len(det_cls)
            if num_det_cls == 1:
                det_max.append(det_cls)  # No NMS required if only 1 prediction
                continue
            if num_det_cls > 100:
                det_cls = det_cls[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                while det_cls.shape[0]:
                    det_max.append(det_cls[:1])  # save highest conf detection
                    if len(det_cls) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(det_cls[0], det_cls[1:])  # iou with other boxes
                    det_cls = det_cls[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(det_cls) > 1:
                    iou = bbox_iou(det_cls[0], det_cls[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(det_cls[:1])
                    det_cls = det_cls[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(det_cls):
                    if len(det_cls) == 1:
                        det_max.append(det_cls)
                        break
                    i = bbox_iou(det_cls[0], det_cls) > nms_thres  # iou with other boxes
                    weights = det_cls[i, 4:5]
                    det_cls[0, :4] = (weights * det_cls[i, :4]).sum(0) / weights.sum()
                    det_max.append(det_cls[:1])
                    det_cls = det_cls[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(det_cls):
                    if len(det_cls) == 1:
                        det_max.append(det_cls)
                        break
                    det_max.append(det_cls[:1])
                    iou = bbox_iou(det_cls[0], det_cls[1:])  # iou with other boxes
                    det_cls = det_cls[1:]
                    det_cls[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if det_max:
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def scale_coords(img1_shape, coords, img0_shape):
    """Rescale coords (xyxy) from img1_shape to img0_shape

    :param img1_shape: img1 shape
    :param coords: (xyxy)
    :param img0_shape: img0 shape
    :return: rescaled coords
    """

    gain = max(img1_shape) / max(img0_shape)
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)

    :param boxes: boxes
    :param img_shape: img shape
    :return: cliped boxes
    """

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    """Plots training images overlaid with targets

    :param imgs: images
    :param targets: targets
    :param paths: paths
    :param fname: file name
    :return: images.jpg
    """

    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    batch_size, _, height, width = imgs.shape  # batch size, _, height, width
    batch_size = min(batch_size, 16)  # limit plot to 16 images
    num_subplot = np.ceil(batch_size ** 0.5).astype(np.int32)

    for i in range(batch_size):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= width
        boxes[[1, 3]] *= height
        plt.clf()
        plt.subplot(num_subplot, num_subplot, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            save_path = Path(paths[i]).name
            plt.title(save_path[:min(len(save_path), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def save_weights(model, path='model.weights', cutoff=-1):
    """Converts a PyTorch model to Darket format (*.pt to *.weights)
    Note: Does not work if model.fuse() is applied

    :param model: model
    :param path: the path to save *.weights
    :param cutoff: cutoff
    :return: *.weights
    """

    with open(path, 'wb') as file:
        model.version.tofile(file)
        model.seen.tofile(file)

        # Iterate through layers
        for _, (module_def, module) in enumerate(zip(model.module_dicts[:cutoff], model.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(file)
                    bn_layer.weight.data.cpu().numpy().tofile(file)
                    bn_layer.running_mean.data.cpu().numpy().tofile(file)
                    bn_layer.running_var.data.cpu().numpy().tofile(file)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(file)
                conv_layer.weight.data.cpu().numpy().tofile(file)
