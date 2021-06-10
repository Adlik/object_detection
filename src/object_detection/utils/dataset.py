# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom dateset"""

import glob
import logging
import math
import os
import random
import argparse
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import xyxy2xywh, plot_images

# pylint: disable=W0212, W0104, C0103
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
logger = logging.getLogger(__name__)


# Get orientation exif tag
for orientation in ExifTags.TAGS:
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def _exif_size(img):
    """Returns exif-corrected PIL size

    :param img: original images
    :return: image size
    """

    img_size = img.size  # (width, height)
    rotation = dict(img._getexif().items())[orientation]
    if rotation == 6:  # rotation 270
        img_size = (img_size[1], img_size[0])
    elif rotation == 8:  # rotation 90
        img_size = (img_size[1], img_size[0])
    return img_size


def _get_hash(files):
    """Returns a single hash value of a list of files

    :param files: all files
    :return: the size of files
    """

    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def _img2label_paths(img_paths):
    """ Converts image path to label path.

    :param img_paths: '../data/coco/images/train2017/000000000009.jpg'
    :return:  '../data/coco/labels/train2017/000000000009.txt'
    """

    return [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in img_paths]


class LoadImagesAndLabels(Dataset):
    """Load images and labels"""

    def __init__(self, path, img_size=608, batch_size=16, augment=False, hyp=None, rect=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        try:
            img_path = []
            dirname = os.path.dirname(path) + os.sep
            if os.path.isdir(path):
                img_path += glob.glob(path + os.sep + '**' + os.sep + '*.*', recursive=True)
            elif os.path.isfile(path):
                with open(path, 'r') as f_path:
                    f_path = f_path.read().splitlines()
                    img_path += [x.replace('./', dirname) if x.startswith('./') else x for x in f_path]
            else:
                raise Exception(f'{path} does not exist!')
            self.img_files = sorted([x for x in img_path if x.split('.')[-1].lower() in img_formats])
            if len(self.img_files) == 0:
                raise AssertionError('No images found!')
        except Exception as exception:
            raise Exception(f'Error loading data from {path}: {exception}\n See {help_url}') from exception

        # Define labels and check cache
        self.label_files = _img2label_paths(self.img_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cache labels
        if cache_path.is_file():             # if cache labels exits
            cache = torch.load(cache_path)   # load
            if cache['hash'] != _get_hash(self.label_files + self.img_files) or 'results' not in cache:
                cache = self.__cache_labels(cache_path)
        else:
            cache = self.__cache_labels(cache_path)  # cache

        # Display cache
        [num_found, num_missing, num_empty, num_corrupted, num_total] = cache.pop('results')
        desc = f"Scanning '{cache_path}' for images and labels..." \
               f"{num_found} found, {num_missing} missing, {num_empty} empty, {num_corrupted} corrupted"
        tqdm(None, desc=desc, total=num_total, initial=num_total)
        if num_found == 0 and augment:
            raise AssertionError(f'No labels found in {cache_path}. Can not train without labels.')

        # Read cache
        cache.pop('hash')
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())                # update
        self.label_files = _img2label_paths(cache.keys())   # update
        if single_cls:
            for lab in self.labels:
                lab[:, 0] = 0

        num_img = len(self.img_files)
        batch_index = np.floor(np.arange(num_img) / batch_size).astype(np.int)
        num_batches = batch_index[-1] + 1
        self.batch = batch_index   # batch index of image
        self.num_img = num_img
        self.indices = range(num_img)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            img_shape = self.shapes   # wh
            aspect_ratio = img_shape[:, 1] / img_shape[:, 0]      # aspect ratio
            index_rect = aspect_ratio.argsort()   # index with aspect ratio in ascending order
            self.img_files = [self.img_files[i] for i in index_rect]
            self.label_files = [self.label_files[i] for i in index_rect]
            self.labels = [self.labels[i] for i in index_rect]
            self.shapes = img_shape[index_rect]           # wh
            aspect_ratio = aspect_ratio[index_rect]

            # Set training image shapes
            shapes = [[1, 1]] * num_batches
            for i in range(num_batches):
                ar_index = aspect_ratio[batch_index == i]
                min_index, max_index = ar_index.min(), ar_index.max()
                if max_index < 1:
                    shapes[i] = [max_index, 1]
                elif min_index > 1:
                    shapes[i] == [1, 1 / min_index]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * num_img
        if cache_images:
            gigabytes = 0   # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * num_img, [None] * num_img
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(num_img)))  # 8 threads
            pbar = tqdm(enumerate(results), total=num_img)
            for i, result in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = result
                gigabytes += self.imgs[i].nbytes
                pbar.desc = f'Caching images ({gigabytes / 1E9:.1f}GB)'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
            # MixUp  https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.num_img - 1))
                mixup_ratio = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * mixup_ratio + img2 * (1 - mixup_ratio)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            img, (orig_h, orig_w), (height, width) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = _letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (orig_h, orig_w), ((height / orig_h, width / orig_w), pad)   # for COCO mAP rescaling

            # Load labels
            labels = []
            label = self.labels[index]
            if label.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = label.copy()
                labels[:, 1] = ratio[0] * width * (label[:, 1] - label[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * height * (label[:, 2] - label[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * width * (label[:, 1] + label[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * height * (label[:, 2] + label[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        num_labels = len(labels)
        if num_labels:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]           # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]           # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if num_labels:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((num_labels, 6))
        if num_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        """collate functions"""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, lab in enumerate(label):
            lab[:, 0] = i                         # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    def __cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes

        :param self: self
        :param path: the cache file path
        :return: img_label_dict, keys = [img_file, 'hash', 'results'],
        {img_file_1: [labels_1, shape_1], img_file_2: [labels_2, shapes_2]}
        """
        img_label_dict = {}
        num_missing, num_found, num_empty, num_corrupted, num_total = 0, 0, 0, 0, 0
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (img_file, label_file) in enumerate(pbar):
            try:
                # verify images
                img = Image.open(img_file)
                img.verify()                  # PIL verify
                shape = _exif_size(img)        # image size
                if (shape[0] < 10) and (shape[1] < 10):
                    raise AssertionError('image size < 10 pixels')

                # verify labels
                if os.path.isfile(label_file):
                    num_found += 1
                    with open(label_file, 'r') as f_label:
                        labels = np.array([x.split() for x in f_label.read().strip().splitlines()], dtype=np.float32)
                    if len(labels):
                        if labels.shape[1] != 5:
                            raise AssertionError('labels requirs 5 columns each')
                        if (labels < 0).any():
                            raise AssertionError('labels cannot be negative!')
                        if (labels[:, 1:] > 1).any():
                            raise AssertionError('non-normalized or out of bounds coordinate labels')
                        if np.unique(labels, axis=0).shape[0] != labels.shape[0]:
                            raise AssertionError('duplicate labels')
                    else:
                        num_empty += 1
                        labels = np.zeros((0, 5), dtype=np.float32)
                else:
                    num_missing += 1
                    labels = np.zeros((0, 5), dtype=np.float32)
                img_label_dict[img_file] = [labels, shape]
            except IndexError:
                num_corrupted += 1
                print(f'WARNING: Ignoring corrupted image and/or label {img_file}')

            num_total = i

            pbar.desc = f"Scanning '{path.parent / path.stem}' for images and labels..." \
                        f"{num_found} found, {num_missing} missing, {num_empty} empty, {num_corrupted} corrupted"

        if num_found == 0:
            print(f'WARNING: No labels found in {path}. See {help_url}')

        img_label_dict['hash'] = _get_hash(self.label_files + self.img_files)
        img_label_dict['results'] = [num_found, num_missing, num_empty, num_corrupted, num_total + 1]
        torch.save(img_label_dict, path)     # save for next time
        logging.info("New cache created: %s", path)
        return img_label_dict


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    """Loads 1 image from dataset, returns img, original hw, resized hw

    :param self: self
    :param index: index
    :return: img, original hw, resized hw
    """

    img = self.imgs[index]
    if img is None:  # not cached images
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        if img is None:
            raise AssertionError(f'Image Not Found {path}')
        orig_h, orig_w = img.shape[:2]
        size_ratio = self.img_size / max(orig_h, orig_w)  # resize image to img_size
        if size_ratio != 1:   # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if size_ratio < 1 and self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(orig_w * size_ratio), int(orig_h * size_ratio)), interpolation=interp)
        return img, (orig_h, orig_w), img.shape[:2]
    return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """Augment colorspace

    :param img: images
    :param hgain: Hue gain
    :param sgain: Saturation gain
    :param vgain: Value gain
    :return: None
    """

    random_gains = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    pixel_value = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((pixel_value * random_gains[0]) % 180).astype(dtype)
    lut_sat = np.clip(pixel_value * random_gains[1], 0, 255).astype(dtype)
    lut_val = np.clip(pixel_value * random_gains[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    """Loads images in a mosaic

    :param self: self
    :param index: index
    :return: img4, labels4
    """

    labels4 = []
    img_size = self.img_size
    y_center, x_center = [int(random.uniform(-x, 2 * img_size + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [self.indices[random.randint(0, self.num_img - 1)] for _ in range(3)]  # 3 additional img indice
    for i, idx in enumerate(indices):
        img, _, (height, width) = load_image(self, idx)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((img_size * 2, img_size * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(x_center - width, 0), max(y_center - height, 0), x_center, y_center
            x1b, y1b, x2b, y2b = width - (x2a - x1a), height - (y2a - y1a), width, height  # xmin, ymin, xmax, ymax
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = x_center, max(y_center - height, 0), min(x_center + width, img_size * 2), y_center
            x1b, y1b, x2b, y2b = 0, height - (y2a - y1a), min(width, x2a - x1a), height
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(x_center - width, 0), y_center, x_center, min(img_size * 2, y_center + height)
            x1b, y1b, x2b, y2b = width - (x2a - x1a), 0, width, min(y2a - y1a, height)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = x_center, y_center, min(x_center + width, img_size * 2), \
                                 min(img_size * 2, y_center + height)
            x1b, y1b, x2b, y2b = 0, 0, min(width, x2a - x1a), min(y2a - y1a, height)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        label = self.labels[idx]
        label4 = label.copy()
        if label.size > 0:  # Normalized xywh to pixel xyxy format
            label4[:, 1] = width * (label[:, 1] - label[:, 3] / 2) + padw
            label4[:, 2] = height * (label[:, 2] - label[:, 4] / 2) + padh
            label4[:, 3] = width * (label[:, 1] + label[:, 3] / 2) + padw
            label4[:, 4] = height * (label[:, 2] + label[:, 4] / 2) + padh
        labels4.append(label4)

    # Concat/clip labels
    if labels4:
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * img_size, out=labels4[:, 1:])  # use with random_perspective

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """random perspective
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    :param img: image
    :param targets: [cls, xyxy]
    :param degrees: range of degrees to select from
    :param translate: tuple of maximum absolute fraction for horizontal and vertical translations
    :param shear: range of degrees to select from
    :param perspective: perspective transformation
    :param border: border
    :return: img, targets
    """

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    center = np.eye(3)
    center[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspectives = np.eye(3)
    perspectives[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    perspectives[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    rotation = np.eye(3)
    r_angle = random.uniform(-degrees, degrees)
    r_scale = random.uniform(1 - scale, 1 + scale)
    rotation[:2] = cv2.getRotationMatrix2D(angle=r_angle, center=(0, 0), scale=r_scale)

    # Shear
    shears = np.eye(3)
    shears[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    shears[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = np.eye(3)
    translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    matrix = translation @ shears @ rotation @ perspectives @ center  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, matrix, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, matrix[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    num_targets = len(targets)
    if num_targets:
        # warp points
        box_xy = np.ones((num_targets * 4, 3))
        box_xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(num_targets * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        box_xy = box_xy @ matrix.T  # transform
        if perspective:
            box_xy = (box_xy[:, :2] / box_xy[:, 2:3]).reshape(num_targets, 8)  # rescale
        else:  # affine
            box_xy = box_xy[:, :2].reshape(num_targets, 8)

        # create new boxes
        box_x = box_xy[:, [0, 2, 4, 6]]
        box_y = box_xy[:, [1, 3, 5, 7]]
        box_xy = np.concatenate((box_x.min(1), box_y.min(1), box_x.max(1), box_y.max(1))).reshape(4, num_targets).T

        # clip boxes
        box_xy[:, [0, 2]] = box_xy[:, [0, 2]].clip(0, width)
        box_xy[:, [1, 3]] = box_xy[:, [1, 3]].clip(0, height)

        # filter candidates
        filter_candidate = _box_candidates(box1=targets[:, 1:5].T * r_scale, box2=box_xy.T)
        targets = targets[filter_candidate]
        targets[:, 1:5] = box_xy[filter_candidate]

    return img, targets


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    """
    Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    """

    box1_w, box1_h = box1[2] - box1[0], box1[3] - box1[1]
    box2_w, box2_h = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(box2_w / (box2_h + 1e-16), box2_h / (box2_w + 1e-16))
    return (box2_w > wh_thr) & (box2_h > wh_thr) & (box2_w * box2_h / (box1_w * box1_h + 1e-16) > area_thr) & \
           (aspect_ratio < ar_thr)


def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scalefill=False, scaleup=True):
    """
    Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    """

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        scale_ratio = min(scale_ratio, 1.0)

    # Compute padding
    ratio = scale_ratio, scale_ratio  # width, height ratios
    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))
    padding_w, padding_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)  # wh padding
    elif scalefill:  # stretch
        padding_w, padding_h = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    padding_w /= 2  # divide padding into 2 sides
    padding_h /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(padding_h - 0.1)), int(round(padding_h + 0.1))
    left, right = int(round(padding_w - 0.1)), int(round(padding_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (padding_w, padding_h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/coco2017.yaml', help='data.yaml path')
    parser.add_argument('--img_size', type=int, default=608, help='set the images size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--hyp', type=str, default='../data/hyp_yolov5.yaml', help='hyperparameters path')
    opt = parser.parse_args()

    with open(opt.hyp) as file:
        hyper = yaml.safe_load(file)
    with open(opt.data) as file:
        data_dict = yaml.safe_load(file)
    train_path = data_dict['train']
    dataset = LoadImagesAndLabels(train_path, opt.img_size, opt.batch_size,
                                  augment=True,       # augment images
                                  hyp=hyper,            # augmentation hyperparameters
                                  rect=False,         # rectangular training
                                  cache_images=False,
                                  single_cls=False,
                                  stride=32,
                                  pad=0)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             num_workers=0,
                                             shuffle=True,
                                             sampler=None,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    for imgs, labs, paths, _ in dataloader:
        # Plots training images overlaid with targets
        file_name = "../data/train_images.jpg"
        plot_images(imgs, labs, paths, file_name)
        break
