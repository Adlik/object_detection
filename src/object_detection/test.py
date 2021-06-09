# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E0401, R0801

"""Test"""

import os
import argparse
import json
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.models import Model, load_darknet_weights
from utils.dataset import LoadImagesAndLabels
from utils.utils import coco80_to_coco91_class, non_max_suppression, scale_coords, clip_coords,\
    xyxy2xywh, xywh2xyxy, bbox_iou, plot_images
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, intersect_dicts


# pylint: disable=R0913, R0914, R0915
def test(cfg, data, weights=None, batch_size=16, img_size=608, iou_thres=0.5, conf_thres=0.001, nms_thres=0.5,
         save_json=True, hyp=None, model=None, single_cls=False):
    """test the metrics of the trained model

    :param str cfg: model cfg file
    :param str data: data dict
    :param str weights: weights path
    :param int batch_size: batch size
    :param int img_size: image size
    :param float iou_thres: iou threshold
    :param float conf_thres: confidence threshold
    :param float nms_thres: nms threshold
    :param bool save_json: Whether to save the model
    :param str hyp: hyperparameter
    :param str model: yolov4 model
    :param bool single_cls: only one class
    :return: results
    """

    if model is None:
        device = select_device(opt.device)
        verbose = False
        # Initialize model
        model = Model(cfg, img_size).to(device)
        # Load weights
        if weights.endswith('.pt'):
            checkpoint = torch.load(weights, map_location=device)
            state_dict = intersect_dicts(checkpoint['model'], model.state_dict())
            model.load_state_dict(state_dict, strict=False)
        elif len(weights) > 0:
            load_darknet_weights(model, weights)
        print(f'Loaded weights from {weights}!')

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device
        verbose = False

    test_path = data['valid']
    num_classes, names = (1, ['item']) if single_cls else (int(data['num_classes']), data['names'])

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size, hyp=hyp)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    output_format = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'Pre', 'Rec', 'mAP', 'F1')
    precision, recall, f_1, mean_pre, mean_rec, mean_ap, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    json_dict, stats, aver_pre, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=output_format)):
        targets = targets.to(device)
        imgs = imgs.to(device) / 255.0
        _, _, height, width = imgs.shape    # batch size, channels, height, width

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        with torch.no_grad():
            inference_output, train_output = model(imgs)

            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_output, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            output = non_max_suppression(inference_output, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for i, pred in enumerate(output):
            labels = targets[targets[:, 0] == i, 1:]
            num_labels = len(labels)
            target_class = labels[:, 0].tolist() if num_labels else []
            seen += 1

            if pred is None:
                if num_labels:
                    stats.append(([], torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[i]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[i].shape[1:], box, shapes[i][0])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for det_i, det in enumerate(pred):
                    json_dict.append({'image_id': image_id,
                                      'category_id': coco91class[int(det[6])],
                                      'bbox': [float(format(x, '.%gf' % 3)) for x in box[det_i]],
                                      'score': float(format(det[4], '.%gf' % 5))})

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if num_labels:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for j, (*pbox, _, _, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == num_labels:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in target_class:
                        continue

                    # Best iou, index between pred and targets
                    mask = (pcls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    iou, best_iou = bbox_iou(pbox, tbox[mask]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and mask[best_iou] not in detected:  # and pcls == target_class[bi]:
                        correct[j] = 1
                        detected.append(mask[best_iou])

            # Append statistics (correct, conf, pcls, target_class)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), target_class))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
    if len(stats):
        precision, recall, aver_pre, f_1, ap_class = ap_per_class(*stats)
        mean_pre, mean_rec, mean_ap, mf1 = precision.mean(), recall.mean(), aver_pre.mean(), f_1.mean()
        num_targets = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        num_targets = torch.zeros(1)

    # Print results
    print_format = '%20s' + '%10.3g' * 6
    print(print_format % ('all', seen, num_targets.sum(), mean_pre, mean_rec, mean_ap, mf1))

    # Print results per class
    if verbose and num_classes > 1 and stats:
        for i, class_ in enumerate(ap_class):
            print(print_format % (names[class_], seen, num_targets[class_],
                                  precision[i], recall[i], aver_pre[i], f_1[i]))

    # Save JSON
    if save_json and mean_ap and json_dict:
        try:
            img_ids = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
            with open('results.json', 'w') as file:
                json.dump(json_dict, file)

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocogt = COCO('data/coco/annotations/instances_val2017.json')  # initialize COCO ground truth api
            cocodt = cocogt.loadRes('results.json')  # initialize COCO pred api

            cocoeval = COCOeval(cocogt, cocodt, 'bbox')
            cocoeval.params.imgIds = img_ids  # [:32]  # only evaluate these images
            cocoeval.evaluate()
            cocoeval.accumulate()
            cocoeval.summarize()
            mean_ap = cocoeval.stats[1]  # update mAP to pycocotools mAP
        except ImportError:
            print('WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    # Return results
    maps = np.zeros(num_classes) + mean_ap
    for i, class_ in enumerate(ap_class):
        maps[class_] = aver_pre[i]
    return (mean_pre, mean_rec, mean_ap, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco2017.yaml', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov4_last.pt', help='path to weights file')
    parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save_json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')
    opt = parser.parse_args()
    print(opt)

    with open(opt.hyp) as f_hyp:
        opt.hyp = yaml.safe_load(f_hyp)
    with open(opt.data) as f_data:
        opt.data = yaml.safe_load(f_data)

    test(opt.cfg,
         opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.iou_thres,
         opt.conf_thres,
         opt.nms_thres,
         opt.hyp)
