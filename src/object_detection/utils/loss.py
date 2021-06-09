# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Provides utilities to loss
"""

import math
import torch
import torch.nn as nn


def compute_loss(pred, targets, model):
    """Compute loss

    :param pred: pred
    :param targets: targets
    :param model: model
    :return: loss, torch.cat((loss_box, loss_obj, loss_cls, loss)).detach()
    """

    torch_type = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
    loss_cls, loss_box, loss_obj = torch_type([0]), torch_type([0]), torch_type([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    hyp = model.hyp
    arc = model.arc  # (default, uCE, uBCE) detection architectures

    # Define criteria
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch_type([hyp['cls_pw']]))
    bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch_type([hyp['obj_pw']]))
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()

    # Focal loss
    if 'F' in arc:
        gamma = hyp['fl_gamma']
        bce_cls = FocalLoss(bce_cls, gamma)
        bce_obj = FocalLoss(bce_obj, gamma)
        bce = FocalLoss(bce, gamma)
        cross_entropy = FocalLoss(cross_entropy, gamma)

    # Compute losses
    for index_layer, pred_layer in enumerate(pred):
        target_image, anchor, grid_y, grid_x = indices[index_layer]
        targets_obj = torch.zeros_like(pred_layer[..., 0])

        # Compute losses
        num_targets = len(target_image)
        if num_targets:
            sub_pred = pred_layer[target_image, anchor, grid_y, grid_x]  # prediction subset corresponding to targets
            targets_obj[target_image, anchor, grid_y, grid_x] = 1.0

            # box loss
            pred_xy = torch.sigmoid(sub_pred[:, 0:2])  # pred_xy = pred_xy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pred_box = torch.cat((pred_xy, torch.exp(sub_pred[:, 2:4]) * anchor_vec[index_layer]), 1)
            giou = bbox_giou(pred_box.t(), tbox[index_layer], x1y1x2y2=False, giou=True)
            loss_box += (1.0 - giou).mean()

            # cls loss (only if multiple classes)
            if 'default' in arc and model.num_classes > 1:
                target = torch.zeros_like(sub_pred[:, 5:])
                target[range(num_targets), tcls[index_layer]] = 1.0
                loss_cls += bce_cls(sub_pred[:, 5:], target)

        # obj loss
        if 'default' in arc:  # seperate obj and cls
            loss_obj += bce_obj(pred_layer[..., 4], targets_obj)

        elif 'bce' in arc:  # unified BCE (80 classes)
            target = torch.zeros_like(pred_layer[..., 5:])
            if num_targets:
                target[target_image, anchor, grid_y, grid_x, tcls[index_layer]] = 1.0
            loss_obj += bce(pred_layer[..., 5:], target)

        elif 'cross_entropy' in arc:  # unified CE (1 background + 80 classes)
            target = torch.zeros_like(pred_layer[..., 0], dtype=torch.long)
            if num_targets:
                target[target_image, anchor, grid_y, grid_x] = tcls[index_layer] + 1
            loss_cls += cross_entropy(pred_layer[..., 4:].view(-1, model.num_classes + 1), target.view(-1))

    loss_box *= hyp['box']
    loss_obj *= hyp['obj']
    loss_cls *= hyp['cls']
    loss = loss_box + loss_obj + loss_cls
    return loss, torch.cat((loss_box, loss_obj, loss_cls, loss)).detach()


def build_targets(model, targets):
    """Build targets between the defaults bbox and ground truth.

    :param model: model
    :param targets: [image, class, x, y, w, h]
    :return: targets_cls, targets_box, indices, anchors
    """

    num_targets = len(targets)
    targets_cls, targets_box, indices, anchors = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i in model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            num_grid, anchor_vec = model.module.module_list[i].num_grid, model.module.module_list[i].anchor_vec
        else:
            num_grid, anchor_vec = model.module_list[i].num_grid, model.module_list[i].anchor_vec

        # IoU of targets with anchors
        target, anchor = targets, []
        grid_wh = target[:, 4:6] * num_grid
        if num_targets:
            iou = torch.stack([wh_iou(x, grid_wh) for x in anchor_vec], 0)

            use_best_anchor = False
            if use_best_anchor:
                iou, anchor = iou.max(0)        # best iou and anchor
            else:
                num_anchors = len(anchor_vec)   # use all anchors
                anchor = torch.arange(num_anchors).view((1, -1)).repeat([1, num_targets]).view(-1)
                target = targets.repeat([num_anchors, 1])
                grid_wh = grid_wh.repeat([num_anchors, 1])
                iou = iou.view(-1)              # use all iou

            # reject anchors below iou_threshold (Optional, increases P, lowers R)
            reject = True
            if reject:
                j = iou > model.hyp['iou_threshold']
                target, anchor, grid_wh = target[j], anchor[j], grid_wh[j]

        # Indices
        target_image, cls = target[:, :2].long().t()
        grid_xy = target[:, 2:4] * num_grid
        grid_x_indice, grid_y_indice = grid_xy.long().t()
        indices.append((target_image, anchor, grid_y_indice, grid_x_indice))

        # GIoU
        grid_xy -= grid_xy.floor()
        targets_box.append(torch.cat((grid_xy, grid_wh), 1))
        anchors.append(anchor_vec[anchor])

        # Class
        targets_cls.append(cls)
        if cls.shape[0]:
            if cls.max() > model.num_classes:
                raise AssertionError('Target classes exceed model classes.')

    return targets_cls, targets_box, indices, anchors


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    """

    def __init__(self, loss_function, gamma=0.5, alpha=1, reduction='mean'):
        super().__init__()
        loss_function.reduction = 'none'
        self.loss_function = loss_function
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        """Compute focal loss"""
        loss = self.loss_function(inputs, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

        return loss


def bbox_iou(box1, box2, x1y1x2y2=True, giou=False, diou=False, ciou=False, eps=1e-9):
    """Compute the IoU of box1 to box2. box1 is 4, box2 is nx4

    :param box1: box1
    :param box2: box2
    :param x1y1x2y2: xyxy or xywh
    :param giou: giou
    :param diou: diou
    :param ciou: ciou
    :param eps: 1e-9
    :return: iou or giou, diou, ciou
    """

    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[2], box1[3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        box1_x1, box1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        box1_y1, box1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        box2_x1, box2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        box2_y1, box2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
            (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union area
    box1_w, box1_h = box1_x2 - box1_x1, box1_y2 - box1_y1 + eps
    box2_w, box2_h = box2_x2 - box2_x1, box2_y2 - box2_y1 + eps
    union = box1_w * box1_h + box2_w * box2_h - inter + eps

    iou = inter / union
    if giou or diou or ciou:
        convex_width = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)   # convex width
        convex_height = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)  # convex height
        if ciou or diou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + eps
            center_distance_squared = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 +
                                       (box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4
            if diou:
                return iou - center_distance_squared / convex_diagonal_squared
            if ciou:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                upsilon = (4 / math.pi ** 2) * torch.pow(torch.atan(box2_w / box2_h) - torch.atan(box1_w / box1_h), 2)
                with torch.no_grad():
                    alpha = upsilon / ((1 + eps) - iou + upsilon)
                return iou - (center_distance_squared / convex_diagonal_squared + upsilon * alpha)
        else:  # giou https://arxiv.org/pdf/1902.09630.pdf
            convex_area = convex_width * convex_height + eps
            return iou - (convex_area - union) / convex_area

    return iou


def bbox_giou(box1, box2, x1y1x2y2=True, giou=False):
    """Compute the giou of box1 to box2. box1 is 4, box2 is nx4

    :param box1: box1
    :param box2: box2
    :param x1y1x2y2: xyxy or xywh
    :param giou: giou
    :return: iou or giou
    """
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[2], box1[3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        box1_x1, box1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        box1_y1, box1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        box2_x1, box2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        box2_y1, box2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
                 (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union Area
    union_area = ((box1_x2 - box1_x1) * (box1_y2 - box1_y1) + 1e-16) + \
                 (box2_x2 - box2_x1) * (box2_y2 - box2_y1) - inter_area

    iou = inter_area / union_area
    if giou:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(box1_x1, box2_x1), torch.max(box1_x2, box2_x2)
        c_y1, c_y2 = torch.min(box1_y1, box2_y1), torch.max(box1_y2, box2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-16  # convex area
        return iou - (c_area - union_area) / c_area  # giou

    return iou


def wh_iou(box1, box2):
    """
    Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2

    :param box1: anchor (default anchor) --> 2
    :param box2: targets (ground truth)  --> n x 2
    :return: IoU of box1 to box2
    """

    box2 = box2.t()

    # w, h = box1
    box1_w, box1_h = box1[0], box1[1]
    box2_w, box2_h = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(box1_w, box2_w) * torch.min(box1_h, box2_h)

    # Union Area
    union_area = (box1_w * box1_h + 1e-16) + box2_w * box2_h - inter_area

    return inter_area / union_area
