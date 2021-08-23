# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructed model by cfg file"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_utils import fuse_conv_and_bn, model_info

logger = logging.getLogger(__name__)

ONNX_EXPORT = False


class Model(nn.Module):
    """Create the YOLOv4 model from cfg file"""

    def __init__(self, cfg, img_size=608, arc='default'):
        super().__init__()
        # Converts the cfg file that defines the model to a list of elements as dictionaries
        if isinstance(cfg, str):
            self.module_dicts = parse_model_cfg(cfg)
        elif isinstance(cfg, list):
            self.module_dicts = cfg

        self.module_list, self.routs, self.hyperparams = create_modules(self.module_dicts, img_size, arc)
        self.yolo_layers = _get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)           # (int64) number of images seen during training
        self.img_size = img_size

        # Print model information
        logger.info('Model information')
        model_info(self, verbose=False, img_size=self.img_size)

    def forward(self, inputs):
        """forward"""
        img_size = inputs.shape[-2:]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_dicts, self.module_list)):
            module_type = module_def['type']
            if module_type in ['convolutional', 'upsample', 'maxpool']:
                inputs = module(inputs)
            elif module_type == 'route':
                layers = [int(x) for x in module_def['layers'].split(',')]
                if len(layers) == 1:
                    inputs = layer_outputs[layers[0]]
                    if 'groups' in module_def:
                        inputs = inputs[:, (inputs.shape[1] // 2):]
                else:
                    try:
                        inputs = torch.cat([layer_outputs[i] for i in layers], 1)
                    except IndexError:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        inputs = torch.cat([layer_outputs[i] for i in layers], 1)
            elif module_type == 'shortcut':
                inputs = inputs + layer_outputs[int(module_def['from'])]
            elif module_type == 'yolo':
                inputs = module(inputs, img_size)
                output.append(inputs)
            layer_outputs.append(inputs if i in self.routs else [])

        if self.training:
            return output
        if ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            num_classes = self.module_list[self.yolo_layers[0]].num_classes   # number of classes
            return output[5:5 + num_classes].t(), output[:4].t()   # ONNX scores, boxes
        infer_output, pred = list(zip(*output))  # inference output, training output
        return torch.cat(infer_output, 1), pred

    def fuse(self):
        """Fuse Conv2d and BatchNorm2d layers throughout model
        https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
        :return: module_list that fuse the BN and conv layer
        """

        fused_list = nn.ModuleList()
        for layer in list(self.children())[0]:
            if isinstance(layer, nn.Sequential):
                for i, batchnorm in enumerate(layer):
                    if isinstance(batchnorm, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = layer[i - 1]
                        fused = fuse_conv_and_bn(conv, batchnorm)
                        layer = nn.Sequential(fused, *list(layer.children())[i + 1:])
                        break
            fused_list.append(layer)
        self.module_list = fused_list
        model_info(self, verbose=False, img_size=self.img_size)


def create_modules(module_dicts, img_size, arc):
    """
    Constructs module list of layer blocks from module configuration in module_dicts
    :param module_dicts: dict that contains all module from cfg file
    :param img_size: images size
    :param arc: the network architecture type
    :return: module_list
    :return: routs: list of layers which rout to deeper layers
    :return: hyperparams: dict

    module_list:
    ModuleList(
  (0): Sequential(
    (Conv2d): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (BatchNorm2d): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): Mish()
  )
  (1): Sequential(
    (Conv2d): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (BatchNorm2d): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): Mish()
  )
  (2): Sequential(
    (Conv2d): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (BatchNorm2d): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): Mish()
  )
  (3): Sequential()
  )
    """

    hyperparams = module_dicts.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, module_def in enumerate(module_dicts):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            batch_norm = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=pad,
                                                   bias=not batch_norm))
            if batch_norm:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if module_def['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif module_def['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:   # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif module_def['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            if 'groups' in module_def:
                filters = filters // 2
            routs.extend([layer if layer > 0 else layer + i for layer in layers])

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            layer = int(module_def['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif module_def['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in module_def['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(anchors=module_def['anchors'][mask],  # anchor list
                                num_classes=int(module_def['classes']),    # number of classes
                                img_size=img_size,
                                yolo_index=yolo_index,  # 0, 1 or 2
                                arc=arc)  # yolo architecture

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)\
            try:
                if arc in ['defaultpw', 'Fdefaultpw']:  # default with positive weights
                    init_bias = [-4, -3.6]  # obj, cls
                elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    init_bias = [-5.5, -4.0]
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    init_bias = [0, -8.5]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    init_bias = [10, -0.1]
                elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                    init_bias = [-2.1, -1.8]
                elif arc == ['uFBCE', 'uFBCEpw']:  # unified FocalBCE (5120 obj, 80 classes)
                    init_bias = [0, -6.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    init_bias = [7.7, -1.1]

                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 = 3 x 85
                bias[:, 4] += init_bias[0] - bias[:, 4].mean()   # obj
                bias[:, 5:] += init_bias[1] - bias[:, 5:].mean()  # cls
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
            except IndexError:
                print('WARNING: smart bias initialization failure.')

        else:
            print('WARNING: Unrecognized Layer Type:' + module_def['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs, hyperparams


def parse_model_cfg(cfg_path):
    """Parses the yolov4 layer configuration file and returns the list module definitions dict

    :param cfg_path: cfg/yolov4.cfg
    :return: module_dicts(list): module definitions dict from cfg file
    [{'type': 'convolutional', 'batch_normalize': '1', 'filters': '64', 'size': '3',  \
        'stride': '1', 'pad': '1', 'activation': 'mish'}, ......,
        {'type': 'shortcut', 'from': '-3', 'activation': 'linear'}]
    """

    with open(cfg_path, 'r') as file:
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x and not x.startswith('#')]

    module_dicts = []  # module definitions
    for line in lines:
        if line.startswith('['):
            module_dicts.append({})
            module_dicts[-1]['type'] = line[1:-1].rstrip()
            if module_dicts[-1]['type'] == 'convolutional':
                module_dicts[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split('=')
            key = key.rstrip()

            if 'anchors' in key:
                module_dicts[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            else:
                module_dicts[-1][key] = val.strip()

    return module_dicts


class Mish(nn.Module):
    """Mish activation function.

    Mish(x) = x * tanh(ln(1+e^x))
    """
    @classmethod
    def forward(cls, inputs):
        """forward"""
        return inputs.mul(torch.tanh(F.softplus(inputs)))


class YOLOLayer(nn.Module):
    """YOLO head

    Attributes:
        anchors: 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
        num_classes: num_classes
        img_size: images size
        yolo_index: the index of yolo layer
        arc: arc
    """
    def __init__(self, anchors, num_classes, img_size, yolo_index, arc):
        super().__init__()

        self.anchors = torch.Tensor(anchors)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_x = 0   # initialize number of x gridpoints
        self.num_y = 0   # initialize number of y gridpoints
        self.arc = arc

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]   # stride of this layer
            num_x = int(img_size[1] / stride)  # number x grid points
            num_y = int(img_size[0] / stride)  # number y grid points
            self.create_grids(img_size, (num_x, num_y))

    def forward(self, pred, img_size):
        """forward"""
        if ONNX_EXPORT:
            batch_size = 1
        else:
            batch_size, num_y, num_x = pred.shape[0], pred.shape[-2], pred.shape[-1]
            if (self.num_x, self.num_y) != (num_x, num_y):
                self.create_grids(img_size, (num_x, num_y), pred.device, pred.dtype)

        # pred.view(batch_size, 255, 13, 13) -- > (bs, 3, 13, 13, 85) # (bs, anchors, grid, grid, classes+xywh)
        pred = pred.view(batch_size, self.num_anchors, self.num_classes+5, self.num_y, self.num_x)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return pred
        if ONNX_EXPORT:
            # Constants can not be broadcast, ensure correct shape!
            num_gu = self.num_grid.repeat((1, self.num_anchors * self.num_x * self.num_y, 1))
            grid_xy = self.grid_xy.repeat((1, self.num_anchors, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((1, 1, self.num_x, self.num_y, 1)).view((1, -1, 2)) / num_gu

            pred = pred.view(-1, 5 + self.num_classes)
            center_xy = torch.sigmoid(pred[..., 0:2]) + grid_xy[0]  # x, y
            w_h = torch.exp(pred[..., 2:4]) * anchor_wh[0]    # width, height
            pred_conf = torch.sigmoid(pred[:, 4:5])          # conf
            pred_cls = F.softmax(pred[:, 5:85], 1) * pred_conf  # SSD-like conf
            return torch.cat((center_xy / num_gu[0], w_h, pred_conf, pred_cls), 1).t()
        # inference
        # s = 1.5 , scale_xy (pxy = pxy * s - (s - 1) / 2)
        infer_output = pred.clone()   # inference output
        infer_output[..., 0:2] = torch.sigmoid(infer_output[..., 0:2]) + self.grid_xy   # xy
        infer_output[..., 2:4] = torch.exp(infer_output[..., 2:4]) * self.anchor_wh   # wh yolo method
        infer_output[..., :4] *= self.stride

        if 'default' in self.arc:    # seperate obj and cls
            torch.sigmoid_(infer_output[..., 4:])
        elif 'BCE' in self.arc:      # unified BCE (80 classes)
            torch.sigmoid_(infer_output[..., 5:])
            infer_output[..., 4] = 1
        elif 'CE' in self.arc:       # unified CE (1 background + 80 classes)
            infer_output[..., 4:] = F.softmax(infer_output[..., 4:], dim=4)
            infer_output[..., 4] = 1

        if self.num_classes == 1:
            infer_output[..., 5] = 1   # single-class model https://github.com/ultralytics/yolov3/issues/235

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return infer_output.view(batch_size, -1, 5 + self.num_classes), pred

    def create_grids(self, img_size=608, num_grid=(13, 13), devices='cpu', torch_type=torch.float32):
        """Create the grid that generates Anchors

        :param self: self
        :param img_size: img_size
        :param num_grid: the size of the feature map that detected
        :param devices: devices
        :param torch_type: torch.float32
        :return: grid_xy and anchor_wh
        """

        num_x, num_y = num_grid
        self.img_size = max(img_size)
        self.stride = self.img_size / max(num_grid)

        # build xy offsets
        y_grid, x_grid = torch.meshgrid([torch.arange(num_y), torch.arange(num_x)])
        self.grid_xy = torch.stack((x_grid, y_grid), 2).to(devices).type(torch_type).view((1, 1, num_x, num_y, 2))

        # build wh gains
        self.anchor_vec = self.anchors.to(devices) / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2).to(devices).type(torch_type)
        self.num_grid = torch.Tensor(num_grid).to(devices)
        self.num_x = num_x
        self.num_y = num_y


def _get_yolo_layers(models):
    """the index of YOLO output layer"""

    return [i for i, x in enumerate(models.module_dicts) if x['type'] == 'yolo']


def load_darknet_weights(models, weights, cutoffs=-1):
    """Parses and loads the weights stored in '.weights'

    :param models: models
    :param weights: the weights stored in '.weights
    :param cutoffs: loaded cutoff between 0 and cutoff, if cutoff = -1 all are loaded.
    :return:
    """

    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoffs = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoffs = 15
    elif file == 'yolov4.conv.137':
        cutoffs = 137
    elif file == 'yolov4-tiny.conv.29':
        cutoffs = 29

    # Read weights file
    with open(weights, 'rb') as f_weights:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        models.version = np.fromfile(f_weights, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        models.seen = np.fromfile(f_weights, dtype=np.int64, count=1)    # (int64) number of images seen during training

        weights = np.fromfile(f_weights, dtype=np.float32)   # The rest are weights

    ptr = 0
    for i, (module_def, module) in enumerate(zip(models.module_dicts[:cutoffs], models.module_list[:cutoffs])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_bias = bn_layer.bias.numel()
                # Bias
                bn_bias = torch.from_numpy(weights[ptr:ptr + num_bias]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_bias)
                ptr += num_bias
                # Weights
                bn_weight = torch.from_numpy(weights[ptr:ptr + num_bias]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_weight)
                ptr += num_bias
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_bias]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_bias
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_bias]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_bias
                # Load conv.weights
                num_weight = conv_layer.weight.numel()
                conv_weight = torch.from_numpy(weights[ptr:ptr + num_weight]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_weight)
                ptr += num_weight
            else:
                if os.path.basename(file) in ['yolov4.weights', 'yolov4-tiny.weights', 'yolov3.weights']:
                    # Load weights, for example, 'yolov3.weights' or 'yolov3-tiny-weights.'
                    num_bias = 255
                    ptr += num_bias
                    num_weight = int(models.module_dicts[i-1]['filters']) * 255
                    ptr += num_weight
                else:
                    # Load conv.bias
                    num_bias = conv_layer.bias.numel()
                    conv_bias = torch.from_numpy(weights[ptr:ptr + num_bias]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_bias)
                    ptr += num_bias
                    # Load conv.weights
                    num_weight = conv_layer.weight.numel()
                    conv_weight = torch.from_numpy(weights[ptr:ptr + num_weight]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_weight)
                    ptr += num_weight
    if ptr != len(weights):
        raise AssertionError('The weights doesn’t match the models')
    return cutoffs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../cfg/yolov4.cfg', help='model cfg file')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu')
    parser.add_argument('--weights', type=str, default='../weights/yolov4.conv.137', help='initial weights path')
    opt = parser.parse_args()

    if opt.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # pylint: disable=C0103
    # Create model
    print(device)
    model = Model(opt.cfg).to(device)
    load_darknet_weights(model, opt.weights)
