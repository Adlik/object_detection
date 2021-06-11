# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Provides utilities to model and train.
"""

import os
import time
import logging
from copy import deepcopy
import torch
from thop import profile


logger = logging.getLogger(__name__)


def select_device(device=''):
    """Select the device CPU or GPU

    :param device: 'cpu' or '0' or '0,1,2,3'
    :return: torch.device
    """

    strings = f'Using torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if not torch.cuda.is_available():
            raise AssertionError(f'CUDA unavailable, invalid device {device} requested')

    cuda = torch.cuda.is_available() and not cpu
    if cuda:
        num_gpu = torch.cuda.device_count()
        space = ' ' * len(strings)
        device_index = device.split(',') if device else range(num_gpu)
        for i, devices in enumerate(device_index):
            properties = torch.cuda.get_device_properties(i)
            strings += f"{'' if i == 0 else space}CUDA:{devices} ({properties.name}, " \
                       f"{properties.total_memory / 1024 ** 2}MB)\n"
    else:
        strings += 'CPU'

    logger.info(strings)
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    """pytorch-accurate time

    :return: time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bns):
    """
    In inference stage, we can integrate convolution layer and BN layer to reduce inference time
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    :param conv: convolution layer
    :param bn: batchnormalize layer
    :return: fused convolution layer
    """
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = conv.diag(bns.weight.div(torch.sqrt(bns.eps + bns.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bns.bias - bns.weight.mul(bns.running_mean).div(torch.sqrt(bns.running_var + bns.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv


def model_info(model, verbose=False, img_size=608):
    """
    print the model information,
    :param model: model
    :param verbose: if verbose is True, print the details of model
    :param img_size: image size
    :return: number of layers, parameters and gradients, GFLOPs
    """
    num_param = sum(x.numel() for x in model.parameters())
    num_grad = sum(x.numel() for x in model.parameters() if x.requires_grad)
    params_size = num_param * 4.0 / 1024 / 1024

    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, param) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, param.requires_grad, param.numel(), list(param.shape), param.mean(), param.std()))

    try:
        stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2   # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]   # expend if int/float
        gflops = (flops * img_size[0] / stride * img_size[1] / stride)        # img_size GFLOPs
    except ImportError:
        gflops = 0

    logger.info('Model Summary: %d layers, %dM parameters, %d gradients, %d GFLOPs', len(list(model.modules())),
                params_size, num_grad, gflops)


def intersect_dicts(checkpoint_state_dict, model_state_dict):
    """
    Dictionary intersection of matching keys and shapes using checkpoint_state_dict values
    :param checkpoint_state_dict: checkpoint state dict
    :param model_state_dict: model state dict
    :return: intersect dicts
    """

    return {key: value for key, value in checkpoint_state_dict.items() if model_state_dict[key].shape == value.shape}
