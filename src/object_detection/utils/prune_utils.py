# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Provides utilities to prune and knowledge distillation.
"""

import time
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# pylint: disable=R0911, R0912, R0914
def get_input_mask(module_dicts, index, cba_index_mask):
    """get the channel index of the i-1 layer

    :param module_dicts: module dicts
    :param index: index
    :param cba_index_mask: {index: mask_remain_filters}
    :return:
    """

    if index == 0:
        return np.ones(3)

    if module_dicts[index - 1]['type'] == 'convolutional':
        return cba_index_mask[index - 1]
    if module_dicts[index - 1]['type'] == 'shortcut':
        return cba_index_mask[index - 2]
    if module_dicts[index - 1]['type'] == 'route':
        route_in_indexs = []
        for layer_i in module_dicts[index - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_indexs.append(index - 1 + int(layer_i))
            else:
                route_in_indexs.append(int(layer_i))

        if len(route_in_indexs) == 1:
            mask = cba_index_mask[route_in_indexs[0]]
            if 'groups' in module_dicts[index - 1]:
                mask = mask[(mask.shape[0]//2):]
            return mask

        if len(route_in_indexs) == 2:
            if module_dicts[route_in_indexs[0]]['type'] == 'upsample':
                mask1 = cba_index_mask[route_in_indexs[0] - 1]
            elif module_dicts[route_in_indexs[0]]['type'] == 'convolutional':
                mask1 = cba_index_mask[route_in_indexs[0]]
            if module_dicts[route_in_indexs[1]]['type'] == 'convolutional':
                mask2 = cba_index_mask[route_in_indexs[1]]
            else:
                mask2 = cba_index_mask[route_in_indexs[1] - 1]
            return np.concatenate([mask1, mask2])

        if len(route_in_indexs) == 4:
            # the last route in the SPP structure
            mask = cba_index_mask[route_in_indexs[-1]]
            return np.concatenate([mask, mask, mask, mask])

    # yolo-tiny
    if module_dicts[index - 1]['type'] == 'maxpool':
        if module_dicts[index - 2]['type'] == 'route':
            return get_input_mask(module_dicts, index - 1, cba_index_mask)
        return cba_index_mask[index - 2]

    return None


def _update_activation(i, pruned_model, activation, cba_index):
    """update activation"""
    next_index = i + 1
    if pruned_model.module_dicts[next_index]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_index][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_index in cba_index:
            next_bn = pruned_model.module_list[next_index][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)


def obtain_bn_mask(bn_module, threshold):
    """Obtain bn mask

    :param bn_module: bn layer
    :param threshold: the percentage of the number of pruning channels
    :return: mask
    """
    threshold = threshold.cuda()
    mask = bn_module.weight.data.abs().ge(threshold).float()
    return mask


def parse_module_index(module_dicts):
    """Gets a list of indexes for different modules

    :param module_dicts: module dict
    :return: cba_index, conv_index, prune_index
    """

    cba_index = []
    conv_index = []
    ignore_index = set()

    for i, module_def in enumerate(module_dicts):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                cba_index.append(i)
            else:
                conv_index.append(i)
            if module_dicts[i+1]['type'] == 'maxpool' and module_dicts[i+2]['type'] == 'route':
                # The previous CBL layer of the spp layer is not pruned to distinguish tiny-yolo
                ignore_index.add(i)
            if module_dicts[i+1]['type'] == 'route' and 'groups' in module_dicts[i+1]:
                ignore_index.add(i)

        elif module_def['type'] == 'shortcut':
            # The previous convolution layer of the shortcut layer is not pruned
            ignore_index.add(i-1)
            identity_index = (i + int(module_def['from']))
            if module_dicts[identity_index]['type'] == 'convolutional':
                ignore_index.add(identity_index)
            elif module_dicts[identity_index]['type'] == 'shortcut':
                ignore_index.add(i-1)

        elif module_def['type'] == 'upsample':
            # The previous convolution layer of the upsample layer is not pruned
            ignore_index.add(i-1)

    prune_index = [index for index in cba_index if index not in ignore_index]

    return cba_index, conv_index, prune_index


def gather_bn_weights(module_list, prune_index):
    """Gather bn weights that will be pruned layers

    :param module_list: the module list
    :param prune_index: the index of pruned layers
    :return: bn_weights(tensor)
    """

    list_bn_size = [module_list[index][1].weight.data.shape[0] for index in prune_index]
    bn_weights = torch.zeros(sum(list_bn_size))
    index = 0
    for idx, size in zip(prune_index, list_bn_size):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def bn_l1_regularization(module_list, penalty_factor, prune_index, epoch, epochs):
    """L1 regularization of the BN layer

    :param module_list: module list
    :param penalty_factor: penalty factor
    :param prune_index: the index of prune layer
    :param epoch: epoch
    :param epochs: the total number of epochs
    :return: L1-regularized BN weight
    """

    penalty_factor = penalty_factor if epoch <= epochs * 0.5 else penalty_factor * 0.1
    for index in prune_index:
        bn_module = module_list[index][1]
        bn_module.weight.grad.data.add_(penalty_factor * torch.sign(bn_module.weight.data))  # L1 regularization


def prune_model_keep_size(model, prune_index, cba_index, cba_index_mask):
    """Adds the bias parameter of the BN layer to the activation function

    :param model: model
    :param prune_index: the index of prune layer
    :param cba_index: the index of CBL layer
    :param cba_index_mask: {index: mask_remain_filters}
    :return: pruned_model
    """

    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_dicts):
        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_index:
                mask = torch.from_numpy(cba_index_mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                _update_activation(i, pruned_model, activation, cba_index)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            activation1 = activations[i - 1]
            from_layer = int(model_def['from'])
            activation2 = activations[i + from_layer]
            activation = activation1 + activation2
            _update_activation(i, pruned_model, activation, cba_index)
            activations.append(activation)

        elif model_def['type'] == 'route':
            # SPP doesn't participate in pruning. Route in SPP does not need to be updated, only placeholder
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                if 'group' in model_def:
                    activation = activation[(activation.shape[0] // 2):]
                _update_activation(i, pruned_model, activation, cba_index)
            elif len(from_layers) == 2:
                activation1 = activations[i + from_layers[0]]
                activation2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((activation1, activation2))
                _update_activation(i, pruned_model, activation, cba_index)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            activations.append(activations[i-1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':
            # Distinguish between spp and tiny
            if model.module_dicts[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                _update_activation(i, pruned_model, activation, cba_index)
                activations.append(activation)

    return pruned_model


def init_weights_from_orig_model(compact_model, orig_model, cba_index, conv_index, cba_index_mask):
    """Pruned model's weights that initialize weights from the initial model

    :param compact_model: the structure of pruned model
    :param orig_model: the structure of unpruned model
    :param cba_index: the index of CBL layers
    :param conv_index: the index of conv layers that contain only the convolution layer, not bn
    :param cba_index_mask: {key: value} --> {cba_index: mask}
    :return: None
    """

    for index in cba_index:
        compact_cba = compact_model.module_list[index]
        orig_cba = orig_model.module_list[index]
        out_channel_index = np.argwhere(cba_index_mask[index])[:, 0].tolist()   # output channel index after pruning

        compact_bn, orig_bn = compact_cba[1], orig_cba[1]
        compact_bn.weight.data = orig_bn.weight.data[out_channel_index].clone()
        compact_bn.bias.data = orig_bn.bias.data[out_channel_index].clone()
        compact_bn.running_mean.data = orig_bn.running_mean.data[out_channel_index].clone()
        compact_bn.running_var.data = orig_bn.running_var.data[out_channel_index].clone()

        input_mask = get_input_mask(orig_model.module_dicts, index, cba_index_mask)
        in_channel_index = np.argwhere(input_mask)[:, 0].tolist()     # input channel index after pruning
        compact_conv, orig_conv = compact_cba[0], orig_cba[0]
        tmp = orig_conv.weight.data[:, in_channel_index, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_index, :, :, :].clone()

    for index in conv_index:
        compact_conv = compact_model.module_list[index][0]
        orig_conv = orig_model.module_list[index][0]

        input_mask = get_input_mask(orig_model.module_dicts, index, cba_index_mask)
        in_channel_index = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = orig_conv.weight.data[:, in_channel_index, :, :].clone()
        compact_conv.bias.data = orig_conv.bias.data.clone()


def obtain_avg_forward_time(inputs, model, repeat=200):
    """Compute the average inference time

    :param inputs:  (batch_size, channels, height, width)
    :param model:  model
    :param repeat:  the number of inputs
    :return:  average inference time, the model output
    """

    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            output = model(inputs)[0]
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


def write_cfg(cfg_file, module_dicts):
    """Builds the cfg file that compact model

    :param cfg_file: the cfg file that unpruned model
    :param module_dicts:  module dicts
    :return: the cfg file that compact model
    """

    with open(cfg_file, 'w') as file:
        for module_def in module_dicts:
            file.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue
                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                    file.write(f"{key}={value}\n")
            file.write("\n")
    return cfg_file


def distillation_loss(output_student, output_teacher, num_classes, batch_size, temperature=3.0):
    """object detection distillation loss

    :param output_student: the output of student model
    :param output_teacher:  the output of teacher model
    :param num_classes:  classes number
    :param batch_size:  batch size
    :return: loss_st * Lambda_ST
    """

    lambda_st = 0.01
    criterion_st = torch.nn.KLDivLoss(reduction='batchmean')
    output_student = torch.cat([i.view(-1, num_classes + 5) for i in output_student])
    output_teacher = torch.cat([i.view(-1, num_classes + 5) for i in output_teacher])
    log_softmax_student = nn.functional.log_softmax(output_student/temperature, dim=1)
    softmax_teacher = nn.functional.softmax(output_teacher/temperature, dim=1)
    loss_st = criterion_st(log_softmax_student, softmax_teacher) * (temperature * temperature) / batch_size
    return loss_st * lambda_st
