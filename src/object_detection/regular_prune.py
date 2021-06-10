# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E0611

"""Regular prune"""

import argparse
import logging
from copy import deepcopy
from terminaltables import AsciiTable
import numpy as np
import yaml

import torch
from .test import test
from .utils.utils import save_weights
from .utils.models import Model, load_darknet_weights
from .utils.prune_utils import obtain_bn_mask, parse_module_index, gather_bn_weights, prune_model_keep_size, \
    init_weights_from_orig_model, obtain_avg_forward_time, write_cfg

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main(opt):
    """Main"""

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(opt.data) as f_data:
        opt.data_dict = yaml.safe_load(f_data)  # data dict

    model = Model(opt.cfg, img_size).to(device)
    if opt.weights.endswith('.pt'):
        model.load_state_dict(torch.load(opt.weights)['model'])
    else:
        load_darknet_weights(model, opt.weights)
    logger.info('\nloaded weights from %s', opt.weights)

    logger.info("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model, opt)

    num_origin_parameters = num_parameters(model)

    # Get the index of the corresponding module
    cba_index, conv_index, prune_index = parse_module_index(model.module_dicts)

    # Contains the gamma parameter of BN layer for all pruning layers
    bn_weights = gather_bn_weights(model.module_list, prune_index)

    # The gamma parameters of the BN layer are arranged in ascending order
    sorted_bn = torch.sort(bn_weights)[0]

    percent = opt.percent
    logger.info('the required prune percent is %f', percent)
    threshold = prune_and_eval(model, sorted_bn, prune_index, opt, percent)

    num_remain_filters, mask_remain_filters = obtain_filters_mask(model, threshold, cba_index, prune_index)

    cba_index_mask = {index: mask.astype('float32') for index, mask in zip(cba_index, mask_remain_filters)}

    # add offset of BN bata to next layer
    pruned_model = prune_model_keep_size(model, cba_index, cba_index, cba_index_mask)

    logger.info("\nnow prune the model but keep size, actually add offset of BN bata to next layer,"
                "let's see how the mAP goes")
    with torch.no_grad():
        eval_model(pruned_model, opt)

    # Compressed model
    compact_module_dicts = deepcopy(model.module_dicts)
    for index, num in zip(cba_index, num_remain_filters):
        if compact_module_dicts[index]['type'] != 'convolutional':
            raise AssertionError('the index of compact module dicts is not convolutional')
        compact_module_dicts[index]['filters'] = str(num)

    compact_model = Model([model.hyperparams.copy()] + compact_module_dicts, img_size).to(device)
    num_compact_parameters = num_parameters(compact_model)

    init_weights_from_orig_model(compact_model, pruned_model, cba_index, conv_index, cba_index_mask)

    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    logger.info('\ntesting average forward time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output - compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        logger.info('Something wrong with the pruned model!')

    # Test the model in val set, and statistical model quantity
    logger.info('testing the mAP of final pruned model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model, opt)

    # Compare changes in the number of parameters and indicator performance before and after the pruning,
    metric_table = [
        ["Metric", "Unpruning", "Pruning"],
        ["mAP", f'{origin_model_metric[0][2]:.3f}', f'{compact_model_metric[0][2]:.3f}'],
        ["Parameters", f'{num_origin_parameters}', f'{num_compact_parameters}'],
        ["Inference", f'{pruned_forward_time:.3f}', f'{compact_forward_time:.3f}']
    ]
    print(AsciiTable(metric_table).table)

    # Generate the cfg file after the pruning and save the model
    pruned_cfg_name = opt.cfg.replace('/', f'/prune_{percent}_')
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_dicts)
    logger.info('Config files has been saved: %s', pruned_cfg_file)

    compact_model_name = opt.weights.replace('/', f'/prune_{percent}_')
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, compact_model_name)
    logger.info('Compact model has been saved: %s', compact_model_name)


def prune_and_eval(model, sorted_bn, prune_index, opt, percent=.0):
    """regular prune

    :param model: model
    :param sorted_bn: list of all BN sorting
    :param percent: percentage of the number of pruning channels
    :return: threshold
    """

    # The number of filters reserved must be a multiple of 8
    int_multiple = 8
    filter_switch = list(range(0, 1024, int_multiple))

    model_copy = deepcopy(model)

    # Set the threshold index at the pruning percent
    threshold_index = int(len(sorted_bn) * percent)

    # Get the threshold value of the alpha parameter, and all channels less than the threshold value are pruned
    threshold = sorted_bn[threshold_index]
    logger.info('Gamma value that less than %.4f are set to zero!', threshold)

    num_remain_bn = 0
    for index in prune_index:
        bn_module = model_copy.module_list[index][1]
        # The value of reserved BN channel is 1, others are 0
        mask = obtain_bn_mask(bn_module, threshold)
        num_layer_remain_bn = int(mask.sum())
        # When all BN values of this layer are less than thre, the eighth largest value of BN is taken as the thre.
        if num_layer_remain_bn < 8:
            layer_sort_bn = bn_module.weight.data.abs().clone()
            value_sort_bn = torch.sort(layer_sort_bn)[0]
            layer_threshold = value_sort_bn[-8]
            mask = obtain_bn_mask(bn_module, layer_threshold)
        else:
            # The number of channels pruned in each layer must be a multiple of 8
            for i, _ in enumerate(filter_switch):
                if num_layer_remain_bn < filter_switch[i]:
                    num_layer_remain_bn = filter_switch[i-1]
                    break
            layer_sort_bn = bn_module.weight.data.abs().clone()
            value_sort_bn = torch.sort(layer_sort_bn)[0]
            layer_threshold = value_sort_bn[-num_layer_remain_bn]
            mask = obtain_bn_mask(bn_module, layer_threshold)

        num_remain_bn += int(mask.sum())
        bn_module.weight.data.mul_(mask)

    logger.info("let's test the current model!")
    with torch.no_grad():
        mean_ap = eval_model(model_copy, opt)[0][2]

    logger.info('Number of channels has been reduced from %d to %d', len(sorted_bn), num_remain_bn)
    logger.info('Prune ratio: %.3f', 1 - num_remain_bn/len(sorted_bn))
    logger.info("mAP of the 'pruned' model is %.3f", mean_ap)

    return threshold


def obtain_filters_mask(model, threshold, cba_index, prune_index):
    """Gets the mask of the pruning channels

    :param model: model
    :param threshold: threshold
    :param cba_index: stores all convolution layers with BN layer
    :param prune_index: list of the index of pruning channels
    :return: num_remain_filters, mask_remain_filters
    """

    num_pruned_bn = 0
    num_total_bn = 0
    num_remain_filters = []
    mask_remain_filters = []

    # The number of filters reserved must be a multiple of 8
    int_multiple = 8
    filter_switch = list(range(0, 1024, int_multiple))

    # cba_index stores all convolution layers with BN layer (the previous layer of YOLO layer is without BN layer)
    for index in cba_index:
        bn_module = model.module_list[index][1]
        if index in prune_index:
            mask = obtain_bn_mask(bn_module, threshold).cpu().numpy()
            num_layer_remain_bn = int(mask.sum())
            if num_layer_remain_bn < 8:
                layer_sort_bn = bn_module.weight.data.abs().clone()
                value_sort_bn = torch.sort(layer_sort_bn)[0]
                layer_threshold = value_sort_bn[-8]
                mask = obtain_bn_mask(bn_module, layer_threshold).cpu().numpy()
            else:
                for i, _ in enumerate(filter_switch):
                    if num_layer_remain_bn < filter_switch[i]:
                        num_layer_remain_bn = filter_switch[i - 1]
                        break
                layer_sort_bn = bn_module.weight.data.abs().clone()
                value_sort_bn = torch.sort(layer_sort_bn)[0]
                layer_threshold = value_sort_bn[-num_layer_remain_bn]
                mask = obtain_bn_mask(bn_module, layer_threshold).cpu().numpy()

            num_remain_bn = int(mask.sum())
            num_pruned_bn = num_pruned_bn + mask.shape[0] - num_remain_bn

            if num_remain_bn == 0:
                print("Channels would be all pruned!")
                raise Exception

            logger.info('layer index: %d \t total channel: %d \t remaining channel: %d',
                        index, mask.shape[0], num_remain_bn)
        else:
            mask = np.ones(bn_module.weight.data.shape)
            num_remain_bn = mask.shape[0]
        num_total_bn += mask.shape[0]
        num_remain_filters.append(num_remain_bn)
        mask_remain_filters.append(mask.copy())

    prune_ratio = num_pruned_bn / num_total_bn
    logger.info('Prune channels: %d \t Prune ratio: %.3f', num_pruned_bn, prune_ratio)

    return num_remain_filters, mask_remain_filters


def eval_model(model, opt):
    """Test the model"""
    return test(opt.cfg, opt.data_dict, weights=opt.weights, batch_size=16, img_size=opt.img_size, iou_thres=0.5,
                conf_thres=0.001, nms_thres=0.5, save_json=False, model=model)


def num_parameters(model):
    """Computer the number of model parameters"""
    return sum([param.nelement() for param in model.parameters()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco2017.yaml', help='data file path')
    parser.add_argument('--weights', type=str, default='weights/sparsity_last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.3, help='channels prune percent')
    parser.add_argument('--img_size', type=int, default=608, help='inference size')
    args = parser.parse_args()
    logger.info(args)
    main(args)
