# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the object_detection.utils.models modules """

import argparse
from object_detection.utils.models import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../src/object_detection/cfg/yolov4.cfg', help='model cfg file')
    parser.add_argument('--weights', type=str, default='../src/object_detection/weights/yolov4.conv.137',
                        help='initial weights path')
    opt = parser.parse_args()

    model = Model(opt.cfg).cuda()
