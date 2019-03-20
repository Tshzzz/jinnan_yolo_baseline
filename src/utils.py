#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""

import torch
import numpy as np



def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()

    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias));
    start = start + num_b

    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight));
    start = start + num_w

    return start

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b

    conv_weight = torch.from_numpy(buf[start:start + num_w])
    conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight))
    start = start + num_w
    return start

def bbox_iou(box1, box2):

    mx = np.min((box1[:, 0], box2[:, 0]), axis=0)
    Mx = np.max((box1[:, 2], box2[:, 2]), axis=0)
    my = np.min((box1[:, 1], box2[:, 1]), axis=0)
    My = np.max((box1[:, 3], box2[:, 3]), axis=0)
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh


    cw = np.clip(cw,0,1)
    ch = np.clip(ch,0,1)

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch

    uarea = area1 + area2 - carea

    return carea / uarea


def py_cpu_nms(dets, scores, thresh):
    # dets:(m,5)  thresh:scaler

    temp_len = 0  # np.max(dets[:,2]) * 0.05

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 0] + dets[:, 2]  # dets[:, 2]#
    y2 = dets[:, 1] + dets[:, 3]  # dets[:, 3]#

    areas = (y2 - y1 + temp_len) * (x2 - x1 + temp_len)

    keep = []

    index = scores.argsort()[::-1][:200]


    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + temp_len)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + temp_len)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]

    return keep
