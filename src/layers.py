#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""

import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self, inplane, outplane, kernel_size, stride=1, pad=1):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out


class residual_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(residual_block, self).__init__()
        self.conv1 = conv_block(in_planes, in_planes // 2, kernel_size=1, stride=stride, pad=0)
        self.conv2 = conv_block(in_planes // 2, out_planes, kernel_size=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + out
        return out

class pred_module(nn.Module):
    def __init__(self, inplance, cls_num, bbox_num):
        super(pred_module, self).__init__()

        self.cls_num = cls_num
        self.bbox_num = bbox_num

        self.pre_cls_layer = nn.Conv2d(inplance, cls_num * bbox_num, 1, stride=1, padding=0, bias=True)
        self.pre_loc_layer = nn.Conv2d(inplance, 4 * bbox_num, 1, stride=1, padding=0, bias=True)
        self.pre_conf_layer = nn.Conv2d(inplance, 1 * bbox_num, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        pred_cls = self.pre_cls_layer(x).permute(0, 2, 3, 1)
        pred_cls = pred_cls.view((pred_cls.shape[0], pred_cls.shape[1], pred_cls.shape[2], self.bbox_num, -1)).sigmoid()

        pred_loc = self.pre_loc_layer(x).permute(0, 2, 3, 1)
        pred_loc = pred_loc.view((pred_cls.shape[0], pred_cls.shape[1], pred_cls.shape[2], self.bbox_num, -1))
        pred_loc[:, :, :, :, 0:2] = pred_loc[:, :, :, :, 0:2].sigmoid()

        pred_conf = self.pre_conf_layer(x).permute(0, 2, 3, 1)
        pred_conf = pred_conf.view(
            (pred_cls.shape[0], pred_cls.shape[1], pred_cls.shape[2], self.bbox_num, -1)).sigmoid()

        pred = (pred_cls, pred_conf, pred_loc)

        return pred

class pred_module_v3(nn.Module):
    def __init__(self,inplance,plance,cls_num,bbox_num):
        super(pred_module_v3,self).__init__()

        self.cls_num = cls_num
        self.bbox_num = bbox_num
        self.extra_layer = conv_block(inplance, plance, 3,stride=1,pad=1)
        self.detect_layer = nn.Conv2d(plance, (self.cls_num + 5) * self.bbox_num, 1, stride=1, padding=0, bias=True)

    def forward(self, x):

        feat_size = x.shape[-1]
        B = x.shape[0]

        x = self.extra_layer(x)
        prediction = self.detect_layer(x)
        prediction = prediction.view(B, (5+self.cls_num) * self.bbox_num, feat_size * feat_size)
        prediction = prediction.transpose(1, 2).contiguous()
        prediction = prediction.view(B, feat_size , feat_size , self.bbox_num, 5+self.cls_num)

        pred_cls = prediction[:,:,:,:,5:]#.sigmoid()
        pred_loc = prediction[:,:,:,:,0:4]
        pred_loc[:,:,:,:,0:2] = pred_loc[:,:,:,:,0:2].sigmoid()

        pred_conf = prediction[:,:,:,:,4].view(B, feat_size , feat_size , self.bbox_num, -1).sigmoid()
        pred = (pred_cls,pred_conf,pred_loc)

        return pred


class conv_sets(nn.Module):
    def __init__(self, inplance, plance, outplance):
        super(conv_sets, self).__init__()

        self.conv1 = conv_block(inplance, outplance, 1, stride=1, pad=0)
        self.conv2 = conv_block(outplance, plance, 3, stride=1, pad=1)
        self.conv3 = conv_block(plance, outplance, 1, stride=1, pad=0)
        self.conv4 = conv_block(outplance, plance, 3, stride=1, pad=1)
        self.conv5 = conv_block(plance, outplance, 1, stride=1, pad=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class up_sample(nn.Module):

    def __init__(self, inplance, outplance):
        super(up_sample, self).__init__()
        self.conv1 = conv_block(inplance, outplance, 1, stride=1, pad=0)

    def forward(self, x):
        out = self.conv1(x)

        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')

        return out







