#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""
import torch
import torch.nn as nn
import numpy as np
from src.layers import conv_block,residual_block
from src.utils import load_conv_bn

class darknet53(nn.Module):

    def __init__(self, in_planes=3):
        super(darknet53, self).__init__()

        self.conv1 = conv_block(in_planes, 32, 3)
        self.conv2 = conv_block(32, 64, 3, stride=2, pad=1)

        self.block1 = residual_block(64, 64)
        self.conv3 = conv_block(64, 128, 3, stride=2, pad=1)

        self.block2 = nn.ModuleList()
        self.block2.append(residual_block(128, 128))
        self.block2.append(residual_block(128, 128))

        self.conv4 = conv_block(128, 256, 3, stride=2, pad=1)

        self.block3 = nn.ModuleList()
        for i in range(8):
            self.block3.append(residual_block(256, 256))

        self.conv5 = conv_block(256, 512, 3, stride=2, pad=1)

        self.block4 = nn.ModuleList()
        for i in range(8):
            self.block4.append(residual_block(512, 512))

        self.conv6 = conv_block(512, 1024, 3, stride=2, pad=1)

        self.block5 = nn.ModuleList()
        for i in range(4):
            self.block5.append(residual_block(1024, 1024))

    def load_part(self, buf, start, part):
        for idx, m in enumerate(part.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf, start, conv, bn)
        return start

    def load_weight(self, weight_file):

        if weight_file is not None:
            print("Load pretrained models !")

            fp = open(weight_file, 'rb')
            header = np.fromfile(fp, count=5, dtype=np.int32)
            header = torch.from_numpy(header)
            buf = np.fromfile(fp, dtype=np.float32)

            start = 0
            start = self.load_part(buf, start, self.conv1)
            start = self.load_part(buf, start, self.conv2)
            start = self.load_part(buf, start, self.block1)
            start = self.load_part(buf, start, self.conv3)
            start = self.load_part(buf, start, self.block2)
            start = self.load_part(buf, start, self.conv4)
            start = self.load_part(buf, start, self.block3)
            start = self.load_part(buf, start, self.conv5)
            start = self.load_part(buf, start, self.block4)
            start = self.load_part(buf, start, self.conv6)
            start = self.load_part(buf, start, self.block5)

            print(start, buf.shape[0])

    def forward(self, x):
        detect_feat = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.block1(out)
        out = self.conv3(out)

        for modu in self.block2:
            out = modu(out)

        out = self.conv4(out)
        for modu in self.block3:
            out = modu(out)
        detect_feat.append(out)

        out = self.conv5(out)
        for modu in self.block4:
            out = modu(out)
        detect_feat.append(out)

        out = self.conv6(out)
        for modu in self.block5:
            out = modu(out)
        detect_feat.append(out)
        return detect_feat

