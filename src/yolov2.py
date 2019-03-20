#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""
import torch
import torch.nn as nn
from src.layers import conv_sets,pred_module
from src.box_coder import yolo_box_decoder,gen_yolo_box
from src.loss import yolo_loss
from src.darknet53 import darknet53


class YOLOv2(nn.Module):

    def __init__(self, basenet, anchor, do_detect,featmap_size ,cls_num=20):
        super(YOLOv2, self).__init__()
        self.cls_num = cls_num
        self.feature = basenet
        self.bbox_num = len(anchor)
        self.convsets1 = conv_sets(1024, 1024, 512)
        self.detector = pred_module(512, self.cls_num, self.bbox_num)


        self.detect = do_detect


        if self.detect:
            self.decoder = yolo_box_decoder(anchor, cls_num, featmap_size)
        else:
            self.loss = yolo_loss(anchor, featmap_size)

    def forward(self, x , target =None):

        B = x.size(0)
        feats = self.feature(x)
        layer1 = self.convsets1(feats[2])

        pred = self.detector(layer1)

        if self.detect:
            pred_cls, pred_conf, pred_bboxes = pred
            print(pred_cls.shape)
            pred_cls = torch.nn.functional.softmax(pred_cls.float(), dim=-1)
            print(pred_cls.shape)
            pred = self.decoder((pred_cls, pred_conf, pred_bboxes))
            return pred
        else:
            loss,loss_info = self.loss(pred,target)
            return loss,loss_info

def build_yolov2(cls_num, anchor, featmap_size,do_detect =True,pretrained=None):

    basenet = darknet53()
    basenet.load_weight(pretrained)
    net = YOLOv2(basenet, anchor, do_detect,featmap_size,cls_num)

    return net


if __name__ == '__main__':

    net = build_yolov2(5, 5, 'darknet53.conv.74')

    data = torch.randn(1, 3, 608, 608)



