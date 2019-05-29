#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""
import torch
import torch.nn as nn
from src.layers import conv_sets,pred_module
from src.box_coder import yolo_box_decoder,gen_yolo_box,group_decoder
from src.loss import yolo_loss,yolov3_loss
from src.darknet53 import darknet53


class YOLOv2(nn.Module):

    def __init__(self, basenet, anchor, do_detect,featmap_size ,cls_num=20):
        super(YOLOv2, self).__init__()
        self.cls_num = cls_num
        self.feature = basenet
        self.bbox_num = len(anchor[0])
        self.convsets1 = conv_sets(1024, 1024, 512)
        self.detector = pred_module(512, self.cls_num, self.bbox_num)


        self.do_detect = do_detect

        if self.do_detect:
            self.decoder = group_decoder(anchor, cls_num, featmap_size)
        else:
            self.loss = []
            for i in range(len(anchor)):
                #print(len(anchor))
                self.loss.append(yolov3_loss(anchor[i],featmap_size[i]))
        '''
        if self.detect:
            self.decoder = yolo_box_decoder(anchor, cls_num, featmap_size)
        else:
            self.loss = yolo_loss(anchor, featmap_size)
        '''
    def forward(self, x , target =None):

        B = x.size(0)
        feats = self.feature(x)
        layer1 = self.convsets1(feats[2])

        pred = [self.detector(layer1)]

        if self.do_detect:
            pred_cls, pred_conf, pred_bboxes = pred[0]
            pred_cls = torch.nn.functional.softmax(pred_cls.float(), dim=-1)
            pred = self.decoder(pred)
            return pred
        else:

            loss_info = {}
            loss = 0
            for i in range(len(pred)):
                loss_temp,loss_info_temp = self.loss[i](pred[i],target[i])
                loss += loss_temp
                if i == 0:
                    loss_info.update(loss_info_temp)
                else:
                    for k, v in loss_info_temp.items():
                        loss_info[k] += v


            loss_info['mean_iou'] /= len(pred)
            loss_info['recall_50'] /= len(pred)
            loss_info['recall_75'] /= len(pred)

            return loss,loss_info
            #loss,loss_info = self.loss(pred,target)
            #return loss,loss_info

def build_yolov2(cls_num, anchor, featmap_size,do_detect =True,pretrained=None):

    basenet = darknet53()
    basenet.load_weight(pretrained)
    net = YOLOv2(basenet, anchor, do_detect,featmap_size,cls_num)

    return net


if __name__ == '__main__':

    net = build_yolov2(5, 5, 'darknet53.conv.74')

    data = torch.randn(1, 3, 608, 608)



