#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""
import numpy as np
import torch
from src.utils import py_cpu_nms,bbox_iou


def gen_yolo_box(featmaps,anchor_wh):

    #featmaps = [b,c,h,w]

    output = np.zeros((featmaps[0], featmaps[1], len(anchor_wh), 4))


    for i in range(featmaps[0]):
        for j in range(featmaps[1]):
            cx = (j ) #/ featmaps[0]
            cy = (i ) #/ featmaps[1]

            for k,(w,h) in enumerate(anchor_wh):
                output[i,j,k,:] = [cx, cy, w , h ]

    return output


class yolo_box_encoder(object):

    def __init__(self,anchor,class_num,featmap_size):
        # anchor B,13,13,5
        self.anchor = gen_yolo_box(featmap_size,anchor)
        self.class_num = class_num
        self.featmap_size = featmap_size
        self.boxes_num = len(anchor)

    def __call__(self,bs):
        #global tw_a,tw_b

        # b,c,h,w -> b,c,x,y

        bb_class = np.zeros((self.featmap_size[0],self.featmap_size[1],self.boxes_num,self.class_num))
        bb_boxes = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, 4))
        bb_conf = np.zeros((self.featmap_size[0],self.featmap_size[1],self.boxes_num,1))

        for i in range(bs.shape[0]):
            local_x = int(min(0.999, max(0, bs[i, 0] + bs[i, 2] / 2)) * (self.featmap_size[0]) )
            local_y = int(min(0.999, max(0, bs[i, 1] + bs[i, 3] / 2)) * (self.featmap_size[1]) )

            ious = []
            for k in range(self.boxes_num):
                temp_x,temp_y,temp_w,temp_h = self.anchor[local_y,local_x,k,:]
                temp_w = temp_w / self.featmap_size[0]
                temp_h = temp_h / self.featmap_size[1]

                anchor_ = np.array([[0,0,temp_w,temp_h]])
                gt = np.array([[0,0,bs[i,2],bs[i,3]]])
                ious.append(bbox_iou(anchor_, gt)[0])


            selected_ = np.argsort(ious)[::-1]

            for kk,selected_anchor in enumerate(selected_):


                if  bb_conf[local_y,local_x, selected_anchor,0] == 0 and bs[i,2]>0.02 and bs[i,3]>0.02 :

                    tx =  (bs[i, 0] + bs[i, 2] / 2) * self.featmap_size[0] \
                                  - (self.anchor[local_y,local_x,selected_anchor,0] )

                    ty =  (bs[i, 1] + bs[i, 3] / 2) * self.featmap_size[1] \
                                  - (self.anchor[local_y,local_x,selected_anchor,1] )

                    tw = np.log(max(0.01,bs[i,2]* self.featmap_size[0] / self.anchor[local_y,local_x,selected_anchor,2]) )
                    th = np.log(max(0.01,bs[i,3]* self.featmap_size[1] / self.anchor[local_y,local_x,selected_anchor,3]) )

                    bb_boxes[local_y,local_x, selected_anchor,:] = np.array([tx,ty,tw,th])

                    #考虑背景 使用 softmax
                    #bb_class[local_x, local_y, selected_anchor,:] = 0
                    bb_class[local_y, local_x, selected_anchor, int(bs[i, 4])] = 1
                    bb_conf[local_y,local_x, selected_anchor,0] = 1
                    break


        target = (bb_class,bb_conf,bb_boxes)

        return target

class yolo_box_decoder(object):

    def __init__(self, anchor, class_num,featmap_size,conf=0.05,nms_thresh=0.5):
        self.class_num = class_num#
        self.anchor = torch.from_numpy(gen_yolo_box(featmap_size, anchor)).float()
        self.boxes_num = len(anchor)
        self.featmap_size = featmap_size
        self.conf_thresh = conf
        self.nms_thresh = nms_thresh
    def __call__(self, pred):
        boxes = []
        classes = []
        pred_cls, pred_conf, pred_bboxes = pred
        featmap_size = torch.Tensor([pred_cls.shape[1], pred_cls.shape[2]])


        pred_cls = pred_cls.cpu().float().view(-1,self.class_num)
        pred_conf = pred_conf.cpu().float().view(-1,1)#.sigmoid()
        pred_bboxes = pred_bboxes.cpu().float().view(-1,4)
        anchor = self.anchor.repeat(1, 1, 1, 1, 1).cpu().view(-1,4)

        #找最anchor中置信度最高的

        pred_mask = (pred_conf>self.conf_thresh).view(-1)

        pred_bboxes = pred_bboxes[pred_mask]
        pred_conf = pred_conf[pred_mask]
        pred_cls = pred_cls[pred_mask]
        anchor = anchor[pred_mask]

        for cls in range(self.class_num):
            cls_prob = pred_cls[:, cls].float() * pred_conf[:, 0]

            mask_a = cls_prob.gt(self.conf_thresh)

            bbox = pred_bboxes[mask_a]
            anchor_ = anchor[mask_a]
            cls_prob = cls_prob[mask_a]

            if bbox.shape[0] > 0:

                bbox[:, 2:4] = torch.exp(bbox[:, 2:4]) * anchor_[:, 2:4] / (featmap_size[0:2])
                bbox[:, 0:2] = (bbox[:, 0:2] + (anchor_[:, 0:2]))/ (featmap_size[0:2]) - bbox[:, 2:4] / 2


                pre_cls_box = bbox.data.numpy()
                pre_cls_score = cls_prob.data.view(-1).numpy()

                keep = py_cpu_nms(pre_cls_box, pre_cls_score, thresh=self.nms_thresh)
                for conf_keep, loc_keep in zip(pre_cls_score[keep], pre_cls_box[keep]):
                    boxes.append(loc_keep)
                    classes.append([cls, conf_keep])

        boxes = np.array(boxes)
        classes = np.array(classes)

        return boxes,classes

