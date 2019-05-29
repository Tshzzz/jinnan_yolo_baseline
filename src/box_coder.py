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


def gen_yolo_anchor_tensor(featmaps,anchor_wh):
    assert featmaps[0] == featmaps[1]

    box_num = len(anchor_wh)
    feat_len = featmaps[0]

    anchor_wh = torch.Tensor(anchor_wh)

    x = torch.linspace(0,featmaps[0]-1,featmaps[0])
    x = x.repeat(feat_len).view(-1,feat_len).unsqueeze(2)

    y = torch.linspace(0,featmaps[0]-1,featmaps[0]).view(-1,1)
    y = y.repeat(1,feat_len).unsqueeze(2)

    cc = torch.cat((x,y),2).unsqueeze(2)
    cc = cc.repeat(1,1,box_num,1)
    anchor_wh = anchor_wh.unsqueeze(0).unsqueeze(0)
    anchor_wh = anchor_wh.repeat(feat_len,feat_len,1,1)
    dddd = torch.cat((cc,anchor_wh),-1)

    return dddd



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
        self.anchor_wh = anchor
        self.boxes_num = len(anchor)
        self.featmap_size = featmap_size
        self.conf_thresh = conf
        self.nms_thresh = nms_thresh
    def __call__(self, pred):
        boxes = []
        classes = []
        pred_cls, pred_conf, pred_bboxes = pred
        featmap_size = torch.Tensor([pred_cls.shape[1], pred_cls.shape[2]])

        #anchor = gen_yolo_anchor_tensor([pred_cls.shape[1], pred_cls.shape[2]],self.anchor_wh)
        anchor = self.anchor.repeat(1, 1, 1, 1, 1).cpu().view(-1,4)

        pred_cls = pred_cls.cpu().float().view(-1,self.class_num)
        pred_conf = pred_conf.cpu().float().view(-1,1)
        pred_bboxes = pred_bboxes.cpu().float().view(-1,4)

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


class single_decoder(object):

    def __init__(self, anchor, class_num, featmap_size, conf=0.01):
        self.class_num = class_num
        self.anchor = torch.from_numpy(gen_yolo_box(featmap_size, anchor)).float()
        self.boxes_num = len(anchor)
        self.featmap_size = featmap_size
        self.conf_thresh = conf

    def __call__(self, pred):
        pred_cls, pred_conf, pred_bboxes = pred
        pred_cls = torch.nn.functional.softmax(pred_cls.float(), dim=-1)
        featmap_size = torch.Tensor([pred_cls.shape[1], pred_cls.shape[2]])

        pred_cls = pred_cls.cpu().float().view(-1, self.class_num)
        pred_conf = pred_conf.cpu().float().view(-1, 1)
        pred_bboxes = pred_bboxes.cpu().float().view(-1, 4)


        anchor = self.anchor.repeat(1, 1, 1, 1, 1).cpu().view(-1, 4)

        # 找最anchor中置信度最高的
        pred_mask = (pred_conf > self.conf_thresh).view(-1)
        pred_bboxes = pred_bboxes[pred_mask]
        pred_conf = pred_conf[pred_mask]
        pred_cls = pred_cls[pred_mask]

        anchor = anchor[pred_mask]

        pred_bboxes[:, 2:4] = torch.exp(pred_bboxes[:, 2:4]) * anchor[:, 2:4] / (featmap_size[0:2])
        pred_bboxes[:, 0:2] = (pred_bboxes[:, 0:2] + (anchor[:, 0:2]))/ (featmap_size[0:2]) - pred_bboxes[:, 2:4] / 2


        return pred_cls, pred_conf, pred_bboxes


class group_decoder(object):
    def __init__(self, anchor, class_num, featmap_size, conf=0.01, nms_thresh=0.5):

        self.decoder = []
        for i in range(len(anchor)):
            self.decoder.append(single_decoder(anchor[i], class_num, featmap_size[i], conf))

        self.class_num = class_num
        self.conf_thresh = conf
        self.nms_thresh = nms_thresh

    def __call__(self, preds):


        pred_cls = []
        pred_conf = []
        pred_bboxes = []
        for pred,decoder in zip(preds,self.decoder):
            cls,conf,bbox = decoder(pred)
            pred_cls.append(cls)
            pred_conf.append(conf)
            pred_bboxes.append(bbox)

        pred_cls = torch.cat([cls for cls in pred_cls])
        pred_bboxes = torch.cat([bbox for bbox in pred_bboxes])
        pred_conf = torch.cat([conf for conf in pred_conf])

        boxes = []
        classes = []

        for cls in range(self.class_num):
            cls_prob = pred_cls[:, cls].float() * pred_conf[:, 0]

            mask_a = cls_prob.gt(self.conf_thresh)

            bbox = pred_bboxes[mask_a]
            cls_prob = cls_prob[mask_a]
            iou_prob = pred_conf[mask_a]

            if bbox.shape[0] > 0:

                pre_cls_box = bbox.data.numpy()
                pre_cls_score = cls_prob.data.view(-1).numpy()
                iou_prob = iou_prob.data.view(-1).numpy()

                keep = py_cpu_nms(pre_cls_box, pre_cls_score, thresh=self.nms_thresh)
                for conf_keep, loc_keep in zip(pre_cls_score[keep], pre_cls_box[keep]):
                    boxes.append(loc_keep)
                    classes.append([cls, conf_keep])

        boxes = np.array(boxes)
        classes = np.array(classes)

        return boxes, classes





class single_encoder(object):

    def __init__(self, anchor, class_num, featmap_size):
        # anchor B,13,13,5
        self.anchor = gen_yolo_box(featmap_size, anchor)
        self.class_num = class_num
        self.featmap_size = featmap_size
        self.boxes_num = len(anchor)

        self.bb_class = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, self.class_num))
        self.bb_boxes = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, 4))
        self.bb_conf = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, 1))

    def get_target(self):

        return (self.bb_class,self.bb_conf,self.bb_boxes)

    def clean_target(self):
        self.bb_class = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, self.class_num))
        self.bb_boxes = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, 4))
        self.bb_conf = np.zeros((self.featmap_size[0], self.featmap_size[1], self.boxes_num, 1))
        return

    def __call__(self, bs):

        for i in range(bs.shape[0]):
            local_x = int(min(0.999, max(0, bs[i, 0] + bs[i, 2] / 2)) * (self.featmap_size[0]) )
            local_y = int(min(0.999, max(0, bs[i, 1] + bs[i, 3] / 2)) * (self.featmap_size[1]) )
            ious = []
            for k in range(self.boxes_num):
                temp_x, temp_y, temp_w, temp_h = self.anchor[local_y, local_x, k, :]
                temp_w = temp_w / self.featmap_size[0]
                temp_h = temp_h / self.featmap_size[1]
                anchor_ = np.array([[0, 0, temp_w, temp_h]])
                gt = np.array([[0, 0, bs[i,2], bs[i,3]]])
                ious.append(bbox_iou(anchor_, gt)[0])

            selected_ = np.argsort(ious)[::-1]

            for kk, selected_anchor in enumerate(selected_):

                if self.bb_conf[local_y, local_x, selected_anchor, 0] == 0 and bs[i,2] > 0.02 and bs[i,3] > 0.02:
                    tx = (bs[i,0] + bs[i,2] / 2) * self.featmap_size[0] - (self.anchor[local_y, local_x, selected_anchor, 0])
                    ty = (bs[i,1] + bs[i,3] / 2) * self.featmap_size[1] - (self.anchor[local_y, local_x, selected_anchor, 1])
                    tw = np.log(max(0.01, bs[i,2] * self.featmap_size[0] / self.anchor[local_y, local_x, selected_anchor, 2]))
                    th = np.log(max(0.01, bs[i,3] * self.featmap_size[1] / self.anchor[local_y, local_x, selected_anchor, 3]))

                    self.bb_boxes[local_y, local_x, selected_anchor, :] = np.array([tx, ty, tw, th])
                    # 考虑背景 使用 softmax
                    self.bb_class[local_y, local_x, selected_anchor, int(bs[i,4])] = 1
                    self.bb_conf[local_y, local_x, selected_anchor, 0] = 1
                    break
        return


class group_encoder(object):

    def __init__(self, anchor, class_num, featmap_size):
        print(featmap_size)
        # anchor B,13,13,5
        self.anchor = anchor
        self.class_num = class_num
        self.featmap_size = featmap_size
        self.boxes_num = len(anchor)
        self.featmap_num = len(featmap_size)

        self.encoder = []
        for i in range(len(anchor)):
            self.encoder.append(single_encoder(anchor[i], class_num, featmap_size[i]))

    def __call__(self, bs):
        # b,c,h,w -> b,c,x,y
        #for i in range(bs.shape[0]):
        for encoder in self.encoder:
            encoder(bs)

        target = []
        for encoder in self.encoder:
            target.append(encoder.get_target())
        for encoder in self.encoder:
            encoder.clean_target()

        return target


if __name__ == "__main__":
    featmaps=[26,26]
    anchor_wh = [[2.8523827, 2.4452496],
                 [1.3892268, 1.8958333],
                 [1.6490009, 0.9559666],
                 [0.7680278, 1.3883946],
                 [0.5605738, 0.6916781]]

    a = gen_yolo_box(featmaps, anchor_wh)
    a = torch.Tensor(a)

    b = gen_yolo_anchor_tensor(featmaps, anchor_wh)

    '''
    anchor_wh = torch.Tensor(anchor_wh)
    x = torch.linspace(0,featmaps[0]-1,featmaps[0]).view(-1,1)
    x = x.repeat(1,19).unsqueeze(2)
    y = torch.linspace(0,featmaps[0]-1,featmaps[0])
    y = y.repeat(19).view(-1,19).unsqueeze(2)
    cc = torch.cat((x,y),2).unsqueeze(2)
    cc = cc.repeat(1,1,5,1)
    anchor_wh = anchor_wh.unsqueeze(0).unsqueeze(0)
    anchor_wh = anchor_wh.repeat(19,19,1,1)
    dddd = torch.cat((cc,anchor_wh),-1)
    '''

    print(a == b)
