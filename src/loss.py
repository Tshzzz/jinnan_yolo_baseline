#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz
"""

import torch
import torch.nn as nn
import numpy as np
from src.box_coder import gen_yolo_box

class yolov2_loss(nn.Module):
    def __init__(self, anchor,featmap_size, l_coord=3, object_scale=5, noobject_scale=1):
        super(yolov2_loss, self).__init__()

        self.l_coord = l_coord
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.cls_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.anchor = torch.from_numpy(gen_yolo_box(featmap_size, anchor)).float()




    def bbox_ious(self, boxes1, boxes2):
        """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.
        Args:
            boxes1 (torch.Tensor): List of bounding boxes
            boxes2 (torch.Tensor): List of bounding boxes
        Note:
            List format: [[xc, yc, w, h],...]
        """
        b1_len = boxes1.size(0)
        b2_len = boxes2.size(0)

        b1x1, b1y1 = (boxes1[:, :2]).split(1, 1)
        b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 1)).split(1, 1)
        b2x1, b2y1 = (boxes2[:, :2]).split(1, 1)
        b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 1)).split(1, 1)

        dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
        dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        unions = (areas1 + areas2.t()) - intersections

        return intersections / unions

    def forward(self, pred, target):

        label_cls, label_conf, label_bboxes = target
        pred_cls, pred_conf, pred_bboxes = pred

        featmap_size = [pred_bboxes.shape[1],pred_bboxes.shape[2]]
        #print('featmap_size{}',format(featmap_size))

        device = pred_bboxes.device
        label_cls = label_cls.type_as(pred_cls).to(device)
        label_conf = label_conf.type_as(pred_conf).to(device)
        label_bboxes = label_bboxes.type_as(pred_bboxes).to(device)

        boxes_num = label_bboxes.shape[3]
        featmap_size = [label_bboxes.shape[1], label_bboxes.shape[2]]
        batch_size = label_bboxes.shape[0]

        anchor = self.anchor.repeat(batch_size, 1, 1, 1, 1).type_as(pred_bboxes).to(device)

        # 获取非背景的正样本
        pos_mask = label_conf[:, :, :, :, 0] > 0
        num_pos = pos_mask.shape[-1]

        t_pos_cls = label_cls[pos_mask]
        p_pos_cls = pred_cls[pos_mask]

        t_pos_box = label_bboxes[pos_mask]
        p_pos_box = pred_bboxes[pos_mask]

        t_pos_conf = label_conf[pos_mask]
        p_pos_conf = pred_conf[pos_mask]

        # 计算 iou
        delta_gt_box = label_bboxes[pos_mask]
        delta_pred_box = pred_bboxes[pos_mask]
        a_pos_ = anchor[pos_mask]

        gt_box = torch.zeros_like(a_pos_).type_as(pred_bboxes).to(device)
        gt_box[:, 2] = torch.exp(delta_gt_box[:, 2]) * a_pos_[:, 2] / (featmap_size[0])
        gt_box[:, 3] = torch.exp(delta_gt_box[:, 3]) * a_pos_[:, 3] / (featmap_size[1])
        gt_box[:, 0] = (delta_gt_box[:, 0] + a_pos_[:, 0]) / (featmap_size[0]) - gt_box[:, 2] / 2
        gt_box[:, 1] = (delta_gt_box[:, 1] + a_pos_[:, 1]) / (featmap_size[1]) - gt_box[:, 3] / 2

        anchor = anchor.view(-1, 4)
        pred_box = torch.zeros_like(anchor).type_as(pred_bboxes).to(device)
        pred_bboxes = pred_bboxes.contiguous().view(-1, 4)
        pred_box[:, 2] = torch.exp(pred_bboxes[:, 2]) * anchor[:, 2] / (featmap_size[0])
        pred_box[:, 3] = torch.exp(pred_bboxes[:, 3]) * anchor[:, 3] / (featmap_size[1])
        pred_box[:, 0] = (pred_bboxes[:, 0] + anchor[:, 0]) / (featmap_size[0]) - pred_box[:, 2] / 2
        pred_box[:, 1] = (pred_bboxes[:, 1] + anchor[:, 1]) / (featmap_size[1]) - pred_box[:, 3] / 2

        ious_ = self.bbox_ious(gt_box, pred_box).permute(1, 0).contiguous()

        ious_mask = label_conf.view(-1) > 0

        gt_num = gt_box.shape[0]

        pos_ious_ = ious_[ious_mask].view(-1, gt_num)

        pos_ious_mask = torch.zeros_like(pos_ious_).byte().to(device)
        for i in range(pos_ious_.shape[0]):
            pos_ious_mask[i, i] = 1
        pos_ious_ = pos_ious_[pos_ious_mask].view(-1)


        if gt_num == 0:
            gt_num = 1
        recall_50 = (pos_ious_ > 0.5).sum().float() / float(gt_num)
        recall_75 = (pos_ious_ > 0.75).sum().float() / float(gt_num)
        loss_pos_conf = self.mse_loss(p_pos_conf.view(-1), pos_ious_.detach().view(-1)) * self.object_scale / batch_size

        # 获取负样本类别
        ious_ = ious_.max(1)[0]
        ious_mask = label_conf.view(-1) > 0
        ious_[ious_mask] = 1

        neg_mask = (ious_ < 0.6)
        p_neg_conf = pred_conf.view(-1)[neg_mask]
        t_neg_conf = label_conf.view(-1)[neg_mask]


        loss_neg_conf = self.mse_loss(p_neg_conf, t_neg_conf) * self.noobject_scale / batch_size
        loss_cls = self.cls_loss(p_pos_cls, t_pos_cls.argmax(-1)) / batch_size
        loss_x = self.mse_loss(p_pos_box[:, 0], t_pos_box[:, 0]) * self.l_coord / batch_size
        loss_y = self.mse_loss(p_pos_box[:, 1], t_pos_box[:, 1]) * self.l_coord / batch_size
        loss_w = self.smooth_l1_loss(p_pos_box[:, 2], t_pos_box[:, 2]) * self.l_coord / batch_size
        loss_h = self.smooth_l1_loss(p_pos_box[:, 3], t_pos_box[:, 3]) * self.l_coord / batch_size

        loss_all = loss_x + loss_y + loss_w + loss_h + loss_cls + loss_neg_conf + loss_pos_conf

        loss_info = {
            'loss_all': loss_all.data.cpu(),
            'loss_x': loss_x.data.cpu(),
            'loss_y': loss_y.data.cpu(),
            'loss_w': loss_w.data.cpu(),
            'loss_h': loss_h.data.cpu(),
            'loss_pos_conf': loss_pos_conf.data.cpu(),
            'loss_neg_conf': loss_neg_conf.data.cpu(),
            'loss_cls': loss_cls.data.cpu(),
            'mean_iou': torch.mean(pos_ious_).data.cpu(),
            'recall_50': recall_50.data.cpu(),
            'recall_75': recall_75.data.cpu(),
        }

        return loss_all, loss_info


