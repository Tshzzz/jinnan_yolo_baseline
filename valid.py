#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:05:18 2018

@author: vl-tshzzz
"""

import torch
import config

from torch.utils import data

from src import COCODataset
from torchvision import transforms

import json

from src import build_yolov2,build_yolov3

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def test_result(model, dataset):

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=8)

    result = []
    for images, label, img_id, img_size in test_loader:

        height, width = int(img_size[1]), int(img_size[0])

        images = images.cuda()
        label = label.view(-1,5)
        label = label[label.sum(1)>0]

        gt = label[:,4]

        gt_dict = [0, 0, 0, 0, 0]
        for i in gt:
            gt_dict[int(i.numpy())] = 1

        pred_boxes, pred_conf = model(images)


        for j in range(len(pred_boxes)):
            x1 = round(pred_boxes[j,0]*width)
            y1 = round(pred_boxes[j,1]*height)
            w =  round(pred_boxes[j,2]*width)
            h =  round(pred_boxes[j,3]*height)

            dict = {
                'image_id': int(img_id.numpy()[0]),
                'category_id': int(pred_conf[j,0])+1,
                'bbox': [x1,y1,w,h],
                'score': float(pred_conf[j,1])
            }

            result.append(dict)

    with open("./coco_valid.json", 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False)

    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes('coco_valid.json')

    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return


if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Resize(config.YOLO['image_size']),
        transforms.ToTensor(),
    ])
    anchor_wh = config.YOLO['anchor']

    featmap_size = config.YOLO['featmap_size']


    model = build_yolov2(config.YOLO['class_num'], anchor_wh, featmap_size, do_detect = True)
    #model = build_yolov3(config.YOLO['class_num'], anchor_wh, featmap_size, do_detect = True)
    annFile = './datasets/jinnan2_round1_train_20190305/jinnan_round1_val.json'


    dataset = COCODataset(annFile,'{}restricted/'.format(config.datasets_path),True,False,transform,None)

    model.load_state_dict(torch.load('608/model_290.pkl'))
    model.cuda()
    model.eval()

    test_result(model, dataset)



