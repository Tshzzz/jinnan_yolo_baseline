#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: tshzzz

这部分参考了 maskrcnn_benchmark的代码
https://github.com/facebookresearch/maskrcnn-benchmark

"""
import torch
import torchvision
from src.bounding_box import BoxList

from src.data_augment import load_data_detection
import numpy as np


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, train, transforms=None,box_encoder=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.train = train
        self.box_encoder = box_encoder


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        src_img_size = img.size
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        ##注意类别！！！！！！！！！！！！！
        classes = [self.json_category_id_to_contiguous_id[c]-1 for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        box_coord = target.bbox
        box_cls = target.get_field('labels').view(-1, 1).float()

        bbox_info = torch.cat((box_coord,box_cls),1)

        img, bbox = load_data_detection(img, bbox_info.numpy(),
                                        self.transforms.transforms[0].size,
                                        self.train)


        if self.box_encoder is not None:
            gt = self.box_encoder(bbox)
        else:
            gt = np.zeros((50, 5), dtype=np.float32)
            gt[:len(bbox), :] = bbox
            gt = torch.from_numpy(gt).float()

        if self.transforms is not None:
            img = self.transforms(img)

        if self.train:
            return img, gt
        else:
            return img,gt,anno[0]['image_id'],src_img_size

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data