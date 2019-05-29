import numpy as np

YOLO = dict()

# Datasets Parameter
YOLO['classes'] = ["aeroplane", "bicycle", "bird", "boat", "bottle"]
YOLO['datasets_path'] = '/home/tshzzz/disk_m2/jinnan_round2/datasets/round1_train/'
YOLO['anno_path'] = '/home/tshzzz/disk_m2/jinnan_round2/datasets/round1_train/jinnan_round1_train.json'
YOLO['val_path'] = '/home/tshzzz/disk_m2/jinnan_round2/datasets/round1_train/jinnan_round1_val.json'

YOLO['pretrain_model'] = './darknet53.conv.74'
YOLO['class_num'] = len(YOLO['classes'])


# Training Parameter
YOLO['save_dir'] = './608/'
YOLO['pretrain_model'] = './darknet53.conv.74'

YOLO['epochs'] = 100
YOLO['epochs_start'] = 1
YOLO['steps'] = [50,60]


YOLO['batch_size'] = 8
YOLO['start_lr'] = 1e-3
YOLO['image_size'] = [608, 608]


img_size = 608
feat_size = img_size // 32

anchor_big = np.array([[116, 90], [156, 198], [373, 326]]) / 32
anchor_medium = np.array([[30, 61], [62, 45], [59, 119]]) / 16
anchor_small = np.array([[10, 13], [16, 30], [33, 23]]) / 8
'''
anchor_big = np.array([[0.127, 0.158], [0.1574, 0.068], [0.0452, 0.085]]) * (feat_size)
anchor_medium = np.array([[0.0643, 0.189], [0.249, 0.184], [0.0217, 0.0628]]) * (2 * feat_size)
anchor_small = np.array([[0.0869, 0.0976], [0.077, 0.0485], [0.0461, 0.0282]]) * (4 * feat_size)
'''
anchor_wh = [anchor_big, anchor_medium, anchor_small]
anchor = [anchor_big, anchor_medium, anchor_small]

YOLO['featmap_size'] = [[feat_size, feat_size], [feat_size * 2, feat_size * 2], [feat_size * 4, feat_size * 4]]
YOLO['anchor'] = anchor

