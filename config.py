import numpy as np

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle"]

datasets_path = './datasets/jinnan2_round1_train_20190305/'



anchor_yolov2 = [[2.8523827,2.4452496 ],
                [1.3892268,1.8958333 ],
                [1.6490009,0.95596665],
                [0.7680278,1.3883946 ],
                [0.5605738,0.69167805]]
epochs_start = 208
batch_size = 10
start_lr = 0.0001

save_dir = './608/'

YOLO = {
    'pretrain_model': './darknet53.conv.74',
    'image_size': [608, 608],
    'featmap_size': [19, 19],
    'class_num': len(classes),
    'anchor': anchor_yolov2
}
'''

anchor_big = np.array([[116, 90], [156, 198], [373, 326]]) / 32
anchor_medium = np.array([[30, 61], [62, 45], [59, 119]]) / 32
anchor_small = np.array([[10, 13], [16, 30], [33, 23]]) / 32
anchor = [anchor_big, anchor_medium, anchor_small]
'''

'''
img_size = 608
feat_size = img_size // 32

anchor_big = np.array([[116, 90], [156, 198], [373, 326]]) / 32
anchor_medium = np.array([[30, 61], [62, 45], [59, 119]]) / 16
anchor_small = np.array([[10, 13], [16, 30], [33, 23]]) / 8

anchor_big = np.array([[0.127, 0.158], [0.1574, 0.068], [0.0452, 0.085]]) * (feat_size)
anchor_medium = np.array([[0.0643, 0.189], [0.249, 0.184], [0.0217, 0.0628]]) * (2 * feat_size)
anchor_small = np.array([[0.0869, 0.0976], [0.077, 0.0485], [0.0461, 0.0282]]) * (4 * feat_size)
anchor_wh = [anchor_big, anchor_medium, anchor_small]
anchor = [anchor_big, anchor_medium, anchor_small]



feat = [[feat_size, feat_size], [feat_size * 2, feat_size * 2], [feat_size * 4, feat_size * 4]]

batch_size = 8
start_lr = 0.001

save_dir = './YOLOV3/'

YOLO = {
    'pretrain_model': './darknet53.conv.74',
    'image_size': [img_size, img_size],
    'featmap_size': feat,
    'class_num': len(classes),
    'anchor': anchor
}

'''
