classes = ["aeroplane", "bicycle", "bird", "boat", "bottle"]

datasets_path = './datasets/jinnan2_round1_train_20190305/'

anchor_wh = [[2.8523827,2.4452496 ],
             [1.3892268,1.8958333 ],
             [1.6490009,0.95596665],
             [0.7680278,1.3883946 ],
             [0.5605738,0.69167805]]

batch_size = 10
start_lr = 0.001

save_dir = './608/'

YOLO = {
    'object_scale': 5,
    'class_scale': 1,
    'coord_scale': 1,
    'pretrain_model': './darknet53.conv.74',
    'image_size': [608, 608],
    'featmap_size': [19, 19],
    'class_num': len(classes),
    'anchor': anchor_wh
}