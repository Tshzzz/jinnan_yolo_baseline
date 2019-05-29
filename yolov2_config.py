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

YOLO['epochs'] = 80
YOLO['epochs_start'] = 0
YOLO['steps'] = [50,60]


YOLO['batch_size'] = 10
YOLO['start_lr'] = 1e-3
YOLO['image_size'] = [608, 608]
YOLO['featmap_size'] = [[19, 19]]

YOLO['anchor'] = [[[2.8523827,2.4452496 ],
                [1.3892268,1.8958333 ],
                [1.6490009,0.95596665],
                [0.7680278,1.3883946 ],
                [0.5605738,0.69167805]]]
