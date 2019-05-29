# [津南数字制造算法挑战赛](https://tianchi.aliyun.com/competition/entrance/231703/introduction)YOLO Baseline
This YOLO V2 train on VOC datasets get more than 77mAp

## Result:
the result training on jinnan datasets.

| Model             |  Ap.        |
| ------------------| ----------- |
| Test Online       |  0.3319     |
| Test Offline      |  0.363      |


Local Val Result:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.720
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.422
```

## Dependence:
- *Python3*
- *Pytorch 1.0 or higher*
- *cv2*
- *coco API*

## Training:
download the pretrain model:
```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

configure the config.py to set the dataset paths
```bash
python tools/split_datasets.py
python train_yolov2.py
```
valid the model:
```bash
python valid.py
```

## Problem:
The yolov3 is on the bad performs.They are still some bugs.




