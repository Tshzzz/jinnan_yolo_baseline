# 津南数字制造算法挑战赛YOLO Baseline
基于YOLOv2算法的实现。在VOC数据集上大概77mAP。
和原版的区别是使用了DarkNet53网络。能够直接在津南的数据集上训练网络。

## 结果:
训练了190个epoch的结果,感觉继续训练下去loss还能下降...

| Model             |  Ap.        |
| ------------------| ----------- |
| Test Online       |  0.3319     |
| Test Offline      |  0.363      |
详细的本地结果:
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

## 特点:
- *代码简单，容易上手，快速入门，能够方便的自定义自己想要的东西。*
- *训练、测试速度快。1080Ti在(608,608)的尺度下训练津南的数据集大概42秒一个epoch。*
- *实现了一些小工具，包括划分验证集、kmean聚类生成anchor的长宽、统计数据集的像素均值*

## 依赖项:
- *Python3*
- *Pytorch 0.4 或者更高*
- *cv2*
- *coco API*

## 使用方法:
下载预训练模型:
```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

新建目录datasets，把比赛数据文件解压在datasets目录下。或者你在config.py里去设置自己的目录。
```bash
python tools/split_datasets.py
python train.py
```
线下评估模型:
```bash
python valid.py
```
因为主办方给的数据中，area这个字段是空的，所以我把coco API里的cocoeval.py中的251行左右的代码注释掉了
```
for g in gt:
   #print(g['area'])
   #if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
   #    g['_ignore'] = 1
   #else:
        g['_ignore'] = 0
```

## 提升性能的一些方向:
### 多尺度训练:
由于刚开始写代码的时候没有考虑过多尺度训练,所以一直没有加入多尺度训练。

### 用更大的图作为网络的输入
我是在(608,608)的尺度上训练的网络,感觉可以用更大的图像输入来提升性能。  

### 换更好的前端网络
我之前实现的YOLOv2使用DarkNet19效果比DarkNet53差很多。可以考虑使用更深更大的ResNet101等等。
前端网络对检测的影响很大。

### 使用YOLOv3
我一直觉得YOLOv3是YOLOv2 * 3。YOLOv3用了3层不用的特征层，分别预测不同大小的物体。其实我这个代码
就是从YOLOv3退化来的。。。因为我自己写的YOLOv3，他训练不下去。而且最近要开始准备写论文找工作，就暂时
将他搁置了。

## 一些建议:
YOLO这类算法不适合这个比赛。除非比赛要考虑时间和效率。比赛初期，就是对每幅图像进行分类，所以我就试着用YOLO
去做这个比赛。但是后面改了规则，我也就弃坑了。如果你们希望在这个比赛取得好的成绩，还是用二阶检测方法吧。
推荐几个不错的github。但是，YOLO的论文中提到，YOLO和faster rcnn系列算法做融合，能够提升不少的分数。
我觉得YOLO也是可以考虑的辅助的提分手段，虽然不是得分主力。
- **[mmdetection](https://github.com/open-mmlab/mmdetection)**
- **[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)**
- **[Detectron](https://github.com/facebookresearch/Detectron)**  


