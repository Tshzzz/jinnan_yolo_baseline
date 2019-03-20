from src import COCODataset
from src import yolo_box_encoder
from src import yolo_box_decoder
from src import group_decoder,group_encoder


from torchvision import transforms

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils import data
import json

if __name__=="__main__":
    import numpy as np

    transform = transforms.Compose([
        transforms.Resize([416, 416]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])
    ])

    annFile = '../datasets/jinnan2_round1_train_20190305/train_no_poly.json'

    '''
    anchor_wh = [[2.8523827,2.4452496 ],
             [1.3892268,1.8958333 ],
             [1.6490009,0.95596665],
             [0.7680278,1.3883946 ],
             [0.5605738,0.69167805]]

    featmap_size = [13,13]
    encoder = yolo_box_encoder(anchor_wh , 5, featmap_size)
    decoder = yolo_box_decoder(anchor_wh , 5, featmap_size)
    '''

    feat_size = 19
    '''
    anchor_big = np.array([[116, 90], [156, 198], [373, 326]]) / 32 #* feat_size
    anchor_medium = np.array([[30,61],  [62,45],  [59,119]]) / 16 #* feat_size
    anchor_small = np.array([[10,13],  [16,30],  [33,23]]) / 8 # / feat_size
    anchor_wh = [anchor_big, anchor_medium, anchor_small]
    '''
    anchor_big = np.array([[0.127, 0.158],  [0.1574, 0.068],  [0.0452, 0.085]]) * ( feat_size)
    anchor_medium = np.array([[0.0643, 0.189],  [0.249, 0.184],  [0.0217, 0.0628]]) * (2 * feat_size)
    anchor_small = np.array([[0.0869, 0.0976],  [0.077, 0.0485],  [0.0461 , 0.0282]]) * (4 * feat_size)
    anchor_wh = [anchor_big, anchor_medium, anchor_small]


    featmap_size = [[feat_size,feat_size],[feat_size*2,feat_size*2],[feat_size*4,feat_size*4]]

    encoder = group_encoder(anchor_wh , 5, featmap_size)
    decoder = group_decoder(anchor_wh , 5, featmap_size,nms_thresh=0.9)


    dataset = COCODataset(annFile,'../datasets/jinnan2_round1_train_20190305/restricted/',True,False,transform,encoder)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=8)

    result = []
    for img,label,img_id,img_size in data_loader:

        pred_boxes, pred_conf = decoder(label)

        height,width = int(img_size[1]),int(img_size[0])


        for j in range(len(pred_boxes)):
            x1 = round(pred_boxes[j,0]*width)
            y1 = round(pred_boxes[j,1]*height)
            w =  round(pred_boxes[j,2]*width)
            h =  round(pred_boxes[j,3]*height)

            dict = {
                'image_id': int(img_id.numpy()[0]),
                'category_id': int(pred_conf[j,0])+1,
                'bbox': [x1,y1,w,h],
                'score': 1
            }
            result.append(dict)


    with open("./coco_valid.json",'w',encoding='utf-8') as json_file:
        json.dump(result,json_file,ensure_ascii=False)

    cocoGt=COCO(annFile)
    cocoDt=cocoGt.loadRes('coco_valid.json')

    imgIds=sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()