import os
import config
from torchvision import transforms

from src import build_yolo

from PIL import Image
import torch
import json
import cv2
import tqdm

torch.backends.cudnn.benchmark = True

test_path = './datasets/jinnan2_round1_train_20190305/restricted'

filename = os.listdir(test_path)
print(len(filename))


transform = transforms.Compose([
    transforms.Resize(config.YOLO['image_size']),
    transforms.ToTensor(),
])
anchor_wh = config.YOLO['anchor']

featmap_size = config.YOLO['featmap_size']



model = build_yolo(config.YOLO['class_num'], anchor_wh, featmap_size,train = False)

model.load_state_dict(torch.load('608/model_190.pkl'))
model.cuda()
model.eval()


results = []

for file in tqdm.tqdm(filename) :


    img_path = os.path.join(test_path,file)

    src = cv2.imread(img_path)
    img = Image.open(img_path)
    width,height = img.width,img.height
    img = transform(img).unsqueeze(0).cuda()

    pred_boxes, pred_conf = model(img)


    result = dict()
    result['filename'] = file
    result['rects'] = []
    for j in range(len(pred_boxes)):
        x1 = max(0, round(pred_boxes[j, 0] * width))
        y1 = max(0, round(pred_boxes[j, 1] * height))
        w =  max(0, round(pred_boxes[j, 2] * width))
        h =  max(0, round(pred_boxes[j, 3] * height))
        x2 = min(x1 + w,width-1)
        y2 = min(y1 + h,height-1)

        result_dict = {"xmin": x1, "xmax": x2, "ymin": y1, "ymax": y2 \
            , "label": int(pred_conf[j,0]+1), "confidence": float(pred_conf[j,1])}

        result['rects'].append(result_dict)

    results.append(result)

a = dict()
a['results'] = results

with open("./submit.json", 'w', encoding='utf-8') as json_file:
    json.dump(a, json_file, ensure_ascii=False)

