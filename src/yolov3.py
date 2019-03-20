import torch
import torch.nn as nn
from src.layers import conv_sets,pred_module_v3,up_sample
from src.box_coder import group_decoder,gen_yolo_box
from src.loss import yolov3_loss
from src.darknet53 import darknet53
from src.utils import load_conv_bn,load_conv
from collections import Counter



class YOLOv3(nn.Module):

    def __init__(self,basenet,anchor,do_detect,featmap_size,cls_num=20):
        super(YOLOv3, self).__init__()

        self.cls_num = cls_num
        self.feature = basenet
        self.bbox_num = len(anchor[0])

        self.convsets1 = conv_sets(1024,1024,512)
        self.pred_large_obj = pred_module_v3(512,1024,self.cls_num,self.bbox_num)

        self.up_sample1 = up_sample(512,256)
        self.convsets2 = conv_sets(512+256, 512, 256)
        self.pred_med_obj = pred_module_v3(256,512, self.cls_num, self.bbox_num)

        self.up_sample2 = up_sample(256,128)
        self.convsets3 = conv_sets(256+128, 256, 128)
        self.pred_small_obj = pred_module_v3(128,256, self.cls_num, self.bbox_num)

        self.do_detect = do_detect

        self.loss = []


        if self.do_detect:
            self.decoder = group_decoder(anchor, cls_num, featmap_size, conf=0.1)
        else:
            for i in range(len(anchor)):
                self.loss.append(yolov3_loss(anchor[i],featmap_size[i],l_coord=3, object_scale=1, noobject_scale=1))

    def load_part(self,buf,start,part):
        for idx,m in enumerate(part.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf,start,conv,bn)
        return start

    def load_weight(self,weight_file):

        if weight_file is not None:
            print("Load pretrained models !")

            fp = open(weight_file, 'rb')
            header = np.fromfile(fp, count=5, dtype=np.int32)
            header = torch.from_numpy(header)
            print(header)
            buf = np.fromfile(fp, dtype = np.float32)

            start = 0
            start = self.load_part( buf, start, self.feature)
            print(start, buf.shape[0])

            #=================================================
            start = self.load_part(buf, start, self.convsets1)
            start = self.load_part(buf, start, self.pred_large_obj.extra_layer)
            start = load_conv(buf, start, self.pred_large_obj.detect_layer)

            print(start, buf.shape[0])
            # =================================================
            start = self.load_part(buf, start, self.up_sample1)
            start = self.load_part(buf, start, self.convsets2)
            start = self.load_part(buf, start, self.pred_med_obj.extra_layer)
            start = load_conv(buf, start, self.pred_med_obj.detect_layer)
            print(start, buf.shape[0])

            # =================================================
            start = self.load_part(buf, start, self.up_sample2)
            start = self.load_part(buf, start, self.convsets3)
            start = self.load_part(buf, start, self.pred_small_obj.extra_layer)
            start = load_conv(buf, start, self.pred_small_obj.detect_layer)
            print(start,buf.shape[0])

    def forward(self,x,target=None):
        B = x.size(0)
        feats = self.feature(x)

        pred = []

        layer1 = self.convsets1(feats[2])
        pred.append(self.pred_large_obj(layer1))

        layer2 = self.up_sample1(layer1)
        layer2 = torch.cat((layer2,feats[1]),1)
        layer2 = self.convsets2(layer2)
        pred.append(self.pred_med_obj(layer2))



        layer3 = self.up_sample2(layer2)
        layer3 = torch.cat((layer3,feats[0]),1)
        layer3 = self.convsets3(layer3)
        pred.append(self.pred_small_obj(layer3))


        if self.detect:
            pred = self.decoder(pred)
        else:
            loss_info = {}
            loss = 0
            for i in range(len(pred)):
                loss_temp,loss_info_temp = self.loss[i](pred[i],target[i])
                loss += loss_temp
                if i == 0:
                    loss_info.update(loss_info_temp)
                else:
                    for k, v in loss_info_temp.items():
                        loss_info[k] += v


            loss_info['mean_iou'] /= len(pred)
            loss_info['recall_50'] /= len(pred)
            loss_info['recall_75'] /= len(pred)

            return loss,loss_info

            return pred

def build_yolov3(cls_num, anchor, featmap_size, do_detect=True, pretrained=None):
    basenet = darknet53()
    basenet.load_weight(pretrained)
    net = YOLOv3(basenet,anchor,do_detect,featmap_size,cls_num)

    return net
if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms
    import cv2
    import numpy as np

    transform = transforms.Compose([
        transforms.Resize([608, 608]),
        transforms.ToTensor(),
    ])


    anchor_big = np.array([[116, 90], [156, 198], [373, 326]]) / 32
    anchor_medium = np.array([[30,61],  [62,45],  [59,119]]) / 16
    anchor_small = np.array([[10,13],  [16,30],  [33,23]]) / 8
    anchor = [anchor_big, anchor_medium, anchor_small]

    feat_size = 19
    feat = [[feat_size,feat_size],[feat_size*2,feat_size*2],[feat_size*4,feat_size*4]]

    net = build_yolov3(80,anchor,feat)
    net.load_weight('../yolov3.weights')
    net.cuda()
    net.eval()

    img = cv2.imread('dog.jpg')
    output_image = img.copy()
    width = img.shape[1]
    height = img.shape[0]

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = transform(img).unsqueeze(0).cuda()

    pred_boxes, pred_conf = net(img)


    for j in range(len(pred_boxes)):

        x1 = pred_boxes[j,0]
        y1 = pred_boxes[j,1]
        x2 = x1 + pred_boxes[j,2]
        y2 = y1 + pred_boxes[j,3]

        x1,x2 = int(x1*width),int(x2*width)
        y1,y2 = int(y1*height),int(y2*height)
        cls_id = int(pred_conf[j,0])
        scores = pred_conf[j,1]
        color = (0,0,255)

        cv2.rectangle(output_image, (x1+1, y1+1), (x2+1, y2+1), color, 2)

        text_size = cv2.getTextSize(str(cls_id+1) + ' : %.2f' % scores, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(output_image, (x1, y1), (x1 + text_size[0] + 3, y1 + text_size[1] + 4), color, -1)
        cv2.putText(
            output_image, str(cls_id+1) + ' : %.2f' % scores,
            (x1, y1 + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)
        print(x1,x2,y1,y2)

    cv2.imwrite('result.jpg',output_image)


