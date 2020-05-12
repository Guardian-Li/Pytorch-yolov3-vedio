from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def changeBGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img


def changeRGB2BGR(img):
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()

    # RGB > BGR
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r

    return img



class Vedio():
    def __init__(self,
            vedio_file="vedio_samples/video-01.mp4",
            model_def="config/ptsc.cfg",
            weights_path="model_trained/ptsc-new-50-epoch.pth",
            class_path="data/ptsc.names",
            conf_thres=0.8,
            nms_thres=0.1,
            img_size=416):
        self.vedio_file=vedio_file
        self.model_def=model_def
        self.weights_path=weights_path
        self.class_path=class_path
        self.conf_thres=conf_thres
        self.nms_thres=nms_thres
        self.img_size=img_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(model_def, img_size=img_size).to(device)
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.weights_path))
        model.eval()
        self.model=model


    def play_vedio(self):
        classes = load_classes(self.class_path)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # if opt.vedio_file.endswith(".mp4"):
        cap = cv2.VideoCapture(self.vedio_file)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
        a = []
        time_begin = time.time()
        NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # NUM=0
        while cap.isOpened():
            ret, img = cap.read()
            if ret is False:
                break
            img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_CUBIC)
            # PILimg = np.array(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
            # imgTensor = transforms.ToTensor()(PILimg)
            RGBimg = changeBGR2RGB(img)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(Tensor))
            with torch.no_grad():
                detections = self.model(imgTensor)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            a.clear()
            if detections is not None:
                a.extend(detections)
            b = len(a)
            if len(a):
                for detections in a:
                    if detections is not None:
                        detections = rescale_boxes(detections, self.img_size, RGBimg.shape[:2])
                        unique_labels = detections[:, -1].cpu().unique()
                        n_cls_preds = len(unique_labels)
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            color = [int(c) for c in colors[int(cls_pred)]]
                            # print(cls_conf)
                            img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                            cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        color, 2)

                # print()
                # print()
            # cv2.putText(img,"Hello World!",(400,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)

            cv2.imshow('frame', changeRGB2BGR(RGBimg))
            # cv2.waitKey(0)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        time_end = time.time()
        time_total = time_end - time_begin
        print(NUM // time_total)

        cap.release()
        cv2.destroyAllWindows()






if __name__ == "__main__":
    v = Vedio()
    v.play_vedio()
