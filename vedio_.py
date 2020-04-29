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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--vedio_file", type=str, default="vedio_samples/2.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="model_trained/100-epoch-air.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/air.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.cuda()
    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.vedio_file.endswith(".mp4"):
        cap = cv2.VideoCapture(opt.vedio_file)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    while cap.isOpened():
        ret, img = cap.read()
        PILimg = np.array(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        imgTensor = transforms.ToTensor()(PILimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        a.clear()
        if detections is not None:
            a.extend(detections)
        b=len(a)
        if len(a)  :
            for detections in a:
                if detections is not None:
                    detections = rescale_boxes(detections, opt.img_size, PILimg.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        print(cls_conf)
                        img = cv2.rectangle(img, (x1, y1 + box_h), (x2, y1), color, 2)
                        cv2.putText(img, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)

            print()
            print()
        #cv2.putText(img,"Hello World!",(400,50),cv2.FONT_HERSHEY_PLAIN,2.0,(0,0,255),2)
        cv2.imshow('frame', img)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()