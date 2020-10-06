# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import random
from YoloModel import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
ap.add_argument("-l", "--label", type=str, required=True,
                help="label of the class")
args = vars(ap.parse_args())

label = args['label']
size = 1.5
image_size = (1280, 720)
if label == 'plate': 
    size = 0.5

# Load the model
weightsPath = os.path.join(args["yolo"], "yolov4-tiny-train_best.weights")
configPath = "yolov4-tiny-test.cfg"
yolo = YoloModel(configPath, weightsPath, confidenceTresh=args['confidence'], nmsTresh=args['threshold'])

# Images
filenames = os.listdir(os.path.join(args['image']))
img_list = []
for filename in filenames:
    if '.jpg' in filename:
        img_list.append(filename)
random.shuffle(img_list)

i = 0 
# loop through all the images
while i < len(img_list):
    img_name = img_list[i]
    image = cv2.imread(os.path.join(args['image'], img_name))
    boxes = yolo.detect(image)

    for box in boxes: 
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])
        conf = box[4]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0) , 2)
        text = label + ' ' + str(conf)[0:5]
        cv2.putText(image , text, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, size, (0,0,255), 2)
    
    if label == 'plate' and args["yolo"].split('/')[0] == 'model-placas-cropped': 
        image_size = (416, 416)
    image = cv2.resize(image, image_size)
    cv2.putText(image, img_name, (10, 40), cv2.FONT_HERSHEY_DUPLEX, size, (0,255,0), 2)
    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key == 113:
        break
    elif key == 97:
        if i == 1 or i == 0:
            i = -1
        else: 
            i += -2
    i += 1

cv2.destroyAllWindows()
