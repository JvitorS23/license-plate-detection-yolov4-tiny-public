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
ap.add_argument("-y", "--yolo1", required=True,
                help="base path to YOLO vehicle directory")
ap.add_argument("-z", "--yolo2", required=True,
                help="base path to YOLO plate directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

size = 1.5
image_size = (1280, 720)


# Load the model
weightsPathVehicle = os.path.join(args["yolo1"], "yolov4-tiny-train_best.weights")
weightsPathPlate = os.path.join(args["yolo2"], "yolov4-tiny-train_best.weights")

configPath = "yolov4-tiny-test.cfg"
yoloVehicle = YoloModel(configPath, weightsPathVehicle, confidenceTresh=args['confidence'], nmsTresh=args['threshold'])
yoloPlate = YoloModel(configPath, weightsPathPlate, confidenceTresh=args['confidence'], nmsTresh=args['threshold'])

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
    
    vehicles_boxes = yoloVehicle.detect(image)

    if len(vehicles_boxes) > 0:
        
        maior_confidence_idx = 0
        for k in range(0, len(vehicles_boxes)):
            if vehicles_boxes[k][4] > vehicles_boxes[maior_confidence_idx][4]:
                maior_confidence_idx = k
    
        vehicle_bbox = vehicles_boxes[maior_confidence_idx]
        (xVehicle, yVehicle) = (vehicle_bbox[0], vehicle_bbox[1])
        (wVehicle, hVehicle) = (vehicle_bbox[2], vehicle_bbox[3])
        confVehicle = vehicle_bbox[4]

        vehicle_image = image[yVehicle:yVehicle+hVehicle, xVehicle:xVehicle+wVehicle]

        cv2.rectangle(image, (xVehicle, yVehicle), (xVehicle+wVehicle, yVehicle+hVehicle), (0,255,0) , 2)
        text = 'vehicle' + ' ' + str(confVehicle)[0:5]
        cv2.putText(image , text, (xVehicle, yVehicle - 5), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2)
        cv2.putText(image, img_name, (10, 40), cv2.FONT_HERSHEY_DUPLEX, size, (0,255,0), 2)
        plates_boxes = yoloPlate.detect(vehicle_image)

        if len(plates_boxes) > 0:
            maior_confidence_idx = 0
            for k in range(0, len(vehicles_boxes)):
                if plates_boxes[k][4] > plates_boxes[maior_confidence_idx][4]:
                    maior_confidence_idx = k

            plate_bbox = plates_boxes[maior_confidence_idx]
            (xPlate, yPlate) = (plate_bbox[0], plate_bbox[1])
            (wPlate, hPlate) = (plate_bbox[2], plate_bbox[3])
            confPlate = plate_bbox[4]

            cv2.rectangle(vehicle_image, (xPlate, yPlate), (xPlate+wPlate, yPlate+hPlate), (0,255,0) , 2)
            text = 'plate' + ' ' + str(confPlate)[0:5]
            cv2.putText(vehicle_image , text, (xPlate, yPlate - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 2)
            # vehicle_image = cv2.resize(vehicle_image, (416, 416))
            cv2.imshow('vehicle-image', vehicle_image)
    
    cv2.imshow('input-image', cv2.resize(image, image_size))
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
