import os
import cv2
import sys 
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-l", "--label", type=str, required=True,
                help="label of the class")
args = vars(ap.parse_args())

label = args['label']
image_dir = args['image']

def ler_bbox(filepath):
    bbox_list = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines: 
            line = line.strip()
            box = line.split(' ')
            bbox_list.append(box)
    return bbox_list
def draw_bbox(filepath):
    bbox_list = ler_bbox(filepath.replace('.jpg', '.txt'))
    img = cv2.imread(filepath)
    image_h, image_w = img.shape[0], img.shape[1]
    for box in bbox_list:
        # extract the bounding box coordinates
        (x_center, y_center) = (int(float(box[1])*image_w), int(float(box[2])*image_h))
        (w, h) = (int(float(box[3])*image_w), int(float(box[4])*image_h))
        # Rectangle coordinates
        x_min = int(x_center - w / 2)
        y_min = int(y_center - h / 2)
        x_max = x_min+w
        y_max = y_min+h
        # draw a bounding box rectangle and label on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0) , 2)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2)
        cv2.putText(img, filepath, (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2)
    return img
filelist = os.listdir(image_dir)
img_list = [] 

for filename in filelist:
    if '.jpg' in filename: 
        img_list.append(filename)
print('====================')
print('aperte 0 para sair')
print('aperte a para voltar uma a imagem')
print('aperte qualquer tecla pra passar uma imagem')

i = 0
while i<len(img_list):
    
    img = draw_bbox(os.path.join(image_dir, img_list[i]))
    cv2.imshow('image', img)
    key = cv2.waitKey()
    if key == 113: 
        break
    
    if i !=0 and key == 97: 
        i = i - 2
       
    i = i+1    
   