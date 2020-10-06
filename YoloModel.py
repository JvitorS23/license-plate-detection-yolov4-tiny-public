import os
import cv2
import numpy as np

class YoloModel:
	def __init__(self, configPath, weightsPath, confidenceTresh=0.25, nmsTresh=0.25):
		# carrega a yolo
		self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		layer_names = self.net.getLayerNames()
		self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		self.confidenceTresh = confidenceTresh
		self.nmsTresh = nmsTresh

	def detect(self, frame): 
		# grab the frame dimensions and convert the frame to a blob
		height, width, channels = frame.shape
		
		# Detecting objects
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

		# pass the blob through the network and obtain the detections
		# and predictions
		self.net.setInput(blob)
		outs = self.net.forward(self.output_layers)
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > self.confidenceTresh:
					# Object detected

					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences,  self.confidenceTresh, self.nmsTresh)
		finalBoxes = []
		finalConfidences = []
		if len(indexes) > 0:						
			# loop over the indexes we are keeping
			for i in indexes.flatten():		
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				finalBoxes.append([x,y,w,h,confidences[i]])
			

		return finalBoxes