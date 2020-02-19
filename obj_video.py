import numpy as np
import cv2
import time as t
import imutils
import os
from dominant_color import *

labelsPath='coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
weightsPath='yolov3.weights'
configPath = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#vs = cv2.VideoCapture('/home/administrator/Desktop/yolo-object-detection/videos/overpass.mp4')
#vs2 = 'overpass.avi'
writer = None
'''try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1'''

def color_detection(frame):
	# read the next frame from the file
	dict={}
	(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	boxes = []
	confidences = []
	classIDs = []
	p = 0
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5:
				p = p+1
				'''if classID==2:
					p = p+1'''
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				#color_dominant(box)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#print(x, y, w, h)
			dict = color_dominant(x, y, w, h, frame)
	return dict

