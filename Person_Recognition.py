'''from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread("crosswalk.jpg")

image = imutils.resize(image, width=min(400, image.shape[1]))
orig = image.copy()

(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
#print(rects)
#print(weights)

if(len(rects)>0):
    print("Persons found", str(len(rects)))

for (x, y, w, h) in rects:
	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

for (xA, yA, xB, yB) in pick:
	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("After NMS", image)
cv2.imshow("After NMS", orig)
cv2.waitKey(0)

'''#For video
import cv2
import time as t

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture("sample.avi")
frame_rate=1
prev=0
while True:
	time_elapsed = t.time() - prev
	r, frame = cap.read()
	if time_elapsed > 1. / frame_rate:
		prev = t.time()
		frame = cv2.resize(frame, (720, 720))  # Downscale to improve frame rate
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

		rects, weights = hog.detectMultiScale(gray_frame)
		frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

		if(len(rects)>0):
   			print("Persons found:", str(len(rects)))
		
		for i, (x, y, w, h) in enumerate(rects):
			if weights[i] < 0.7:
				continue
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			time = cap.get(cv2.CAP_PROP_POS_MSEC)
			time = time / 1000
			print("person detected at frame: ", rects, " frame number: ", frame_no, " and at ", time, " secs")

		cv2.imshow("preview", frame)
	k = cv2.waitKey(1)
	if k & 0xFF == ord("q"):  # Exit condition
		break
