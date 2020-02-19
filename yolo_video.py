# import the necessary packages
import numpy as np
import imutils
import time as t
import cv2
#from Combine_Yolo import*
in_confidence = 0.5
in_threshold = 0.3
dict = {}

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# loop over frames from the video file stream
def persons(frame):
	(W, H) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = t.time()
	layerOutputs = net.forward(ln)
	end = t.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	person = []
	no_of_persons = 0
	details=[]
	boolen=False
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > in_confidence and classID==0:
				no_of_persons = no_of_persons+1
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(round(float(confidence),2))
				classIDs.append(classID)
				boolen=True
			dict["Is_person_detected"]=boolen
			dict["detected_persons"] = no_of_persons

	for i in range(no_of_persons):
		Dict={}
		Dict["person_id"]=i+1
		Dict["confidence"]=confidences[i]
		details.append(Dict)

	dict["persons_details"] =details

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, in_confidence, in_threshold)
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			color = [0, 255, 0]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
			text = "person {:.4f}".format(confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return dict