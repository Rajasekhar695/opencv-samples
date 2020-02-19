# import the necessary packages
import numpy as np
import time as t
import cv2
import os
#from Combine_Yolo import*
confidence = 0.5
threshold = 0.3
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialize the video stream, pointer to output video file, and
# frame dimensions
writer = None
def animal(frame):
    (W, H) = frame.shape[:2]
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    confs =[]
    total = {}
    Dict = {}
    animalfound=0
    count = 0
    details=[]
    animals=[]
    boolen=False
    # loop over each of the layer outputs
    for output in layerOutputs:
        animal = ""
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence1 = scores[classID]

            if confidence1 > confidence and 24 > classID >= 15:
                animalfound+=1
                id=classID
                animal = (LABELS[id])
                #for animal in total.keys():
                if animal not in total:
                    total[animal] = 1
                else:
                    total[animal] += 1
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
                confidences.append(float(confidence1))
                classIDs.append(classID)
                animals.append(animal)
                boolen=True
    Dict["Is_animal_detected"]=boolen
    Dict["detected_animals"]=animalfound
    Dict["animals_list"]=total
    for i in range(animalfound):
        final = {}
        final["animal_type"] = animals[i]
        final["confidence"] = round(confidences[i], 2)
        details.append(final)
    Dict["animals_details"]=details
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                 confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Dict
