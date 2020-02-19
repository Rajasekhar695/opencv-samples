import numpy as np
import imutils
import time as t
import json
import cv2
from demog import*
from no_plate_detection_video import*
from yolo_edit import*
from yolo_video import*
from Fire_detection_video import*
from obj_video import*
cap = cv2.VideoCapture("combine.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames per second : ' + str(fps))
print('Total number of frames : ' + str(frame_count))
path="/home/administrator/PycharmProjects/Internship/Intern/YOLOv3/combine/combine.mp4"
def _main():
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done")
            break
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time = round(time / 1000, 2)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        finaldict = {}
        finaldict["video_url"]=path
        for j in range(len(path)):
            if(path[j]=="."):
                finaldict["video_type"]=path[j:]
        finaldict["Frame_no"] = frame_no
        finaldict["Time"]= time
        finaldict["Detections"]={}
        finaldict["Detections"]["Person_detections"]=persons(frame)
        finaldict["Detections"]["Demographic_detections"]=demographics(frame)
        finaldict["Detections"]["Color_detections"]=color_detection(frame)
        finaldict["Detections"]["Animal_detections"]=animal(frame)
        finaldict["Detections"]["Numberplate_detections"] = number_plate(frame)
        finaldict["Detections"]["Fire_detections"] = detect_fire(frame)
        print(finaldict)
        i = i + 1
        with open('jsons/'+str(i)+'.json', 'w', encoding='utf-8') as f:
            json.dump(finaldict, f, ensure_ascii=False, indent=4)
        if(frame_no==5):
            break


if __name__ == '__main__':
   _main()