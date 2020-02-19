import cv2
import numpy as np
from utils import*
model_weights="yolov3-wider_16000.weights"
model_cfg="yolov3-face.cfg"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
dict = {}
def demographics(frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))
         # Remove the bounding boxes with low confidence
        faces, genders, ages, confidences = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        male_count = 0
        female_count = 0
        details=[]
        no_of_faces = len(faces)
        boolen = False
        for i in range(len(faces)):
           boolen=True
           Dict={}
           Dict["person_id"]=i+1
           Dict["confidence"]=round(confidences[i],2)
           Dict["gender"]=genders[i]
           Dict["age_range"]=ages[i]
           details.append(Dict)
           if(genders[i]=="Male"):
               male_count+=1
           else:
               female_count+=1
        dict["Is_faces_detected"] = boolen
        dict["Detected_Faces"] = no_of_faces
        dict["Males:"] = male_count
        dict["Females:"] = female_count
        dict["Demographics"] = {}
        dict["Demographics"]["persons_details"] = {}
        dict["Demographics"]["persons_details"] = details
        info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]
        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        return dict
