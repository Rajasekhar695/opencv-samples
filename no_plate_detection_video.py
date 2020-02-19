import numpy as np
import cv2
import imutils
from PIL import Image
from pytesseract import *
# video_capture = cv2.VideoCapture(0)

# For real-time sample video detection
from utils import COLOR_RED


def number_plate(frame):
    #ret, frame = video_capture.read()
    plate_cascade = cv2.CascadeClassifier('./haarcascade_plate_number.xml')
    # frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale

    plate_rect = plate_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=7)
    dict = {}
    details=[]
    if len(plate_rect)==0:
        dict["license_plates detected"] = 0

    for (x, y, w, h) in plate_rect:
        a, b = (int(0.02 * frame.shape[0]), int(0.025 * frame.shape[1]))  # parameter tuning
        plate = frame[y + a:y + h - a, x + b:x + w - b, :]
        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 3)
        # Now crop
        Cropped = frame[y:y+h, x:x+w]
        # cv2.imshow("cropped", Cropped)
        # Read the number plate
        Cropped = cv2.resize(Cropped, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(Cropped, -1, sharpen_kernel)
        text = pytesseract.image_to_string(Cropped)

        no_of_license_plates = len(plate_rect)
        # dict[frame_no]["detected at frame number"] = no_of_license_plates
        # dict[frame_no]["detected number plate number"] = text
        boolen=False
        for i in range(len(plate_rect)):
            Dict={}
            Dict["car_id"]=i+1
            Dict["number_plate"] = text
            details.append(Dict)
            boolen=True
        dict["Is_numberplate_detected"]=boolen
        dict["detected_numberplates"] = no_of_license_plates
        dict["license_plate details"] = details

        info = [
            ('number of faces detected', '{}'.format(len(plate_rect)))
        ]
        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        # cv2.imshow("Detection", frame)
    return dict
