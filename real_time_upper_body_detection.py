import cv2
import imutils
import time  as t

haar_upper_body_cascade = cv2.CascadeClassifier("data/haarcascade_upperbody.xml")

video_capture = cv2.VideoCapture("subway.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)

frame_rate=5
prev=0

print("Total number of frames:",video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#print("Frame rate per second: ",video_capture.get(cv2.CAP_PROP_FPS))

while True:
    time_elapsed=t.time()-prev
    #print(time_elapsed)
    #video_capture.set(cv2.CAP_PROP_FPS, 5)
    ret, frame = video_capture.read()
    if time_elapsed >1./frame_rate:
        prev = t.time()
        frame = imutils.resize(frame, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
        if(len(upper_body)>0):
            print("Faces found",str(len(upper_body)))
        frame_no=video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        for (x, y, w, h) in upper_body:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if upper_body.all():
                time=video_capture.get(cv2.CAP_PROP_POS_MSEC)
                time=time/1000
                print("person detected at frame: ",upper_body," frame number: ",frame_no," and at ",time," secs")
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()