import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
image = cv2.imread("em5.jpg")
model_weights="yolov3-wider_16000.weights"
model_cfg="yolov3-face.cfg"
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
layers_names = net.getLayerNames()

# Get the names of the output layers, i.e. the layers with unconnected outputs
ln = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(ln)

frame_height = image.shape[0]
frame_width = image.shape[1]

# Scan through all the bounding boxes output from the network and keep only
# the ones with high confidence scores. Assign the box's class label as the
# class with the highest score.
confidences = []
boxes = []
final_boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * frame_width)
            center_y = int(detection[1] * frame_height)
            width = int(detection[2] * frame_width)
            height = int(detection[3] * frame_height)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

# Perform non maximum suppression to eliminate redundant
# overlapping boxes with lower confidences.
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

for i in indices:
    i = i[0]
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    final_boxes.append(box)
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

print('# detected faces: {}'.format(len(final_boxes)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Performing emotin detection on founded faces
for i in range(len(final_boxes)):
    faces = final_boxes[i]
    (fX, fY, fW, fH) = faces
    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
    # the ROI for classification via the CNN
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)


    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    print(label)

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        # draw the label + probability bar on the canvas
        # emoji_face = feelings_faces[np.argmax(preds)]

        w = int(prob * 300)
        cv2.putText(gray, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(gray, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)

cv2.imwrite("output.jpeg", gray)
key = cv2.waitKey(1)
if key == 27 or key == ord('q'):
        print('[i] ==> Interrupted by user!')

cv2.destroyAllWindows()
print('==> All done!')