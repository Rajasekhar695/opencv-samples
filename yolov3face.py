import cv2
import numpy as np
image = cv2.imread("1.jpeg")
model_weights="yolov3-wider_16000.weights"
model_cfg="yolov3-face.cfg"

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

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    text = "face detected."
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(image, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

print('# detected faces: {}'.format(len(final_boxes)))
cv2.imwrite("output.jpeg", image)
key = cv2.waitKey(1)
if key == 27 or key == ord('q'):
        print('[i] ==> Interrupted by user!')

cv2.destroyAllWindows()
print('==> All done!')