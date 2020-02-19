import cv2
import os
import sys
import math
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

def construct_firenet(x, y, training=False):
    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if (training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if (training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if (training):
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        tensorboard_verbose=2)

    return model
model = construct_firenet(224, 224, training=False)

model.load(os.path.join("models/FireNet", "firenet"), weights_only=True)
rows = 224
cols = 224
final = {}
def detect_fire(frame):
    width=frame.shape[1]
    height=frame.shape[0]
    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
    output = model.predict([small_frame])
    if round(output[0][0]) == 1:
            #cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 50)
            cv2.putText(frame, 'FIRE', (int(width / 16), int(height / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            final['Is_fire_detected'] = True

    else:
            #cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 50)
            cv2.putText(frame, 'No Fire', (int(width / 16), int(height / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
	    

            final['Is_fire_detected'] = False

    return final

