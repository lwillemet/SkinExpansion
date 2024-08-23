import sys
import cv2
import zmq
import time
import numpy as np
import threading
from threading import Lock

def recv_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

def crop_img(img, crop_start, box_size):
    return img[crop_start[0]:crop_start[0]+box_size,crop_start[1]:crop_start[1]+box_size]

context = zmq.Context()

img_socket = context.socket(zmq.REQ)
img_socket.connect("tcp://172.16.0.2:5555")
flags = 0

cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
i = 0
try:
    while True:
        start_time = time.time()
        img_socket.send_string("sup")
        img = recv_array(img_socket)
        #cv2.imwrite("video/{:05d}.png".format(i), img)
        #i += 1
        cv2.imshow('Color Image', img)
        cv2.waitKey(1)
        sleep_time = 1/30. - (time.time() - start_time)
        try:
            time.sleep(sleep_time)
        except:
            print("camera lagged")

except Exception as e:
    print(e)
