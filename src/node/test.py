#!/usr/bin/env python3
import os
import cv2
import time

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

images = []
scores = []

for i in range(112):
    path = '/home/fizzer/ros_ws/src/controller_pkg/src/node/data/' + str(i+1) + '.jpg'
    images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))

# imgray1 = cv2.cvtColor(cv2.imread('/home/fizzer/ros_ws/src/controller_pkg/src/node/data/21.jpg'), cv2.COLOR_BGR2GRAY)
# imgray2 = cv2.cvtColor(cv2.imread('/home/fizzer/ros_ws/src/controller_pkg/src/node/data/22.jpg'), cv2.COLOR_BGR2GRAY)
for i in range(111):
    t = time.time()
    (score, diff) = ssim(images[i], images[i+1], full=True)
    scores.append(score)
    print(time.time()-t)
# for i in range(111):
#     if scores[i] < 0.95:
#         print(i)