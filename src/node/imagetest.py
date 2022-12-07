#!/usr/bin/env python3
from __future__ import print_function

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError

im = cv2.imread('/home/fizzer/ros_ws/src/controller_pkg/src/node/frame.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = 100
im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
print(np.shape(im_bw))
im_bw_crop = im_bw[250:400, 400:700]

cv2.imshow('img',im_bw_crop)
cv2.imshow('img2',im_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()