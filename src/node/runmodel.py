#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import h5py
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf
from tensorflow import keras

lin = 0
ang = 0
t = time.time()

LINVEL = 0.1
ANGVEL = 0.4

new_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/src/node/my_model2.h5')
print('model loaded')

def stop(frame):
  for i in range(3):
    for pixel in frame[638+i, 360:]:
      if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
        return True
  return False

def nav(frame):
  imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  thresh = 200
  im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
  dim = (320, 180)
  im_rs = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)

  pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
  move = Twist()

  if stop(frame):
    for i in range(6):
      move.linear.x = 0.5
      move.angular.z = 0
      time.sleep(0.1)
  else:
    model_in = np.expand_dims(im_rs, axis=0)

    action = np.argmax(new_model.predict(model_in)[0])

    if action == 1:
      move.linear.x = LINVEL
      move.angular.z = 0
    elif action == 2: 
      move.linear.x = 0
      move.angular.z = -1 * ANGVEL
    else:
      move.linear.x = 0
      move.angular.z = ANGVEL

  pub.publish(move)
  # print(action)

class data_collector:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    rospy.Subscriber('/R1/cmd_vel',Twist,self.gotTwistCB)
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def callback(self,data):

    global lin
    global ang
    global t

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # if lin < 0.01 and abs(ang) < 0.01 and (time.time()-t) > 3:
    #   print('paused')
    #   return
    # else:
    #   nav(cv_image)
    nav(cv_image)

  def gotTwistCB(loc, data):
    global lin
    global ang
    lin = data.linear.x
    ang = data.angular.z


def main(args):
  dc = data_collector()
  rospy.init_node('runmodel', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)