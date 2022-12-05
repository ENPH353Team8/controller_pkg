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

LINVEL = 0.1
ANGVEL = 0.264
START = 1
lin = 0


new_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/src/node/my_model.h5')
print('model loaded')

class data_collector:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    rospy.Subscriber('/R1/cmd_vel',Twist,self.gotTwistCB)

  def nav(self, frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = 200
    im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
    dim = (320, 180)
    im_rs = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)

    # cv2.imshow('img', im_rs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model_in = np.expand_dims(im_rs, axis=0)

    action = np.argmax(new_model.predict(model_in)[0])

    pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    # rate = rospy.Rate(2)
    move = Twist()
    if action == 0:
      move.linear.x = LINVEL
      move.angular.z = 0
    elif action == 1: 
      move.linear.x = 0
      move.angular.z = -1 * ANGVEL
    else:
      move.linear.x = 0
      move.angular.z = ANGVEL
    pub.publish(move)
    print(action)
  
  def pause(self):
    pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    # rate = rospy.Rate(2)
    move = Twist()
    move.linear.x = 0
    move.angular.z = 0
    pub.publish(move)
    print("paused")
    time.sleep(1)

  def stop(self, frame):
    for i in range(10):
      for pixel in frame[635+i]:
        if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
          return True
    return False

  def callback(self,data):

    global new_model
    global START
    global ANGVEL
    global LINVEL
    global lin

    if START == 1:
      pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
      move = Twist()
      move.linear.x = 0.2
      move.angular.z = 0
      time.sleep(5)
      START = 0
      return

    if lin < 0:
      LINVEL = 0.05
      ANGVEL = 0.133

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresh = 200
    im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
    dim = (320, 180)
    im_rs = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)

    # cv2.imshow('img', im_rs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model_in = np.expand_dims(im_rs, axis=0)

    action = np.argmax(new_model.predict(model_in)[0])

    pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    # rate = rospy.Rate(2)
    move = Twist()
    if action == 0:
      move.linear.x = LINVEL
      move.angular.z = 0
    elif action == 1: 
      move.linear.x = 0
      move.angular.z = -1 * ANGVEL
    else:
      move.linear.x = 0
      move.angular.z = ANGVEL
    pub.publish(move)
    print(action)
    
    # def pause():
    #   pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    #   # rate = rospy.Rate(2)
    #   move = Twist()
    #   move.linear.x = 0
    #   move.angular.z = 0
    #   pub.publish(move)
    #   print("paused")
    #   time.sleep(1)

    # def stop(frame):
    #   for i in range(10):
    #     for pixel in frame[635+i]:
    #       if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
    #         return True
    #   return False

    # if not stop(cv_image):
    #   nav()
    # else:
    #   pause()

  def gotTwistCB(loc, data):
    global linvel 
    linvel = data.linear.x
    # if not self.stop(cv_image):
    #   self.nav(cv_image)
    # else:
    #   self.pause()
    if lin < 0:
      print('paused')
      return
    else:
      self.nav(cv_image)

  def gotTwistCB(loc, data):
    global lin
    lin = data.linear.x


def main(args):
  dc = data_collector()
  rospy.init_node('data_collector', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)