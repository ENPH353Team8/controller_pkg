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

new_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/src/node/my_model.h5')
print('model loaded')
i = 1

last_frame = np.zeros((180, 320))

class data_collector:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def callback(self,data):

    global new_model

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    def nav():
      global last_frame
      global i
      imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      thresh = 200
      pedthresh = 150
      im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
      im_bw_ped = cv2.threshold(imgray, pedthresh, 255, cv2.THRESH_BINARY)[1]
      dim = (320, 180)
      im_rs = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)
      im_rs_ped = cv2.resize(im_bw_ped, dim, interpolation = cv2.INTER_AREA)[50:130,100:220]

      path = '/home/fizzer/ros_ws/src/controller_pkg/src/node/data/' + str(i) + ".jpg"
      cv2.imwrite(path, im_rs_ped)
      last_frame = im_rs_ped
      i += 1
      print(path)

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
      # pub.publish(move)
      # print(action)
    
    def pause():
      pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
      # rate = rospy.Rate(2)
      move = Twist()
      move.linear.x = 0
      move.angular.z = 0
      pub.publish(move)
      print("paused")
      time.sleep(1)

    def stop(frame):
      for i in range(10):
        for pixel in frame[635+i]:
          if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
            return True
      return False

    if not stop(cv_image):
      nav()
    else:
      nav()


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