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
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
linvel = 0
angvel = 0
i = 1

class data_collector:



  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    rospy.Subscriber('/R1/cmd_vel',Twist,self.gotTwistCB)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    imgray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresh = 200
    im_bw = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)[1]
    dim = (320, 180)
    im_rs = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('img', im_rs)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    global linvel
    global angvel
    global i

    if linvel > 0.1:
      action = "F"
    elif angvel < -0.3:
      action = "L"
    else:
      action = "R"

    if linvel < 0.1 and angvel < 0.3 and angvel > -0.3:
      print("skip")
    else:
      path = "/home/fizzer/data/" + str(i) + "_" + action + ".jpg"
      cv2.imwrite(path, im_rs)
      print(path)
      i += 1
      time.sleep(0.1)

  def gotTwistCB(loc, data):
    global linvel 
    linvel = data.linear.x
    global angvel 
    angvel = data.angular.z

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