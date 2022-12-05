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
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class data_collector:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def stop(self, frame):
    for i in range(5):
      for pixel in frame[637+i]:
        if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
          return True
    return False

  def callback(self,data):

    global new_model
    global lin

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.stop(cv_image):
        pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        move = Twist()
        for i in range(10):
            move.linear.x = -0.001
            move.angular.z = 0
            pub.publish(move)
            time.sleep(0.05)



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