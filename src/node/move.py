#! /usr/bin/env python3

#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
cv2.__version__
import numpy as np
from geometry_msgs.msg import Twist


class image_converter:

  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    ## Binarize the image to filter out all colours except the road
    # This line was from the example code from lab 2
    _, img_bin = cv2.threshold(cv_image, 119, 255, cv2.THRESH_BINARY)

    ## initialize necessary variables for detecting the line
    foundLine = True
    startOfLine = 0
    endOfLine = 0

    ## iterate horizontally across the image from bottom to top until the first row is detected with line 
    # Most of the code used to detect the midpoint of the line was from my submission of lab 2
    for row in range(799):
      ## iterate from left to right to locate the start and endpoints of the line
      for col in range(799):
          if (img_bin[799 - row][col][1] < 50):
            endOfLine = col
            if (foundLine):
                startOfLine = col
                foundLine = False
          
      if endOfLine != 0:
        rowOfCircle = row
        break


    ## The x-position of the midpoint of the line is calculated:
    colOfCircle = int(startOfLine + (endOfLine - startOfLine)/2)


    ## A circle is added to each frame of the camera on the midpoint
    # This function is used to help test whether the line is correctly detected
    cv2.circle(img=cv_image, center = (colOfCircle,798 - rowOfCircle), radius =20, color =(255,0,0), thickness=-1)

    ## Display the output of the camera on the robot
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    rate = rospy.Rate(2)
    move = Twist()

    
    move.angular.z = 0
    ## if the line is in the centre of the frame, the robot moves forward
    if (colOfCircle > 350 and colOfCircle < 450):
        move.angular.z = 0
    
    ## if the line is at the right side of the frame, the robot turns left
    if (colOfCircle < 350):
        move.angular.z = 2

    ## if the line is at the left side of the frame, the robot turns right
    if (colOfCircle > 450):
        move.angular.z = -2

    move.linear.x = 0.5

    pub.publish(move)

def move():
  rospy.init_node('pid_node')
  pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
  move = Twist()
  move.linear.x = 0.5
  pub.publish(move)

def main(args):
  while(True):
    move()
  rospy.spin()

if __name__ == '__main__':
    main(sys.argv)