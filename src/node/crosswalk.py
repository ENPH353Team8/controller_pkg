# #!/usr/bin/env python3
# from __future__ import print_function

# import sys
# import rospy
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# from skimage import data, img_as_float
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error

# last_frame = np.zeros((80,120))
# first = 1
# iter = 0
# lin = 0
# ang = 0
# t = time.time()

# def stop(frame):
#     for i in range(5):
#       for pixel in frame[637+i]:
#         if pixel[2] > 200 and pixel[1] < 50 and pixel[0] < 50:
#           return True
#     return False

# def wait(frame):
#   global iter
#   global last_frame
#   global first
#   imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   pedthresh = 100
#   im_bw_ped = cv2.threshold(imgray, pedthresh, 255, cv2.THRESH_BINARY)[1]
#   dim = (320, 180)
#   im_rs_ped = cv2.resize(im_bw_ped, dim, interpolation = cv2.INTER_AREA)[50:130,100:220]
#   if first == 1:
#     last_frame = im_rs_ped
#     first = 0
#   if ssim(im_rs_ped, last_frame, full=True)[0] < 0.90:
#     iter += 1
#     if iter > 3:
#       print('movement detected')
#       pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
#       move = Twist()
#       move.linear.x = 0.2
#       move.angular.z = 0
#       pub.publish(move)
#       time.sleep(1)
#       iter = 0
#     last_frame = im_rs_ped
#     return
#   else:
#     last_frame = im_rs_ped
#     for i in range(10):
#         move.linear.x = 0
#         move.angular.z = 0
#         pub.publish(move)
#         time.sleep(0.05)
#     return

# class data_collector:

#   def __init__(self):
#     self.bridge = CvBridge()
#     rospy.Subscriber('/R1/cmd_vel',Twist,self.gotTwistCB)
#     self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

#   def callback(self,data):
#     global lin
#     global ang
#     global t

#     try:
#       cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#     except CvBridgeError as e:
#       print(e)

#     # if stop(cv_image) and (lin > 0 or abs(ang) > 0):
#     #   print('stopped')
#     #   pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
#     #   move = Twist()
#     #   for i in range(10):
#     #     move.linear.x = 0
#     #     move.angular.z = 0
#     #     pub.publish(move)
#     #     time.sleep(0.05)
#     #   wait(cv_image)

#     if lin < 0.01 and abs(ang) < 0.01 and (time.time()-t) > 3:
#       print('waiting')
#       wait(cv_image)
#     elif stop(cv_image):
#       print('stopped')
#       pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
#       move = Twist()
#       for i in range(10):
#         move.linear.x = 0
#         move.angular.z = 0
#         pub.publish(move)
#         time.sleep(0.05)

#   def gotTwistCB(loc, data):
#     global lin
#     global ang
#     lin = data.linear.x
#     ang = data.angular.z


# def main(args):
#   dc = data_collector()
#   rospy.init_node('data_collector', anonymous=True)
#   try:
#     rospy.spin()
#   except KeyboardInterrupt:
#     print("Shutting down")
#   cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main(sys.argv)