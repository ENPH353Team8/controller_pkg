#! /usr/bin/env python3
from __future__ import print_function

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys

from scipy import stats as st
import numpy as np
import cv2 as cv

import rospy
import copy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
cv2.__version__
from geometry_msgs.msg import Twist
from random import randrange

from std_msgs.msg import String

from matplotlib import pyplot as plt

import os

import tensorflow as tf
from tensorflow import keras

# Load images at start
TEST_IMAGE = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/blurryP.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
TEST_IMAGE2 = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/blurryp2.png", cv2.IMREAD_GRAYSCALE)

letter_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller/src/node/letters4.h5')
number_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller/src/node/numbers8.h5')
parking_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller/src/node/parkingID.h5')

predictions = {"1": [[], [], [], []], "2": [[], [], [], []], "3": [[], [], [], []], "4": [[], [], [], []], "5": [[], [], [], []], "6": [[], [], [], []], "7": [[], [], [], []], "8": [[], [], [], []],}

# complete SIFT operations on images before to lower computation time each loop
# Features
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(TEST_IMAGE, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Returns coordinates of four corners of a selected image in a frame
# Returns None if the image is not located inside the frame
def DoSIFT(img, frame):
  grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
  kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
  matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
  good_points = []
  for m, n in matches:
    # increased threshold to 0.8 to increase chance of finding 'P'
      if m.distance < 0.8 * n.distance:
          good_points.append(m)

  query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
  train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    
  matrix = None

  try:
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    
    h, w = img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    siftPts = cv2.perspectiveTransform(pts, matrix)

    return siftPts
  except Exception:
    return None

def get_actual_point(pts, i):
    return [pts[i][0][0], pts[i][0][1]]

# Retrieve the coordinates from points from the data structure returned by SIFT
def cleanPoints(pts):

  p1 = get_actual_point(pts, 0)
  p2 = get_actual_point(pts, 1)
  p3 = get_actual_point(pts, 2)
  p4 = get_actual_point(pts, 3)

  return [p1, p2, p3, p4]

# Returns the average value of a component of a set of vectors
def averageCoord(points, dim):
  sum = 0
  for i in range(len(points)):
    sum += points[i][dim]
  return sum/len(points)

# Returns the average of 3 numbers
def averageNumber(x, y, z):
  sum = x + y + z
  return sum/3

# checks if the blue shade of interest is present on a particular pixel
def checkBlue(B, G, R):
  
  if ((B >= 190 and B <= 220) or (B >= 90 and B <= 130)) and ((G <= 20 and R <= 20)):
    return True
  
  elif (B >= 190 and B <= 220) and (G >= 90 and G <= 130) and (R <= 90 and R >= 105):
    return True
  
  elif (B >= 190 and B <= 210) and (G >= 90 and G <= 110) and (R >= 90 and R <= 110):
    return True
  
  else:
    return False

#checks whether the SIFT prediction occurs within a parked car
def checkValidity(row, numCols, homography):
  for col in range(len(homography[0])):
    B = homography[row][col][0]
    G = homography[row][col][1]
    R = homography[row][col][2]
    checkBlue(B, G, R)
    if checkBlue(B, G, R):
      return True
  return False

#returns leftmost column of license plate
def leftLimit(row, xVal, homography):
  for col in range(xVal):
    val = homography[row][xVal - col][0]
    val2 = homography[row][xVal - col][1]
    val3 = homography[row][xVal - col][2]
    if checkBlue(val, val2, val3):
      return xVal - col
  return 0

#returns rightmost column of license plate
def rightLimit(row, xVal, homography):
  for col in range(len(homography[0]) - xVal):
    val = homography[row][xVal + col - 1][0]
    val2 = homography[row][xVal + col - 1][1]
    val3 = homography[row][xVal + col - 1][2]
    if checkBlue(val, val2, val3):
      return xVal + col
  return len(homography[0])

#checks whether a pixel is part of a letter
def checkLetter(B, G, R):
  if (B >= 85 and B <= 130) and (G >= 8 and G <= 62) and (R >= 8 and R <= 62):
    return True

  if (B >= 105 and B <= 118) and (G >= 40 and G <= 62) and (R >= 40 and R <= 62):
    return True
  
  #Threshold is different for letters on the hill
  elif (B >= 170 and B <= 200) and (G >= 30 and G <= 100) and (R >= 30 and R <= 100):
    return True
  
  elif (B >= 170 and B <= 200) and (G >= 15 and G <= 25) and (R >= 15 and R <= 25):
    return True
  
  return False

#Converts one-hot-encoding of CNN model to prediction 
def prediction(y_predict):
  max = 0
  val = 0
  for i in range(len(y_predict)):
    if (y_predict[i] > max):
      max = y_predict[i]
      val = i


  val += 65
  guess = chr(val)
    
  return str(guess)

def numPrediction(y_predict):
  max = 0
  val = 0
  for i in range(len(y_predict)):
    if (y_predict[i] > max):
      max = y_predict[i]
      val = i
  
  return str(val)

def parkID_Prediction(y_predict):
  max = 0
  val = 0
  for i in range(len(y_predict)):
    if (y_predict[i] > max):
      max = y_predict[i]
      val = i + 1
  
  return str(val)

# Transforms image to be ready for CNN
def img_transform(img, dim, threshold):
  scale1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  gray1 = cv2.cvtColor(scale1, cv2.COLOR_RGB2GRAY)
  _, sizedL1 = cv2.threshold(gray1, threshold, 255, cv2.THRESH_BINARY)
  return sizedL1

# Checks if plate image is valid
def plateValidity(plate):
  if (len(plate) > 20):
    return False
  row = int(len(plate)/2)
  for i in range(int(len(plate[0])/4)):
    if plate[row][i][0] < 10:
      return True
    
  return False

def checkID(B, G, R):
  if B <= 65 and G <= 65 and R <= 65:
    return True

def processParkingID(parkingID):
  for row in range(len(parkingID)):
    for col in range(len(parkingID[0])):
      pixel = parkingID[row][col]
      val = pixel[0]
      val2 = pixel[1]
      val3 = pixel[2]
    if checkID(val, val2, val3):
      parkingID[row][col] = [0,0,0]
    else:
      parkingID[row][col] = [250, 250, 250]

  return parkingID





class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',Image, self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    ## Display the output of the camera on the robot
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(1)

    frame = cv_image

    siftPoints = np.zeros([4, 2])

    homography = frame

    #chooseImg = np.random.randint(0, 2)
    #print(chooseImg)

    # if chooseImg == 0:
    #   img = TEST_IMAGE
    # elif chooseImg == 1:
    #   img = TEST_IMAGE2

    # Image is processed with SIFT to locate letter 'P':
    sift = DoSIFT(TEST_IMAGE, frame)
    if sift is None:
      return
    siftPoints = cleanPoints(sift)

    # If SIFT finds the P, SIFT is performed 2 more times
    sift2 = DoSIFT(TEST_IMAGE, frame)
    siftPoints2 = cleanPoints(sift2)

    sift3 = DoSIFT(TEST_IMAGE, frame)
    siftPoints3 = cleanPoints(sift3)

    # Coordinates of each SIFT iteration are averaged for the centre location of the P
    avgX1 = averageCoord(siftPoints, 0)
    avgY1 = averageCoord(siftPoints, 1)

    avgX2 = averageCoord(siftPoints2, 0)
    avgY2 = averageCoord(siftPoints2, 1)

    avgX3 = averageCoord(siftPoints3, 0)
    avgY3 = averageCoord(siftPoints3, 1)

    # Average location of P is used
    averageX = averageNumber(avgX1, avgX2, avgX3)
    averageY = averageNumber(avgY1, avgY2, avgY3)

    valid = False
    
    # Checks whether the location of the P is within a license plate
    try:
      valid = checkValidity(int(averageY), len(homography[0]), homography)

    except Exception:
      valid = True

    if valid:
      # cv2.circle(img=homography, center = (int(averageX), int(averageY)), radius =20, color =(255,0,0), thickness=-1)
      # cv2.imshow("Homography", homography)
      # cv2.waitKey(1)

      # Calculates left and right endpoints of a license plate
      x1 = leftLimit(int(averageY), int(averageX), homography)
      x2 = rightLimit(int(averageY), int(averageX), homography)

      frame_copy = copy.deepcopy(frame)

      subframe = frame[int(averageY) - 50: int(averageY) + 100, x1: x2]

      rows1 = len(subframe)
      cols1 = len(subframe[0])

      letterFrame = frame_copy[int(averageY) - 50: int(averageY) + 100, x1: x2]

      # Filter out every pixel except pixels with the colour corresponding to the letters:
      for row in range(len(subframe)):
        for col in range(len(subframe[0])):
          val = subframe[row][col][0]
          val2 = subframe[row][col][1]
          val3 = subframe[row][col][2]
          if checkLetter(val, val2, val3):
            subframe[row][col][0] = 0
            subframe[row][col][1] = 0
            subframe[row][col][2] = 0
          else:
            subframe[row][col][0] = 250
            subframe[row][col][1] = 250
            subframe[row][col][2] = 250

      # cv2.imshow("subframe", subframe)
      # cv2.waitKey(1)


      # Filter out the left and right edges which may contain blue pixels
      for row in range(len(subframe)):
        for col in range(6):
          subframe[row][col][0] = 250
          subframe[row][col][1] = 250
          subframe[row][col][2] = 250
          subframe[row][len(subframe[0]) - 1 - col][0] = 250
          subframe[row][len(subframe[0]) - 1 - col][1] = 250
          subframe[row][len(subframe[0]) - 1 - col][2] = 250

      # Find the top and bottom of the license plate
      y_max = 0
      y_min = 0

      # Find top of plate
      for row in range(len(subframe)):
        for col in range(5, len(subframe[0]) - 5):
          val = subframe[row][col][0]
          if val == 0:
            y_max = row
        
        if y_max != 0:
          break

      # find bottom of plate
      for row in range(len(subframe)):
        for col in range(5, len(subframe[0]) - 5):
          val = subframe[len(subframe) - 1 - row][col][0]
          if val == 0:
            y_min = len(subframe) - 1 - row
        
        if y_min != 0:
          break
      
      plate = subframe[y_max - 2: y_min + 2,]
      letterFrame = letterFrame[int(y_max/4):y_max,int(cols1*0.5):cols1]

      for row in range(len(letterFrame)):
        for col in range(len(letterFrame[0])):
          pixel = letterFrame[row][col]
          val = pixel[0]
          val2 = pixel[1]
          val3 = pixel[2]
          if checkID(val, val2, val3):
            letterFrame[row][col] = [0,0,0]
          else:
            letterFrame[row][col] = [250, 250, 250]
            
      try:
        if not plateValidity(plate):
          return
      except:
        return
    
      # cv2.imshow("plate", plate)
      # cv2.waitKey(1)

      # Slice plate into each letter
      length = len(subframe[0]) 
      quad = int(length/4)

      dim = (20, 12)
      threshold = 120

      letter1 = plate[0:len(plate),0:quad]
      sizedL1 = img_transform(letter1, dim, threshold)

      letter2 = plate[0:len(plate),quad: 2*quad]
      sizedL2 = img_transform(letter2, dim, threshold)

      letter3 = plate[0:len(plate),2*quad:3*quad]
      sizedL3 = img_transform(letter3, dim, threshold)

      letter4 = plate[0:len(plate),3*quad:4*quad - 1]
      sizedL4 = img_transform(letter4, dim, threshold)

      dim = (65, 65)
      threshold = 120

      parkingID = img_transform(letterFrame, dim, threshold)
      # cv2.imshow("ID", parkingID)
      # cv2.waitKey(1)

      # photoNum1 = np.random.randint(0, 1000000)
      # photoNum2 = np.random.randint(0, 1000000)
      # photoNum3 = np.random.randint(0, 1000000)
      # photoNum4 = np.random.randint(0, 1000000)

      # cv2.imwrite("/home/fizzer/ros_ws/src/controller/src/node/parkingID/7{}.png".format(photoNum1) , parkingID)

      # cv2.imwrite("/home/fizzer/ros_ws/src/controller/src/node/more3/E{}.png".format(photoNum1) , sizedL1)
      # cv2.imwrite("/home/fizzer/ros_ws/src/controller/src/node/more3/Q{}.png".format(photoNum2) , sizedL2)
      # cv2.imwrite("/home/fizzer/ros_ws/src/controller/src/node/more3/3{}.png".format(photoNum3) , sizedL3)
      # cv2.imwrite("/home/fizzer/ros_ws/src/controller/src/node/more3/7{}.png".format(photoNum4) , sizedL4)

      y_predict1 = letter_model.predict(np.expand_dims(sizedL1, axis=0))[0]
      y_predict2 = letter_model.predict(np.expand_dims(sizedL2, axis=0))[0]
      y_predict3 = number_model.predict(np.expand_dims(sizedL3, axis=0))[0]
      y_predict4 = number_model.predict(np.expand_dims(sizedL4, axis=0))[0]

      ID_predict = parking_model.predict(np.expand_dims(parkingID, axis=0))[0]

      guess = np.array(["", "", "", ""])

      guess[0] = prediction(y_predict1)
      guess[1] = prediction(y_predict2)
      guess[2] = numPrediction(y_predict3)
      guess[3] = numPrediction(y_predict4)

      parkingID_guess = parkID_Prediction(ID_predict)

      key = predictions.get(parkingID_guess)
      key[0].append(guess[0])
      key[1].append(guess[1])
      key[2].append(guess[2])
      key[3].append(guess[3])

      print(key)

      print(guess)
      print("ParkingID: {}".format(parkingID_guess))

      strGuess = "{}{}{}{}".format(most_common(key[0]), most_common(key[1]), most_common(key[2]), most_common(key[3]))
      # strGuess = "{}{}{}{}".format(guess[0], guess[1], guess[2], guess[3])

      msg = str('Team8,gamer,{},{}'.format(parkingID_guess, strGuess))
      pub = rospy.Publisher('/license_plate', String, queue_size=1)
      pub.publish(msg)
      
    else:
      return

# From https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list 
def most_common(lst):
    return max(set(lst), key=lst.count)

# def dataprocess(subimg):

#   # plt.imshow(subimg)
#   # plt.show()
#   return findParkingID(subimg)


# def findParkingID(img):

#   for i in range(1,8):
#     pred = SIFT(i, img)

#     if isinstance(pred, int):
#       return i
    


# def SIFT(chooseImage, cv):

#   frame = cv
  
#   if (chooseImage == 1):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/1.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   elif (chooseImage == 2):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/2.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
  
#   elif (chooseImage == 3):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/3.png", cv2.IMREAD_GRAYSCALE)  # queryiamge
  
#   elif (chooseImage == 4):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/4.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   elif (chooseImage == 5):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/5.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   elif (chooseImage == 6):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/6.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   elif (chooseImage == 7):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/7.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   elif (chooseImage == 8):
#     img = cv2.imread("/home/fizzer/ros_ws/src/controller/src/node/8.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#   #thresh = 0.7

#   #if (chooseImage == 6 or chooseImage == 1 or chooseImage == 7):
#    # thresh = 0.6
  
#   cap = frame
#   # Features
#   sift = cv2.xfeatures2d.SIFT_create()
#   kp_image, desc_image = sift.detectAndCompute(img, None)
#   # Feature matching
#   index_params = dict(algorithm=0, trees=5)
#   search_params = dict()
#   flann = cv2.FlannBasedMatcher(index_params, search_params)

#   grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
#   kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
#   matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
#   good_points = []

#   for m, n in matches:
#     if m.distance < 0.7 * n.distance:
#         good_points.append(m)

#   query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
#   train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    
#   found = True

#   try:
#     matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
#     matches_mask = mask.ravel().tolist()
#   except Exception:
#     return
  
#   return chooseImage


def main(args):
  rospy.init_node('image_converter')

  ic = image_converter()
  ic.__init__
  #rospy.init_node('image_converter', anonymous=True)


  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)