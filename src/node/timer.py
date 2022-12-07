#!/usr/bin/env python3

import roslib
import sys
import rospy
import time
import pickle
from std_msgs.msg import String

def main(args):
    pub = rospy.Publisher("/license_plate", String)
    rospy.init_node('scoring_manager', anonymous=True)
    rate = rospy.Rate(10)
    pub.publish(str('Team8,gamer,0,XXXX'))
    time.sleep(10)
    pub.publish(str('Team8,gamer,-1,XXXX'))

# def savePredictions(pred, filename):
#         '''
#         Save the Q state-action values in a pickle file.
#         '''
#         with open(filename + '.pickle', 'wb') as f:
#             pickle.dump(pred, f)

#         print("Wrote to file: {}".format(filename+".pickle"))

if __name__ == '__main__':
    main(sys.argv)
    predictions = {"1": [[], [], [], []], "2": [[], [], [], []], "3": [[], [], [], []], "4": [[], [], [], []], "5": [[], [], [], []], "6": [[], [], [], []], "7": [[], [], [], []], "8": [[], [], [], []],}
    # savePredictions(predictions, "predictions")
    # Store data (serialize)
    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)

    print(predictions == b)