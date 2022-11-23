#!/usr/bin/env python3

import roslib
import sys
import rospy
import time
from std_msgs.msg import String

def main(args):
    pub = rospy.Publisher("/license_plate", String)
    rospy.init_node('scoring_manager', anonymous=True)
    rate = rospy.Rate(10)
    pub.publish(str('Team8,gamer,0,XXXX'))
    time.sleep(10)
    pub.publish(str('Team8,gamer,-1,XXXX'))

if __name__ == '__main__':
    main(sys.argv)