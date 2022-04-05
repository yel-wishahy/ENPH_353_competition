#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt

debug_node = "test_controller_debug"

#internal debug publisher topics
debug_img_topic = "/test_controller/image"
debug_err_topic = "/test_controller/error"

FREQUENCY = 250 # #hz
MAX_ERROR = 20
ERROR_HISTORY_LENGTH = 5 #array size


def main():
    print('init node')
    rospy.init_node(debug_node)

    debug = Debug()

    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        debug.plot_err()
        debug.show_img()
        rate.sleep()

class Debug():
    def __init__(self):
        self.img_sub = rospy.Subscriber(debug_img_topic,Image,self.img_callback,queue_size=1)
        self.err_sub = rospy.Subscriber(debug_err_topic,Float32MultiArray,self.err_callback,queue_size=1)

        self.bridge = CvBridge()
        self.latest_img = Image()
        self.empty = True


        self.index = 0
        self.time_array = np.ones(ERROR_HISTORY_LENGTH)*1e-5
        self.error_array = np.ones(ERROR_HISTORY_LENGTH)*1e-5
    
    def img_callback(self,img):
        try:
            self.latest_img = np.asarray(self.bridge.imgmsg_to_cv2(img, '8UC3'))
            self.empty = False
        except CvBridgeError as e:
            print(e)

    def err_callback(self,err_pt):
        self.error_array[self.index] = err_pt.data[0]
        self.time_array[self.index] = err_pt.data[1]

        if (self.index == self.error_array.size - 1):
            self.error_array = np.roll(self.error_array, -1)
            self.time_array = np.roll(self.time_array, -1)
        else:
            self.index += 1

    def plot_err(self):
        plt.clf()
        plt.ylim((-1*MAX_ERROR,MAX_ERROR))
        plt.plot(self.time_array[:-1],self.error_array[:-1])
        plt.pause(0.000000001)

    def show_img(self):
        if( not self.empty):
            cv2.imshow("Debug View", self.latest_img)
            cv2.waitKey(1)

main()